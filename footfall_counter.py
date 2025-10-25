# Imports
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
import pandas as pd
import os

# User Parameters
video_source = "Walking Office People.mp4"  # or 0 for webcam
output_video_file = "footfall_live_output.avi"
csv_log_file = "footfall_live_log.csv"
line_y_ratio = 0.6  # counting line at frame height
frame_resize = (320, 180)
process_every_n_frame = 2
draw_every_n_frame = 1

# Initialize Models & Tracker
model = YOLO("yolov8s.pt")
tracker = DeepSort(max_age=30, embedder=None)

# Video Capture & Output
cap = cv2.VideoCapture(video_source)
if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

orig_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 20
line_y = int(orig_height * line_y_ratio)

out = cv2.VideoWriter(output_video_file,
                      cv2.VideoWriter_fourcc(*'XVID'),
                      fps,
                      (orig_width, orig_height))

if os.path.exists(csv_log_file):
    os.remove(csv_log_file)
df_log = pd.DataFrame(columns=["frame", "IN", "OUT"])
df_log.to_csv(csv_log_file, index=False)

# Counting Variables
enter_count = 0
exit_count = 0
track_memory = {}
enter_counted_ids = set()
exit_counted_ids = set()
frame_number = 0

# Main Loop
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_number += 1

    # Skip frames for speed
    if frame_number % process_every_n_frame != 0:
        out.write(frame)
        continue

    # Resize for faster YOLO detection
    frame_small = cv2.resize(frame, frame_resize)
    results = model(frame_small, iou=iou_threshold, conf=conf_threshold, verbose=False)[0]

    detections = []
    for box in results.boxes.data.tolist():
        x1, y1, x2, y2, conf, cls = box
        if int(cls) == 0 and (x2-x1)*(y2-y1) > 200:
            scale_x = frame.shape[1] / frame_resize[0]
            scale_y = frame.shape[0] / frame_resize[1]
            x1o, y1o = int(x1*scale_x), int(y1*scale_y)
            x2o, y2o = int(x2*scale_x), int(y2*scale_y)

            w, h = x2o - x1o, y2o - y1o
            if w / h > 1.5:
                mid_x = x1o + w//2
                detections.append(([x1o, y1o, w//2, h], conf, 'person'))
                detections.append(([mid_x, y1o, w//2, h], conf, 'person'))
            else:
                detections.append(([x1o, y1o, w, h], conf, 'person'))

    # Update tracker
    tracks = tracker.update_tracks(detections, frame=frame)

    # Draw counting line
    cv2.line(frame, (0, line_y), (orig_width, line_y), (255, 0, 0), 2)

    for track in tracks:
        if not track.is_confirmed():
            continue
        x1, y1, x2, y2 = map(int, track.to_ltrb())
        track_id = track.track_id
        cx, cy = int((x1+x2)/2), int((y1+y2)/2)

        if track_id not in track_memory:
            track_memory[track_id] = cy

        prev_y = track_memory[track_id]

        # Moving down → IN
        if prev_y < line_y and cy >= line_y and track_id not in enter_counted_ids:
            enter_count += 1
            enter_counted_ids.add(track_id)

        # Moving up → OUT
        elif prev_y > line_y and cy <= line_y and track_id not in exit_counted_ids:
            exit_count += 1
            exit_counted_ids.add(track_id)

        track_memory[track_id] = cy

        # Bounding box and counts
        if frame_number % draw_every_n_frame == 0:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {track_id}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)

    # Display counts
    cv2.putText(frame, f"IN: {enter_count}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"OUT: {exit_count}", (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Write video
    out.write(frame)

    # Log every 5 frames
    if frame_number % 5 == 0:
        df_log = pd.DataFrame([[frame_number, enter_count, exit_count]],
                              columns=["frame", "IN", "OUT"])
        df_log.to_csv(csv_log_file, mode='a', index=False, header=False)

    # Display
    try:
        cv2.imshow("Footfall Counter", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break
    except:
        pass

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Final Count: IN = {enter_count}, OUT = {exit_count}")
print(f"Video saved: {output_video_file}, CSV log: {csv_log_file}")
