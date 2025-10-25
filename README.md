# Footfall Counter with YOLOv8 + DeepSORT

## Description
This project counts people entering and exiting a defined area using YOLOv8 for detection and DeepSORT for tracking.

## Video Source
- Input video:"original video", src = "https://youtu.be/BFFZ6PZjM_E?si=gj-Wy5xuqMekivwt"
- Alternatively, use a webcam by setting `video_source = 0`.

## Counting Logic
- A horizontal line is drawn at 60% of frame height.
- If a person's centroid moves **down across the line → counted as IN**.
- If a person's centroid moves **up across the line → counted as OUT**.
- Each person is counted only once using `track_id`.

## Dependencies
- Python 3.10+
- OpenCV (`pip install opencv-python`)
- pandas (`pip install pandas`)
- ultralytics (`pip install ultralytics`)
- deep_sort_realtime (`pip install deep_sort_realtime`)

## Setup
1. Install dependencies.
2. Place the video file in the same folder as script.
3. Run the script:
   ```bash
   python footfall_counter.py
