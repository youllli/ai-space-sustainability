# tracking_pipeline.py

from pathlib import Path

import cv2
import pandas as pd
from ultralytics import YOLO


def run_tracking(
    video_path,
    model_path="yolov8n.pt",
    output_csv="tracking_data.csv",
    conf=0.01,
    iou=0.45,
):
    """
    Convert a video into structured people-flow tracking data
    and export it as a CSV file.

    CSV columns:
        frame      : Frame number
        person_id  : ByteTrack tracking ID
        x          : X coordinate of the person's bounding-box center
        y          : Y coordinate of the person's bounding-box center
        time       : Video timestamp in seconds
    """

    # 1. Check whether the files exist
    video_path = Path(video_path)
    model_path = Path(model_path)

    if not video_path.exists():
        raise FileNotFoundError(
            f"Video not found: {video_path.resolve()}"
        )

    if not model_path.exists():
        raise FileNotFoundError(
            f"YOLO model not found: {model_path.resolve()}"
        )

    # 2. Read video information
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        cap.release()
        raise RuntimeError(
            f"Unable to open video: {video_path}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_video_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    cap.release()

    if fps <= 0:
        raise ValueError(
            "Could not obtain a valid video FPS. Please check the video file."
        )

    video_duration = total_video_frames / fps

    print("=" * 55)
    print("Starting YOLO + ByteTrack")
    print(f"Video: {video_path}")
    print(f"FPS: {fps:.2f}")
    print(f"Total video frames: {total_video_frames}")
    print(f"Video duration: approximately {video_duration:.2f} seconds")
    print(f"Confidence threshold: {conf}")
    print("=" * 55)

    # 3. Load the YOLO model
    model = YOLO(str(model_path))

    # 4. Run person detection and tracking
    results = model.track(
        source=str(video_path),
        tracker="bytetrack.yaml",
        classes=[0],
        conf=conf,
        iou=iou,
        persist=True,
        save=True,
        stream=True,
        verbose=False,
    )

    # 5. Collect structured data
    data = []
    frame_num = 0

    for r in results:

        frame_num += 1

        # Calculate the timestamp using the video's actual FPS
        timestamp = (frame_num - 1) / fps

        # No tracking IDs were obtained for this frame
        if r.boxes.id is None:
            continue

        ids = r.boxes.id.cpu().numpy()
        boxes = r.boxes.xyxy.cpu().numpy()

        # 6. Collect each person's coordinates and tracking ID
        for box, track_id in zip(boxes, ids):

            x1, y1, x2, y2 = box

            # Calculate the center of the person's bounding box
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            data.append({
                "frame": frame_num,
                "person_id": int(track_id),
                "x": center_x,
                "y": center_y,
                "time": timestamp,
            })

        # Display progress every 100 frames
        if frame_num % 100 == 0:
            print(
                f"Processing progress: {frame_num} / "
                f"{total_video_frames} frames"
            )

    # 7. Create a DataFrame
    columns = [
        "frame",
        "person_id",
        "x",
        "y",
        "time",
    ]

    df = pd.DataFrame(data, columns=columns)

    # 8. Export the CSV file
    output_path = Path(output_csv)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("\\nTracking completed!")
    print(f"Frames processed: {frame_num}")
    print(f"Number of tracking records: {len(df)}")
    print(f"Number of unique tracking IDs: {df['person_id'].nunique()}")
    print(f"CSV saved to: {output_path.resolve()}")

    if not df.empty:
        print("\\nData preview:")
        print(df.head())
