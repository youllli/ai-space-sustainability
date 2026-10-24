# demo.py

from pathlib import Path

from tracking_pipeline import run_tracking
from compute_metrics import compute_metrics


# ==========================================
# 1. Configure directory paths
# ==========================================

# Directory containing demo.py: testCode/
BASE_DIR = Path(__file__).resolve().parent

# Common parent directory of testCode/ and ultralytics/
PROJECT_DIR = BASE_DIR.parent

# Directory containing the input videos
VIDEO_DIR = PROJECT_DIR / "ultralytics" / "assets"

# Path to the YOLO model
MODEL_PATH = BASE_DIR / "yolov8n.pt"

# Directory for analysis outputs
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. Specify the videos to analyze
# ==========================================

VIDEO_LIST = [
    VIDEO_DIR / "libraryCCTV.mp4",
    VIDEO_DIR / "mallCCTV.mp4",
]


# ==========================================
# 3. YOLO tracking parameters
# ==========================================

CONFIDENCE = 0.01
IOU_THRESHOLD = 0.45


# ==========================================
# 4. Main program
# ==========================================

def main():

    print("=" * 50)
    print("AI-Based Indoor People-Flow Analysis and Sustainability Management System")
    print("=" * 50)

    # Check whether the model file exists
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"YOLO model not found: {MODEL_PATH}"
        )

    # Check whether both video files exist
    for video_path in VIDEO_LIST:
        if not video_path.is_file():
            raise FileNotFoundError(
                f"Video not found: {video_path}"
            )

    # Process each video in sequence
    for index, video_path in enumerate(VIDEO_LIST, start=1):

        print("\\n" + "=" * 50)
        print(f"Starting analysis of video {index}")
        print(f"Video name: {video_path.name}")
        print("=" * 50)

        # Use the video filename to create output filenames
        video_name = video_path.stem

        tracking_csv = (
            OUTPUT_DIR / f"{video_name}_tracking.csv"
        )

        metrics_csv = (
            OUTPUT_DIR / f"{video_name}_metrics.csv"
        )

        # ----------------------------------
        # Step 1: Person tracking
        # ----------------------------------

        print("\\n[1/2] Running YOLO person tracking...")

        run_tracking(
            video_path=video_path,
            model_path=MODEL_PATH,
            output_csv=tracking_csv,
            conf=CONFIDENCE,
            iou=IOU_THRESHOLD,
        )

        # ----------------------------------
        # Step 2: Calculate people-flow metrics
        # ----------------------------------

        print("\\n[2/2] Calculating people-flow metrics...")

        compute_metrics(
            csv_path=tracking_csv,
            output_report=metrics_csv,
        )

        print(f"\\nAnalysis completed for {video_path.name}!")
        print(f"Tracking data: {tracking_csv}")
        print(f"Metrics report: {metrics_csv}")

    print("\\n" + "=" * 50)
    print("Analysis of both videos is complete!")
    print(f"All results are stored in: {OUTPUT_DIR}")
    print("=" * 50)


# ==========================================
# 5. Run the program
# ==========================================

if __name__ == "__main__":
    main()
