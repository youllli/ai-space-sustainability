# compute_metrics.py

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path


# ============================================================
# Configurable parameters
# ============================================================

MAX_CAPACITY = 30
TOTAL_SEATS = 20

CONGESTION_THRESHOLD = 80


def compute_metrics(
    csv_path="tracking_data.csv",
    output_report="metrics_report.csv",
):
    """
    Read tracking_data.csv, calculate three core metrics,
    and export the results to metrics_report.csv.
    """

    # --------------------------------------------------------
    # 1. Load data and perform basic validation
    # --------------------------------------------------------

    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Tracking data not found: {csv_path.resolve()}"
        )

    df = pd.read_csv(csv_path)

    required_columns = {
        "frame",
        "person_id",
        "x",
        "y",
        "time",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Required columns are missing from the CSV: {missing_columns}"
        )

    if df.empty:
        raise ValueError(
            "The tracking data is empty. Metrics cannot be calculated."
        )

    if MAX_CAPACITY <= 0:
        raise ValueError("MAX_CAPACITY must be greater than 0.")

    if TOTAL_SEATS <= 0:
        raise ValueError("TOTAL_SEATS must be greater than 0.")

    # Sort by frame and person ID
    df = df.sort_values(
        ["frame", "person_id"]
    ).reset_index(drop=True)

    video_duration = float(df["time"].max())
    total_frames = int(df["frame"].max())

    print("=" * 55)
    print("AI People-Flow Analysis System — Core Metrics Report")
    print("=" * 55)

    print(f"Last observed timestamp: {video_duration:.2f} seconds")
    print(f"Last observed frame: {total_frames}")
    print()

    # --------------------------------------------------------
    # Metric 1: Average dwell time
    # --------------------------------------------------------
    # Formula:
    # Average_Dwell_Time = (1/n) * sum(T_out,i - T_in,i)
    #
    # This implementation calculates the last observed time
    # minus the first observed time for each tracking ID.
    # If tracking is interrupted, this may not equal the
    # person's actual dwell time.
    # --------------------------------------------------------

    dwell_df = (
        df.groupby("person_id")["time"]
        .agg(
            T_in="min",
            T_out="max",
        )
    )

    dwell_df["dwell_time"] = (
        dwell_df["T_out"] - dwell_df["T_in"]
    )

    # Percentiles are exploratory reference thresholds,
    # not validated criteria for detecting anomalies.
    avg_dwell = float(
        dwell_df["dwell_time"].mean()
    )

    pr25 = float(
        np.percentile(
            dwell_df["dwell_time"], 25
        )
    )

    pr75 = float(
        np.percentile(
            dwell_df["dwell_time"], 75
        )
    )

    print("[Metric 1: Average Dwell Time]")

    print(
        f"Number of tracking IDs: {len(dwell_df)}"
    )

    print(
        f"Average dwell time: {avg_dwell:.2f} seconds"
    )

    print(
        f"PR25 (exploratory lower reference): {pr25:.2f} seconds"
    )

    print(
        f"PR75 (exploratory upper reference): {pr75:.2f} seconds"
    )

    print(
        "Note: These percentiles describe the distribution in this video. "
        "By themselves, they do not prove that the space lacks appeal "
        "or that abnormal occupancy has occurred."
    )

    print()

    # --------------------------------------------------------
    # Metric 2: Congestion rate
    # --------------------------------------------------------
    # Formula:
    # Congestion_Rate_t = (N_t / C_max) * 100%
    #
    # Current implementation:
    # Counts unique tracking IDs in each frame across the
    # entire image. It does not yet restrict the calculation
    # to a specific region of interest (ROI).
    # --------------------------------------------------------

    people_per_frame = (
        df.groupby("frame")["person_id"]
        .nunique()
    )

    congestion_per_frame = (
        people_per_frame / MAX_CAPACITY
    ) * 100

    avg_congestion = float(
        congestion_per_frame.mean()
    )

    max_congestion = float(
        congestion_per_frame.max()
    )

    peak_frame = int(
        congestion_per_frame.idxmax()
    )

    peak_rows = df[
        df["frame"] == peak_frame
    ]

    peak_time = float(
        peak_rows["time"].iloc[0]
    )

    print("[Metric 2: Congestion Rate]")

    print(
        f"Configured maximum capacity: {MAX_CAPACITY} people"
    )

    print(
        f"Average congestion rate: {avg_congestion:.2f}%"
    )

    print(
        f"Maximum congestion rate: {max_congestion:.2f}%"
    )

    print(
        f"Alert threshold: {CONGESTION_THRESHOLD}%"
    )

    congestion_warning = (
        max_congestion >= CONGESTION_THRESHOLD
    )

    if congestion_warning:
        print(
            f"Result: The high-load alert condition has been reached; "
            f"frame {peak_frame}, at {peak_time:.2f} seconds."
        )
    else:
        print(
            "Result: The configured alert threshold was not reached "
            "in this dataset."
        )

    print(
        "Note: This is an estimate based on the full frame. "
        "The calculation is not yet restricted to a specific ROI."
    )

    print()

    # --------------------------------------------------------
    # Metric 3: Space turnover rate
    # --------------------------------------------------------
    # Formula:
    # Turnover_Rate = U_total / S_total
    #
    # U_total: Number of unique tracking IDs during the observation
    # S_total: Total number of seats
    #
    # Note: The number of tracking IDs may not equal the actual
    # number of people who used the space.
    # --------------------------------------------------------

    u_total = int(
        df["person_id"].nunique()
    )

    turnover_rate = (
        u_total / TOTAL_SEATS
    )

    print("[Metric 3: Space Turnover Rate]")

    print(
        f"Configured number of seats: {TOTAL_SEATS}"
    )

    print(
        f"Number of unique tracking IDs: {u_total}"
    )

    print(
        f"Space turnover rate: {turnover_rate:.2f}"
    )

    print(
        "Note: This is the number of tracking IDs divided by the number "
        "of seats for a single video. It is not a validated measure "
        "of actual seat-turnover events."
    )

    print()

    # --------------------------------------------------------
    # 4. Compile the report
    # --------------------------------------------------------

    summary = {
        "Last Observed Time (seconds)": round(
            video_duration, 2
        ),
        "Last Observed Frame": total_frames,
        "Number of Tracking IDs": int(
            df["person_id"].nunique()
        ),
        "Average Dwell Time (seconds)": round(
            avg_dwell, 2
        ),
        "PR25 Dwell Time (seconds)": round(
            pr25, 2
        ),
        "PR75 Dwell Time (seconds)": round(
            pr75, 2
        ),
        "Configured Maximum Capacity": MAX_CAPACITY,
        "Average Congestion Rate (%)": round(
            avg_congestion, 2
        ),
        "Maximum Congestion Rate (%)": round(
            max_congestion, 2
        ),
        "Congestion Assessment": (
            "High-load alert"
            if congestion_warning
            else "Alert threshold not reached"
        ),
        "Configured Number of Seats": TOTAL_SEATS,
        "Number of Unique Tracking IDs": u_total,
        "Space Turnover Rate": round(
            turnover_rate, 2
        ),
    }

    summary_df = pd.DataFrame(
        [summary]
    )

    output_path = Path(output_report)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("=" * 55)

    print(
        f"Metrics report saved to: {output_path.resolve()}"
    )

    # Generate four analysis plots
    video_name = csv_path.stem

    if video_name.endswith("_tracking"):
        video_name = video_name[:-len("_tracking")]

    generate_analysis_plots(
        df=df,
        output_dir=output_path.parent,
        video_name=video_name,
    )

    print("=" * 55)


def generate_analysis_plots(df, output_dir, video_name):
  """Generate 4 core visualization charts based on tracking_data.csv:

  1. crowd_over_time.png (Crowd Count over Time)
  2. clustering_over_time.png (Clustering Degree over Time)
  3. ExplorationGrid.png (Movement Exploration Grid Map)
  4. Heatmap.png (Personnel Position Heatmap)
  """
  output_dir = Path(output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)

  # ==========================================
  # 1. Crowd Count over Time (crowd_over_time)
  # ==========================================
  people_per_frame = df.groupby("frame")["person_id"].nunique().reset_index()
  people_per_frame.columns = ["frame", "count"]

  plt.figure(figsize=(10, 5))
  plt.plot(
      people_per_frame["frame"],
      people_per_frame["count"],
      color="#1f77b4",
      linewidth=1.5,
  )
  plt.title("Crowd Count Over Time", fontsize=14, fontweight="bold")
  plt.xlabel("Frame", fontsize=12)
  plt.ylabel("Number of People", fontsize=12)
  plt.grid(True, linestyle="--", alpha=0.6)
  plt.tight_layout()

  crowd_plot_path = output_dir / f"{video_name}_crowd_over_time.png"
  plt.savefig(crowd_plot_path, dpi=300)
  plt.close()

  # ==========================================
  # 2. Clustering Degree over Time (clustering_over_time)
  # ==========================================
  max_cap = 30
  clustering_pct = (people_per_frame["count"] / max_cap) * 100
  clustering_pct = np.clip(clustering_pct * 1.5, 0, 100)

  plt.figure(figsize=(10, 5))
  plt.plot(
      people_per_frame["frame"],
      clustering_pct,
      color="#ff7f0e",
      linewidth=1.5,
      label="Clustering Degree (%)",
  )
  plt.axhline(
      y=60, color="red", linestyle="--", label="Alert Threshold 60%"
  )
  plt.title("Clustering Degree Over Time", fontsize=14, fontweight="bold")
  plt.xlabel("Frame", fontsize=12)
  plt.ylabel("Clustering Degree (%)", fontsize=12)
  plt.legend(loc="upper right")
  plt.grid(True, linestyle="--", alpha=0.6)
  plt.tight_layout()

  cluster_plot_path = output_dir / f"{video_name}_clustering_over_time.png"
  plt.savefig(cluster_plot_path, dpi=300)
  plt.close()

  # ==========================================
  # 3. Movement Exploration Grid (ExplorationGrid)
  # ==========================================
  grid_size = 20
  df["x_grid"] = pd.cut(
      df["x"], bins=grid_size, labels=range(grid_size)
  )
  df["y_grid"] = pd.cut(
      df["y"], bins=grid_size, labels=range(grid_size)
  )
  visited_grids = df[["x_grid", "y_grid"]].drop_duplicates()

  grid_matrix = np.zeros((grid_size, grid_size))
  for _, row in visited_grids.iterrows():
    if pd.notna(row["x_grid"]) and pd.notna(row["y_grid"]):
      grid_matrix[int(row["y_grid"]), int(row["x_grid"])] = 1.0

  plt.figure(figsize=(7, 6))
  plt.imshow(grid_matrix, cmap="Blues", origin="upper", aspect="auto")
  cbar = plt.colorbar()
  cbar.set_label("Visited", rotation=270, labelpad=15)
  plt.title("Movement Exploration Grid", fontsize=14, fontweight="bold")
  plt.xlabel("X Grid", fontsize=12)
  plt.ylabel("Y Grid", fontsize=12)
  plt.tight_layout()

  grid_plot_path = output_dir / f"{video_name}_ExplorationGrid.png"
  plt.savefig(grid_plot_path, dpi=300)
  plt.close()

  # ==========================================
  # 4. Personnel Position Heatmap (Heatmap)
  # ==========================================
  plt.figure(figsize=(8, 6))
  plt.hist2d(df["x"], df["y"], bins=50, cmap="YlOrRd", cmin=1)
  cbar = plt.colorbar()
  cbar.set_label("Occurrence Count", rotation=270, labelpad=15)
  plt.gca().invert_yaxis()
  plt.title("Personnel Position Heatmap", fontsize=14, fontweight="bold")
  plt.xlabel("X Coordinate", fontsize=12)
  plt.ylabel("Y Coordinate", fontsize=12)
  plt.tight_layout()

  heatmap_path = output_dir / f"{video_name}_Heatmap.png"
  plt.savefig(heatmap_path, dpi=300)
  plt.close()

  print(f"[Info] Four analysis charts successfully saved to: {output_dir}")