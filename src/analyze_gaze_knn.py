#!/usr/bin/env python
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from section_mapping import create_grid, get_region_for_point

def load_knn_model(model_path):
    """
    Loads a trained kNN model from file using joblib.
    """
    return joblib.load(model_path)

def analyze_gaze(landmarks_csv, model_path, screen_width, screen_height,
                 n_cols, n_rows, output_heatmap, output_sections):
    """
    Reads the landmarks CSV, uses the trained kNN model to map eye landmarks to screen
    coordinates in a vectorized manner, divides the screen into a grid, and computes
    the time spent in each region, then produces a heatmap.
    """
    # Load landmarks CSV (must include 'time' column)
    df = pd.read_csv(landmarks_csv)

    # Load the trained kNN model
    knn_model = load_knn_model(model_path)

    # --- Vectorized kNN inference --- #
    features = df[
        [
            "corner_left_x",  "corner_left_y",
            "corner_right_x", "corner_right_y",
            "left_pupil_x",   "left_pupil_y",
            "right_pupil_x",  "right_pupil_y"
        ]
    ].values.astype(np.float32)

    # Predict all at once
    predictions = knn_model.predict(features)
    # predictions.shape = (num_frames, 2)

    df["screen_x"] = predictions[:, 0]
    df["screen_y"] = predictions[:, 1]

    # Create a grid of screen sections
    grid = create_grid(screen_width, screen_height, n_cols, n_rows)

    # Vectorized region assignment
    def get_regions_for_points(xs, ys, grid):
        region_ids = []
        for x, y in zip(xs, ys):
            region_id = get_region_for_point(x, y, grid)
            region_ids.append(region_id)
        return np.array(region_ids, dtype=np.float32)

    df["region"] = get_regions_for_points(df["screen_x"].values, df["screen_y"].values, grid)

    # Estimate average frame duration from 'time' column
    times = df["time"].dropna().values
    avg_frame_duration = np.mean(np.diff(times)) if len(times) > 1 else 0.0

    # Count frames per region
    region_counts = df["region"].value_counts().sort_index()

    # Time spent in each region
    region_times = region_counts * avg_frame_duration
    region_times = region_times.reindex(range(len(grid)), fill_value=0).reset_index()
    region_times.columns = ["region", "time_spent"]
    region_times.to_csv(output_sections, index=False)
    print(f"Section time distribution saved to {output_sections}")

    # Build a 2D array for the heatmap
    heatmap_data = np.zeros((n_rows, n_cols), dtype=np.float32)
    for _, row_data in region_times.iterrows():
        reg_id = int(row_data["region"])
        reg_row = reg_id // n_cols
        reg_col = reg_id % n_cols
        heatmap_data[reg_row, reg_col] = row_data["time_spent"]

    # Plot the heatmap
    plt.figure(figsize=(8, 6))
    plt.imshow(heatmap_data, cmap="hot", interpolation="nearest")
    plt.colorbar(label="Time Spent (s)")
    plt.title("Heatmap of Gaze Duration per Screen Section")
    plt.xlabel("Columns")
    plt.ylabel("Rows")
    plt.savefig(output_heatmap, dpi=150)
    plt.close()
    print(f"Heatmap saved to {output_heatmap}")

def main():
    parser = argparse.ArgumentParser(
        description="Analyze gaze data using a kNN mapping model and generate a heatmap."
    )
    parser.add_argument("--landmarks_csv", required=True,
                        help="CSV with columns for corner_x/y, pupil_x/y, and 'time'.")
    parser.add_argument("--model", required=True,
                        help="Path to the trained kNN mapping model (e.g., knn_mapping_model.joblib)")
    parser.add_argument("--screen_width", type=int, required=True, help="Screen width in pixels")
    parser.add_argument("--screen_height", type=int, required=True, help="Screen height in pixels")
    parser.add_argument("--n_cols", type=int, default=3, help="Number of columns in the grid")
    parser.add_argument("--n_rows", type=int, default=3, help="Number of rows in the grid")
    parser.add_argument("--output_heatmap", required=True, help="Path to save the heatmap image")
    parser.add_argument("--output_sections", required=True, help="Path to save the CSV with section durations")
    args = parser.parse_args()

    analyze_gaze(
        landmarks_csv=args.landmarks_csv,
        model_path=args.model,
        screen_width=args.screen_width,
        screen_height=args.screen_height,
        n_cols=args.n_cols,
        n_rows=args.n_rows,
        output_heatmap=args.output_heatmap,
        output_sections=args.output_sections
    )

if __name__ == "__main__":
    main()
