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
    model = joblib.load(model_path)
    return model

def map_eye_to_screen(row, model):
    """
    Maps eye landmarks to screen coordinates using the kNN model.
    
    Expects the row to have these columns:
      corner_left_x, corner_left_y, corner_right_x, corner_right_y,
      left_pupil_x, left_pupil_y, right_pupil_x, right_pupil_y
    """
    features = np.array([
        row['corner_left_x'], row['corner_left_y'],
        row['corner_right_x'], row['corner_right_y'],
        row['left_pupil_x'],  row['left_pupil_y'],
        row['right_pupil_x'], row['right_pupil_y']
    ], dtype=np.float32)
    
    # kNN model expects a 2D array
    prediction = model.predict(features.reshape(1, -1))
    return prediction.squeeze()  # returns [screen_x, screen_y]

def analyze_gaze(landmarks_csv, model_path, screen_width, screen_height,
                 n_cols, n_rows, output_heatmap, output_sections):
    """
    Reads the landmarks CSV, uses the trained kNN model to map eye landmarks to screen
    coordinates, divides the screen into a grid, and computes the time spent in each region,
    then produces a heatmap.
    """
    # Load landmarks CSV (must include a 'time' column)
    df = pd.read_csv(landmarks_csv)
    
    # Load the trained kNN model
    knn_model = load_knn_model(model_path)
    
    # Map eye landmarks to screen coordinates using the kNN model
    screen_coords = df.apply(lambda row: map_eye_to_screen(row, knn_model), axis=1)
    df['screen_x'] = screen_coords.apply(lambda coord: coord[0])
    df['screen_y'] = screen_coords.apply(lambda coord: coord[1])
    
    # Create a grid of screen sections
    grid = create_grid(screen_width, screen_height, n_cols, n_rows)
    df['region'] = df.apply(lambda r: get_region_for_point(r['screen_x'], r['screen_y'], grid), axis=1)
    
    # Estimate average frame duration from the 'time' column (if available)
    times = df['time'].dropna().values
    avg_frame_duration = np.mean(np.diff(times)) if len(times) > 1 else 0
    
    # Count frames per region and compute time spent per region
    region_counts = df['region'].value_counts().sort_index()
    region_times = region_counts * avg_frame_duration
    region_times = region_times.reindex(range(len(grid)), fill_value=0).reset_index()
    region_times.columns = ['region', 'time_spent']
    region_times.to_csv(output_sections, index=False)
    print(f"Section time distribution saved to {output_sections}")
    
    # Build a 2D array for the heatmap
    heatmap_data = np.zeros((n_rows, n_cols))
    for _, row_data in region_times.iterrows():
        reg_id = row_data['region']
        if pd.isna(reg_id):
            continue
        reg_id = int(reg_id)  # Ensure region id is integer
        reg_row = reg_id // n_cols
        reg_col = reg_id % n_cols
        heatmap_data[reg_row, reg_col] = row_data['time_spent']
    
    # Plot the heatmap
    plt.figure(figsize=(8, 6))
    plt.imshow(heatmap_data, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Time Spent (s)')
    plt.title("Heatmap of Gaze Duration per Screen Section")
    plt.xlabel("Columns")
    plt.ylabel("Rows")
    plt.savefig(output_heatmap, dpi=150)
    plt.show()
    print(f"Heatmap saved to {output_heatmap}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze gaze data using a kNN mapping model and generate a heatmap.")
    parser.add_argument("--landmarks_csv", required=True,
                        help="CSV file with columns: corner_left_x, corner_left_y, corner_right_x, corner_right_y, left_pupil_x, left_pupil_y, right_pupil_x, right_pupil_y, time")
    parser.add_argument("--model", required=True,
                        help="Path to the trained kNN mapping model (e.g., knn_mapping_model.joblib)")
    parser.add_argument("--screen_width", type=int, required=True, help="Screen width in pixels")
    parser.add_argument("--screen_height", type=int, required=True, help="Screen height in pixels")
    parser.add_argument("--n_cols", type=int, default=3, help="Number of columns to divide the screen")
    parser.add_argument("--n_rows", type=int, default=3, help="Number of rows to divide the screen")
    parser.add_argument("--output_heatmap", required=True, help="Path to save the generated heatmap image")
    parser.add_argument("--output_sections", required=True, help="Path to save the CSV with section durations")
    args = parser.parse_args()
    
    analyze_gaze(args.landmarks_csv, args.model, args.screen_width, args.screen_height,
                 args.n_cols, args.n_rows, args.output_heatmap, args.output_sections)
