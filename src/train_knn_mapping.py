#!/usr/bin/env python
import argparse
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
import joblib
import os

def load_calibration_data(csv_path):
    """
    Loads calibration data from a CSV file. The file must contain columns:
      left_corner_x, left_corner_y, right_corner_x, right_corner_y,
      left_pupil_x, left_pupil_y, right_pupil_x, right_pupil_y,
      screen_x, screen_y
    Returns:
        X: numpy array of shape (N, 8) – input eye landmarks
        y: numpy array of shape (N, 2) – target screen coordinates
    """
    df = pd.read_csv(csv_path)
    required_columns = [
        'left_corner_x', 'left_corner_y',
        'right_corner_x', 'right_corner_y',
        'left_pupil_x', 'left_pupil_y',
        'right_pupil_x', 'right_pupil_y',
        'screen_x', 'screen_y'
    ]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in {csv_path}.")
    X = df[[
        'left_corner_x', 'left_corner_y',
        'right_corner_x', 'right_corner_y',
        'left_pupil_x', 'left_pupil_y',
        'right_pupil_x', 'right_pupil_y'
    ]].values
    y = df[['screen_x', 'screen_y']].values
    return X, y

def train_knn_mapping(X, y, n_neighbors=5):
    """
    Trains a k-Nearest Neighbors regressor on the calibration data.
    """
    model = KNeighborsRegressor(n_neighbors=n_neighbors)
    model.fit(X, y)
    return model

def main():
    parser = argparse.ArgumentParser(description="Train kNN mapping from eye landmarks to screen coordinates")
    parser.add_argument("--data", required=True, help="Path to the calibration CSV file")
    parser.add_argument("--output", required=True, help="Path to save the trained kNN model (e.g., knn_mapping_model.joblib)")
    parser.add_argument("--neighbors", type=int, default=5, help="Number of neighbors for kNN regression")
    args = parser.parse_args()
    
    X, y = load_calibration_data(args.data)
    print(f"Loaded calibration data: X shape = {X.shape}, y shape = {y.shape}")
    
    model = train_knn_mapping(X, y, n_neighbors=args.neighbors)
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    joblib.dump(model, args.output)
    print(f"Trained kNN model saved to '{args.output}'")

if __name__ == "__main__":
    main()
