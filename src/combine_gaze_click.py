import argparse
import pandas as pd
import numpy as np
import os

def parse_time_column(df, time_col='time'):
    """
    Parse the given 'time_col' in df.
    - If numeric, convert to float.
    - Otherwise, parse as datetime and convert to seconds from the earliest time.
    Returns df with a new column 'time_sec'.
    """
    if df[time_col].dtype.kind in ('i', 'f'):
        df['time_sec'] = df[time_col].astype(float)
    else:
        df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
        if df[time_col].isnull().any():
            raise ValueError(f"Some values in column '{time_col}' could not be parsed as datetimes.")
        earliest = df[time_col].min()
        df['time_sec'] = (df[time_col] - earliest).dt.total_seconds()
    return df

def load_click_data(click_path):
    """
    Load a click/touch file (CSV or TXT) with columns: timestamp,x,y.
    Rename them to: time, screen_x, screen_y, then parse 'time' into 'time_sec'.
    """
    df = pd.read_csv(click_path, delimiter=',', header=0)
    rename_map = {}
    if 'timestamp' in df.columns:
        rename_map['timestamp'] = 'time'
    if 'x' in df.columns:
        rename_map['x'] = 'screen_x'
    if 'y' in df.columns:
        rename_map['y'] = 'screen_y'
    df = df.rename(columns=rename_map)
    required_cols = ['time', 'screen_x', 'screen_y']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in click file: {click_path}")
    df = parse_time_column(df, time_col='time')
    return df

def combine_gaze_click(gaze_csv, click_file, output_csv, max_time_diff=0.05):
    """
    Merge gaze and click data by matching each click to the nearest gaze entry
    in time (within max_time_diff seconds).
    
    Gaze CSV must have columns:
        time, corner_left_x, corner_left_y, corner_right_x, corner_right_y,
              left_pupil_x, left_pupil_y, right_pupil_x, right_pupil_y.
    Its 'time' column is parsed into 'time_sec'.
    
    Click file (after renaming) must have:
        time, screen_x, screen_y, with 'time' parsed into 'time_sec'.
    
    Output CSV will have:
        time, left_corner_x, left_corner_y, right_corner_x, right_corner_y,
        left_pupil_x, left_pupil_y, right_pupil_x, right_pupil_y,
        screen_x, screen_y.
    
    If output_csv exists and is non-empty, new matches are appended,
    duplicates (completely identical rows) are dropped.
    """
    gaze_df = pd.read_csv(gaze_csv)
    required_gaze_cols = [
        'time',
        'corner_left_x', 'corner_left_y',
        'corner_right_x', 'corner_right_y',
        'left_pupil_x', 'left_pupil_y',
        'right_pupil_x', 'right_pupil_y'
    ]
    for col in required_gaze_cols:
        if col not in gaze_df.columns:
            raise ValueError(f"Column '{col}' not found in gaze CSV: {gaze_csv}")
    gaze_df = parse_time_column(gaze_df, time_col='time')
    click_df = load_click_data(click_file)
    
    combined_rows = []
    for _, click_row in click_df.iterrows():
        click_time_sec = click_row['time_sec']
        candidates = gaze_df[np.abs(gaze_df['time_sec'] - click_time_sec) <= max_time_diff]
        if candidates.empty:
            continue
        closest_idx = (np.abs(candidates['time_sec'] - click_time_sec)).idxmin()
        best_match = gaze_df.loc[closest_idx]
        combined_rows.append({
            'time': best_match['time'],
            'left_corner_x': best_match['corner_left_x'],
            'left_corner_y': best_match['corner_left_y'],
            'right_corner_x': best_match['corner_right_x'],
            'right_corner_y': best_match['corner_right_y'],
            'left_pupil_x': best_match['left_pupil_x'],
            'left_pupil_y': best_match['left_pupil_y'],
            'right_pupil_x': best_match['right_pupil_x'],
            'right_pupil_y': best_match['right_pupil_y'],
            'screen_x': click_row['screen_x'],
            'screen_y': click_row['screen_y']
        })
    
    if not combined_rows:
        print("No matching rows found. Check your time columns or max_time_diff.")
        return
    
    new_matches_df = pd.DataFrame(combined_rows)
    
    # If output file exists and is non-empty, append; otherwise, write new file.
    if os.path.exists(output_csv) and os.path.getsize(output_csv) > 0:
        try:
            existing_df = pd.read_csv(output_csv)
        except pd.errors.EmptyDataError:
            existing_df = pd.DataFrame()
        # Concatenate new and existing data then drop duplicates.
        combined_df = pd.concat([existing_df, new_matches_df], ignore_index=True)
        combined_df = combined_df.drop_duplicates()
        combined_df.to_csv(output_csv, index=False)
        print(f"Appended new rows to existing file '{output_csv}' (duplicates removed).")
    else:
        new_matches_df.to_csv(output_csv, index=False)
        print(f"Combined calibration data saved to new file '{output_csv}'.")
    
    # (Optional) Scale the screen_x and screen_y columns by 2 if desired
    new_matches_df['screen_x'] /= 2
    new_matches_df['screen_y'] /= 2

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine gaze data with click/touch data for calibration.")
    parser.add_argument("--gaze_csv", required=True,
                        help="CSV file with raw eye landmarks (must include a 'time' column and expected column names)")
    parser.add_argument("--click_file", required=True,
                        help="File (CSV or TXT) with 'timestamp,x,y' columns to be merged")
    parser.add_argument("--output_csv", required=True,
                        help="Where to save the merged calibration CSV (appends if file exists and is non-empty)")
    parser.add_argument("--max_time_diff", type=float, default=0.05,
                        help="Max time difference (in seconds) for matching gaze to clicks")
    args = parser.parse_args()
    
    combine_gaze_click(args.gaze_csv, args.click_file, args.output_csv, max_time_diff=args.max_time_diff)
