#!/usr/bin/env python
import cv2
import os
import pandas as pd
import warnings
from detect_eye import detect_eye_and_landmarks

warnings.filterwarnings(
    "ignore",
    message="`layer.apply` is deprecated and will be removed in a future version."
)

def process_video(video_path, config_path, output_csv_path):
    """
    Process the input video to extract eye landmarks using DeepLabCut.
    Every 5 frames, the current results are saved (overwriting any existing file).
    
    Output CSV will contain one row per frame with columns:
      frame, time, left_pupil_x, left_pupil_y, right_pupil_x, right_pupil_y,
      corner_left_x, corner_left_y, corner_right_x, corner_right_y, roll_angle.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_num = 0
    results = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        try:
            detection = detect_eye_and_landmarks(frame, config_path=config_path)
            landmarks = detection['landmarks']
            roll_angle = detection.get('roll_angle', None)
        except Exception as e:
            print(f"Frame {frame_num}: Detection error: {e}")
            landmarks = {
                'left_pupil': (None, None),
                'right_pupil': (None, None),
                'corner_left': (None, None),
                'corner_right': (None, None)
            }
            roll_angle = None
        
        timestamp = frame_num / fps if fps > 0 else None
        results.append({
            "frame": frame_num,
            "time": timestamp,
            "left_pupil_x": landmarks['left_pupil'][0],
            "left_pupil_y": landmarks['left_pupil'][1],
            "right_pupil_x": landmarks['right_pupil'][0],
            "right_pupil_y": landmarks['right_pupil'][1],
            "corner_left_x": landmarks['corner_left'][0],
            "corner_left_y": landmarks['corner_left'][1],
            "corner_right_x": landmarks['corner_right'][0],
            "corner_right_y": landmarks['corner_right'][1],
            "roll_angle": roll_angle
        })
        frame_num += 1
        
        # Every 5 frames, overwrite the output CSV with current results.
        if frame_num % 5 == 0:
            df = pd.DataFrame(results)
            df.to_csv(output_csv_path, index=False)
            print(f"Saved results up to frame {frame_num} to {output_csv_path}")
    
    cap.release()
    
    # Final write (in case the last batch is not a multiple of 5)
    df = pd.DataFrame(results)
    df.to_csv(output_csv_path, index=False)
    print(f"Video processing complete. Final results saved to {output_csv_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Process a video to extract eye landmarks using DeepLabCut, saving progress every 5 frames."
    )
    parser.add_argument("--video", required=True, help="Path to the input video")
    parser.add_argument("--config", required=True, help="Path to the DLC config.yaml file")
    parser.add_argument("--output", required=True, help="Output CSV file path")
    args = parser.parse_args()
    
    process_video(args.video, args.config, args.output)
