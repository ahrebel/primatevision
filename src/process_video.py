#!/usr/bin/env python
import cv2
import os
import pandas as pd
import warnings
import concurrent.futures

from detect_eye import detect_eye_and_landmarks

warnings.filterwarnings(
    "ignore",
    message="`layer.apply` is deprecated and will be removed in a future version."
)

def process_video(video_path, config_path, output_csv_path, workers):
    """
    Process the input video to extract eye landmarks using DeepLabCut.
    A batch of 'workers' frames is processed concurrently.
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

    def process_frame(frame, index):
        try:
            detection = detect_eye_and_landmarks(frame, config_path=config_path)
            landmarks = detection['landmarks']
            roll_angle = detection.get('roll_angle', None)
        except Exception as e:
            print(f"Frame {index}: Detection error: {e}")
            landmarks = {
                'left_pupil': (None, None),
                'right_pupil': (None, None),
                'corner_left': (None, None),
                'corner_right': (None, None)
            }
            roll_angle = None
        timestamp = index / fps if fps > 0 else None
        return {
            "frame": index,
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
        }
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        while True:
            frames_batch = []
            indices = []
            # Collect up to 'workers' frames
            for _ in range(workers):
                ret, frame = cap.read()
                if not ret:
                    break
                frames_batch.append(frame)
                indices.append(frame_num)
                frame_num += 1
            if not frames_batch:
                break
            # Process these frames concurrently
            batch_results = list(executor.map(process_frame, frames_batch, indices))
            results.extend(batch_results)
            
            # Every 5 frames, save the current results (overwriting any existing file)
            if frame_num % 5 == 0:
                df = pd.DataFrame(results)
                df.to_csv(output_csv_path, index=False)
                print(f"Saved results up to frame {frame_num} to {output_csv_path}")
    
    cap.release()
    # Final write (in case the last batch isn't a multiple of 5)
    df = pd.DataFrame(results)
    df.to_csv(output_csv_path, index=False)
    print(f"Video processing complete. Final results saved to {output_csv_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Process a video to extract eye landmarks using DeepLabCut, processing multiple frames concurrently and saving progress every 5 frames."
    )
    parser.add_argument("--video", required=True, help="Path to the input video")
    parser.add_argument("--config", required=True, help="Path to the DLC config.yaml file")
    parser.add_argument("--output", required=True, help="Output CSV file path")
    parser.add_argument("--workers", type=int, default=2, help="Number of worker threads (frames processed concurrently)")
    args = parser.parse_args()
    
    process_video(args.video, args.config, args.output, args.workers)
