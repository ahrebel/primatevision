#!/usr/bin/env python
import cv2
import os
import pandas as pd
import warnings
import concurrent.futures
import time
import argparse

from detect_eye import detect_eye_and_landmarks

warnings.filterwarnings(
    "ignore",
    message="`layer.apply` is deprecated and will be removed in a future version."
)

def process_video(
    video_path,
    config_path,
    output_csv_path,
    workers=2,
    skip_frames=False,
    resize_factor=1.0,
    save_interval=50
):
    """
    Process the input video to extract eye landmarks using DeepLabCut, with optional skipping
    and resizing for faster CPU performance. A batch of 'workers' frames is processed
    concurrently. Results are saved every 'save_interval' processed frames.

    Args:
      video_path (str): Path to the input video.
      config_path (str): Path to the DLC config.yaml file.
      output_csv_path (str): Where to save the output CSV.
      workers (int): Number of threads (frames processed concurrently).
      skip_frames (bool): If True, skip every other frame to speed up analysis.
      resize_factor (float): Scale factor for resizing frames (e.g., 0.5 = half size).
      save_interval (int): Save intermediate CSV after this many processed frames.

    The output CSV has one row per processed frame with columns:
      frame, time, left_pupil_x, left_pupil_y, right_pupil_x, right_pupil_y,
      corner_left_x, corner_left_y, corner_right_x, corner_right_y, roll_angle.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Tracks all frames read, including skipped ones
    global_frame_index = 0  
    # Tracks how many frames were actually processed
    processed_count = 0

    results = []
    start_time = time.time()

    def process_frame(frame, unique_index):
        """
        Worker function to detect eye landmarks in a single frame.
        'unique_index' is the global index at read time.
        """
        try:
            # Optionally resize for speed
            if resize_factor != 1.0:
                new_w = int(frame.shape[1] * resize_factor)
                new_h = int(frame.shape[0] * resize_factor)
                frame = cv2.resize(frame, (new_w, new_h))

            detection = detect_eye_and_landmarks(frame, config_path=config_path)
            landmarks = detection['landmarks']
            roll_angle = detection.get('roll_angle', None)
        except Exception as e:
            print(f"Frame {unique_index}: Detection error: {e}")
            landmarks = {
                'left_pupil': (None, None),
                'right_pupil': (None, None),
                'corner_left': (None, None),
                'corner_right': (None, None)
            }
            roll_angle = None
        
        # Compute timestamp from the global index
        timestamp = unique_index / fps if fps > 0 else None
        
        return {
            "frame": unique_index,
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
            frame_indices = []
            
            # Collect up to 'workers' frames
            for _ in range(workers):
                ret, frame = cap.read()
                if not ret:
                    break

                current_index = global_frame_index
                global_frame_index += 1

                frames_batch.append(frame)
                frame_indices.append(current_index)
                processed_count += 1

                # Optionally skip every other frame
                if skip_frames:
                    ret_skip, _ = cap.read()
                    global_frame_index += 1
                    if not ret_skip:
                        break

            if not frames_batch:
                break

            # Process frames concurrently
            batch_results = list(executor.map(process_frame, frames_batch, frame_indices))
            results.extend(batch_results)
            
            # Save progress every 'save_interval' processed frames
            if processed_count % save_interval == 0:
                df = pd.DataFrame(results)
                df.to_csv(output_csv_path, index=False)

                elapsed_time = time.time() - start_time
                processing_rate = processed_count / elapsed_time if elapsed_time > 0 else 0

                # Estimate time to process 1 minute of video
                # 1 minute = 60 * fps frames
                estimated_time_per_minute = (60 * fps) / processing_rate if processing_rate > 0 else float('inf')

                print(f"Saved results at processed frame {processed_count} (global index: {global_frame_index})")
                print(f"Speed: {processing_rate:.2f} frames/s | "
                      f"Estimated time for 1 minute of video: {estimated_time_per_minute:.2f} s")

    cap.release()

    # Final write
    df = pd.DataFrame(results)
    df.to_csv(output_csv_path, index=False)
    print(f"Processing complete. Final results saved to {output_csv_path}")

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Process a video to extract eye landmarks using DeepLabCut, "
            "optionally skipping frames and resizing for faster CPU performance. "
            "Frames are processed concurrently, and progress is saved periodically."
        )
    )
    parser.add_argument("--video", required=True, help="Path to the input video")
    parser.add_argument("--config", required=True, help="Path to the DLC config.yaml file")
    parser.add_argument("--output", required=True, help="Output CSV file path")
    parser.add_argument("--workers", type=int, default=2, help="Number of worker threads")
    parser.add_argument("--skip_frames", action="store_true", help="Skip every other frame for speed")
    parser.add_argument("--resize_factor", type=float, default=1.0,
                        help="Resize factor for frames (e.g., 0.5 = half resolution)")
    parser.add_argument("--save_interval", type=int, default=50,
                        help="How many processed frames between CSV saves")
    args = parser.parse_args()

    process_video(
        video_path=args.video,
        config_path=args.config,
        output_csv_path=args.output,
        workers=args.workers,
        skip_frames=args.skip_frames,
        resize_factor=args.resize_factor,
        save_interval=args.save_interval
    )

if __name__ == "__main__":
    main()
