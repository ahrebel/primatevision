#!/usr/bin/env python
import cv2
import os
import argparse
import tempfile
import numpy as np
import pandas as pd
import deeplabcut
import time

def process_video_single_call(
    video_path,
    config_path,
    output_csv_path,
    skip_frames=False,
    resize_factor=1.0
):
    """
    Reads an entire video, skipping/resizing frames as needed, and writes them to a single
    temporary video. Then calls DLC's `analyze_videos` once, parses the resulting CSV,
    and outputs a final CSV of landmarks.

    This drastically reduces overhead compared to analyzing each frame individually.

    Args:
      video_path (str): Path to the input video.
      config_path (str): Path to the DLC config.yaml file (with trained model info).
      output_csv_path (str): Final CSV with columns:
          frame, time, left_pupil_x, left_pupil_y, right_pupil_x, right_pupil_y,
          corner_left_x, corner_left_y, corner_right_x, corner_right_y, roll_angle
      skip_frames (bool): If True, skip every other frame (for speed).
      resize_factor (float): E.g., 0.5 = half resolution for faster CPU inference.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    # Gather metadata from original video
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # If resizing, compute new dimensions
    new_width = int(width * resize_factor)
    new_height = int(height * resize_factor)

    # We'll store a mapping from processed_frame_index -> (global_frame_index, time)
    # so we can reconstruct which frames were included after DLC analysis.
    frame_map = []

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_video_path = os.path.join(temp_dir, "combined_temp.mp4")

        # Keep the same fps as original to preserve timing in the final CSV
        out_fps = original_fps

        out_writer = cv2.VideoWriter(temp_video_path, fourcc, out_fps, (new_width, new_height))

        global_frame_index = 0
        processed_frame_count = 0
        start_time = time.time()

        print("Reading and writing frames to a single temporary video...")
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Optionally skip every other frame
            if skip_frames:
                # We'll keep the first, skip the next, etc.
                ret_skip, _ = cap.read()
                global_frame_index += 1
                if not ret_skip:
                    # If we can't read the next, we're done
                    break

            # Resize if needed
            if resize_factor != 1.0:
                frame = cv2.resize(frame, (new_width, new_height))

            out_writer.write(frame)

            # Record the mapping: (processed_frame_count, original_global_index, time)
            frame_map.append((
                processed_frame_count,
                global_frame_index,
                global_frame_index / original_fps if original_fps > 0 else None
            ))

            processed_frame_count += 1
            global_frame_index += 1

        out_writer.release()
        cap.release()

        print(f"Finished writing {processed_frame_count} frames to {temp_video_path}.")
        print(f"Time for pre-processing: {time.time() - start_time:.2f} s")

        # ---------------------------
        # Call DLC once on the combined video
        # ---------------------------
        print("Running DLC analyze_videos on the combined_temp.mp4 (one-time call)...")
        dlc_start_time = time.time()

        deeplabcut.analyze_videos(
            config_path,
            [temp_video_path],
            save_as_csv=True,
            destfolder=temp_dir,  # store DLC results in the temp dir
            videotype=".mp4"
        )

        dlc_elapsed = time.time() - dlc_start_time
        print(f"DLC analysis done. Time: {dlc_elapsed:.2f} s")

        # ---------------------------
        # Parse DLC CSV
        # ---------------------------
        csv_files = [f for f in os.listdir(temp_dir) if f.endswith(".csv")]
        if not csv_files:
            raise ValueError("No DLC results found. Check your DLC configuration.")

        dlc_result_csv = os.path.join(temp_dir, csv_files[0])
        dlc_df = pd.read_csv(dlc_result_csv, header=[1, 2])

        # Example keys: if your labeled bodyparts are "left_pupil", "right_pupil", etc.
        # Adjust if your labels differ.
        final_data = []
        for i in range(len(dlc_df)):
            processed_idx, orig_idx, orig_time = frame_map[i]

            left_pupil_x   = dlc_df[("left_pupil",   "x")].iloc[i]
            left_pupil_y   = dlc_df[("left_pupil",   "y")].iloc[i]
            right_pupil_x  = dlc_df[("right_pupil",  "x")].iloc[i]
            right_pupil_y  = dlc_df[("right_pupil",  "y")].iloc[i]
            corner_left_x  = dlc_df[("corner_left",  "x")].iloc[i]
            corner_left_y  = dlc_df[("corner_left",  "y")].iloc[i]
            corner_right_x = dlc_df[("corner_right", "x")].iloc[i]
            corner_right_y = dlc_df[("corner_right", "y")].iloc[i]

            # For simplicity, skip roll_angle or compute it externally if needed
            roll_angle = None

            final_data.append({
                "frame": orig_idx,
                "time": orig_time,
                "left_pupil_x": left_pupil_x,
                "left_pupil_y": left_pupil_y,
                "right_pupil_x": right_pupil_x,
                "right_pupil_y": right_pupil_y,
                "corner_left_x": corner_left_x,
                "corner_left_y": corner_left_y,
                "corner_right_x": corner_right_x,
                "corner_right_y": corner_right_y,
                "roll_angle": roll_angle
            })

        final_df = pd.DataFrame(final_data)
        final_df.to_csv(output_csv_path, index=False)
        print(f"All done! Final CSV saved to {output_csv_path}")

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Drastically speed up DLC analysis by combining all frames into a single temporary video. "
            "Skip frames and/or resize for CPU performance. Then call DLC's `analyze_videos` once."
        )
    )
    parser.add_argument("--video", required=True, help="Path to the input video")
    parser.add_argument("--config", required=True, help="Path to the DLC config.yaml file")
    parser.add_argument("--output", required=True, help="Output CSV file path")
    parser.add_argument("--skip_frames", action="store_true", help="Skip every other frame for speed")
    parser.add_argument("--resize_factor", type=float, default=1.0,
                        help="Resize factor for frames (e.g., 0.5 = half resolution)")
    args = parser.parse_args()

    process_video_single_call(
        video_path=args.video,
        config_path=args.config,
        output_csv_path=args.output,
        skip_frames=args.skip_frames,
        resize_factor=args.resize_factor
    )

if __name__ == "__main__":
    main()
