#!/usr/bin/env python
import cv2
import os
import argparse
import tempfile
import numpy as np
import pandas as pd
import deeplabcut
import time

def smooth_landmarks_inplace(df, window_size=3):
    """
    Applies a rolling median filter to key landmark columns to reduce noise/outliers.
    Increase 'window_size' for stronger smoothing.
    """
    columns = [
        "left_pupil_x", "left_pupil_y",
        "right_pupil_x", "right_pupil_y",
        "corner_left_x", "corner_left_y",
        "corner_right_x", "corner_right_y"
    ]
    for col in columns:
        if col in df.columns:
            # center=True for symmetrical smoothing around each row
            df[col] = df[col].rolling(window=window_size, center=True, min_periods=1).median()
    return df

def process_video(
    video_path,
    config_path,
    output_csv_path,
    skip_frames=False,
    resize_factor=1.0,
    smooth_window=0
):
    """
    Reads an entire video, optionally skipping every other frame and resizing,
    then writes all processed frames to a single temporary video. Calls DLC once
    on that temporary video, parses the output CSV, and (optionally) applies a
    rolling median filter to improve accuracy.

    Args:
      video_path (str): Path to the input video.
      config_path (str): Path to the DLC config.yaml file (trained model).
      output_csv_path (str): Path to save the final CSV with columns:
          frame, time, left_pupil_x, left_pupil_y, right_pupil_x, right_pupil_y,
          corner_left_x, corner_left_y, corner_right_x, corner_right_y, roll_angle
      skip_frames (bool): If True, skip every other frame (for speed).
      resize_factor (float): Scale frames by this factor (1.0 = original size for best accuracy).
      smooth_window (int): Apply rolling median smoothing over this many frames (0 = no smoothing).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Compute new dimensions if resizing
    new_width = int(width * resize_factor)
    new_height = int(height * resize_factor)

    # This list will map the processed frame index -> (original global frame index, time)
    frame_map = []

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_video_path = os.path.join(temp_dir, "combined_temp.mp4")

        # Keep the same FPS to preserve timing in the final CSV
        out_fps = original_fps
        out_writer = cv2.VideoWriter(temp_video_path, fourcc, out_fps, (new_width, new_height))

        global_frame_index = 0
        processed_frame_count = 0
        start_time = time.time()

        print("Reading frames and writing them to a single temporary video...")
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # If skip_frames is True, we keep the current frame, skip the next
            if skip_frames:
                ret_skip, _ = cap.read()
                global_frame_index += 1
                if not ret_skip:
                    # End if there's no next frame to skip
                    pass

            # Optionally resize
            if resize_factor != 1.0:
                frame = cv2.resize(frame, (new_width, new_height))

            # Write this frame to the temp video
            out_writer.write(frame)

            # Record the mapping (processed_frame_count -> global_frame_index, time)
            frame_map.append((
                processed_frame_count,
                global_frame_index,
                global_frame_index / original_fps if original_fps > 0 else None
            ))

            processed_frame_count += 1
            global_frame_index += 1

        out_writer.release()
        cap.release()

        read_elapsed = time.time() - start_time
        print(f"Finished writing {processed_frame_count} frames to {temp_video_path}")
        print(f"Time for reading/skipping/resizing: {read_elapsed:.2f} s")

        # ---------------------------
        # Single DLC call
        # ---------------------------
        print("Running DLC analyze_videos (one-time call on combined_temp.mp4)...")
        dlc_start_time = time.time()

        deeplabcut.analyze_videos(
            config_path,
            [temp_video_path],
            save_as_csv=True,
            destfolder=temp_dir,
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

        # Build final data
        final_data = []
        for i in range(len(dlc_df)):
            processed_idx, orig_idx, orig_time = frame_map[i]

            # Adjust these bodypart names if your DLC labels differ
            left_pupil_x   = dlc_df[("left_pupil",   "x")].iloc[i]
            left_pupil_y   = dlc_df[("left_pupil",   "y")].iloc[i]
            right_pupil_x  = dlc_df[("right_pupil",  "x")].iloc[i]
            right_pupil_y  = dlc_df[("right_pupil",  "y")].iloc[i]
            corner_left_x  = dlc_df[("corner_left",  "x")].iloc[i]
            corner_left_y  = dlc_df[("corner_left",  "y")].iloc[i]
            corner_right_x = dlc_df[("corner_right", "x")].iloc[i]
            corner_right_y = dlc_df[("corner_right", "y")].iloc[i]

            # If needed, you can add head pose estimation or other fields
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

        # Optional rolling median smoothing for better accuracy
        if smooth_window > 0:
            final_df = smooth_landmarks_inplace(final_df, window_size=smooth_window)

        final_df.to_csv(output_csv_path, index=False)
        print(f"All done! Final CSV saved to {output_csv_path}")

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Process a video for DeepLabCut in a single call. Optionally skip frames, "
            "resize frames, and apply a rolling median smoothing for better accuracy."
        )
    )
    parser.add_argument("--video", required=True, help="Path to the input video")
    parser.add_argument("--config", required=True, help="Path to the DLC config.yaml file")
    parser.add_argument("--output", required=True, help="Output CSV file path")
    parser.add_argument("--skip_frames", action="store_true",
                        help="Skip every other frame (default: False for better accuracy)")
    parser.add_argument("--resize_factor", type=float, default=1.0,
                        help="Scale factor for frames (1.0 = original size for best accuracy)")
    parser.add_argument("--smooth_window", type=int, default=0,
                        help="Rolling median window size for smoothing landmarks (0 = no smoothing)")
    args = parser.parse_args()

    process_video(
        video_path=args.video,
        config_path=args.config,
        output_csv_path=args.output,
        skip_frames=args.skip_frames,
        resize_factor=args.resize_factor,
        smooth_window=args.smooth_window
    )

if __name__ == "__main__":
    main()
