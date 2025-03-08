import cv2
import os
import tempfile
import numpy as np
import pandas as pd
import deeplabcut
from head_pose_estimator import estimate_head_roll  # Optional if you want head roll

def detect_eye_and_landmarks(frame, config_path):
    """
    Process a single frame with DeepLabCut to detect eye landmarks.
    Returns a dictionary with:
      - landmarks: dict with keys 'left_pupil', 'right_pupil', 'corner_left', 'corner_right'
      - roll_angle: head roll (if estimated)
    """
    # Write the single frame to a temporary video file so that DLC can analyze it.
    # Alternatively, you could load the DLC model in memory and run it directly on each frame,
    # but for simplicity, we follow the original approach.
    height, width = frame.shape[:2]
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_video_path = os.path.join(temp_dir, 'temp_frame.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_video_path, fourcc, 1, (width, height))
        out.write(frame)
        out.release()
        
        # Run DLC on the temporary video
        deeplabcut.analyze_videos(
            config_path, 
            [temp_video_path],
            save_as_csv=True, 
            destfolder=temp_dir,
            videotype='.mp4'
        )
        
        csv_files = [f for f in os.listdir(temp_dir) if f.endswith('.csv')]
        if not csv_files:
            raise ValueError("No DLC results found. Check your DLC configuration.")
        result_csv = os.path.join(temp_dir, csv_files[0])
        
        # DLC typically outputs a CSV with multi-level headers
        df = pd.read_csv(result_csv, header=[1, 2])
        
        # If your DLC labels are different, adjust these keys accordingly
        try:
            left_pupil   = (df[('left_pupil', 'x')].iloc[0],   df[('left_pupil', 'y')].iloc[0])
            right_pupil  = (df[('right_pupil', 'x')].iloc[0],  df[('right_pupil', 'y')].iloc[0])
            corner_left  = (df[('corner_left', 'x')].iloc[0],  df[('corner_left', 'y')].iloc[0])
            corner_right = (df[('corner_right', 'x')].iloc[0], df[('corner_right', 'y')].iloc[0])
        except KeyError as e:
            raise ValueError("Missing expected landmarks in DLC output. Check your label names.") from e
        
        # Optionally, compute head roll angle (using a dummy camera matrix here)
        try:
            roll_angle = estimate_head_roll(
                {'left_pupil': left_pupil, 'right_pupil': right_pupil},
                camera_matrix=np.eye(3)
            )
        except Exception:
            roll_angle = None
        
    return {
        'landmarks': {
            'left_pupil': left_pupil,
            'right_pupil': right_pupil,
            'corner_left': corner_left,
            'corner_right': corner_right
        },
        'roll_angle': roll_angle
    }
