# PrimateVision

![Animated Eyes](https://github.com/user-attachments/assets/0f245b14-ec20-4a11-868a-ae207a7dfa1d)

PrimateVision implements an eye–tracking system for Rhesus macaques (and humans) using **DeepLabCut (DLC)**. It detects eye landmarks from video frames, maps raw eye coordinates to screen positions using a trained k–Nearest Neighbors (kNN) regression model, and analyzes gaze/fixation patterns during touchscreen interactions.

> **Key Features:**
> - **Offline Video Processing:** Analyze pre–recorded trial videos in a single DLC call.  
> - **DeepLabCut–Based Landmark Detection:** Utilize a trained DLC model to extract key eye landmarks.  
> - **Gaze Mapping (kNN):** Transform raw eye coordinates to screen positions using a calibration-trained kNN regressor.  
> - **Visualization:** Generate heatmaps and CSV summaries showing time spent in each screen region.  
> - **Cross–Platform Support:** Operates on macOS and Windows (CPU–only supported; GPU available for DeepLabCut).  
> - **Optional Head–Pose Estimation:** Provides head roll estimation via facial landmarks.

---

## Table of Contents

1. [Installation and Setup](#installation-and-setup)  
2. [Data Preparation](#data-preparation)  
3. [DeepLabCut Model Training](#deeplabcut-model-training)  
4. [Pipeline Overview](#pipeline-overview)  
5. [Calibration & Processing](#calibration--processing)  
6. [Gaze Analysis & Visualization](#gaze-analysis--visualization)  
7. [Testing, CI, and Contribution Guidelines](#testing-ci-and-contribution-guidelines)  
8. [Future Improvements](#future-improvements)  
9. [License](#license)

---

## 1. Installation and Setup

### Clone the Repository

```bash
git clone https://github.com/ahrebel/primatevision.git
cd primatevision
```

### Set Up the Python Environment

**Using Conda (recommended):**

```bash
conda create -n primatevision -c conda-forge python=3.8 pytables hdf5 lzo opencv numpy pandas matplotlib scikit-learn scikit-image scipy tqdm statsmodels
conda activate primatevision
```

**Or using pip with a virtual environment:**

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

### Install Required Packages

```bash
pip install -r requirements.txt
```

Install additional DeepLabCut dependencies:

```bash
pip install deeplabcut pyyaml tensorflow tensorpack tf-slim 'deeplabcut[gui]'
```

> *If a `ModuleNotFoundError: No module named 'keras.legacy_tf_layers'` occurs, run:*
> ```bash
> pip install --upgrade tensorflow_macos==2.12.0
> ```

---

## 2. Data Preparation

- **Videos:**  
  Place trial videos (e.g., `1.mp4`, `2.mp4`) in the `videos/input/` folder.

- **Touch/Click Event Files (Optional):**  
  If you have calibration data from touchscreens, store them as CSV files (e.g., `time, screen_x, screen_y`).

---

## 3. DeepLabCut Model Training

1. **Launch the DLC GUI:**

   ```bash
   python -m deeplabcut
   ```

2. **Create a New Project:**  
   Enter the project name, add the video(s), and define your keypoints (e.g., `left_pupil`, `right_pupil`, `corner_left`, `corner_right`).

3. **Label Frames:**  
   Label a diverse set of frames covering various lighting conditions and head poses.

4. **Train and Evaluate:**  
   Train the network (e.g., `mobilenet_v2_1.0` for CPU-only setups) and evaluate its performance.

5. **Update Configuration:**  
   Make sure your scripts (e.g., `detect_eye.py`) point to the correct DLC config (`config.yaml`).

---

## 4. Pipeline Overview

PrimateVision follows these major steps:

1. **Calibration:**  
   Extract eye landmarks from a calibration video.
2. **(Optional) Data Merging:**  
   Combine gaze data with touchscreen logs to form a calibration CSV.
3. **Mapping Model Training:**  
   Train a kNN regression model to map raw eye coordinates to screen coordinates.
4. **Experimental Processing:**  
   Process experimental videos to extract raw landmarks.
5. **Gaze Analysis:**  
   Apply the kNN model to compute screen positions, divide the screen into regions, and calculate fixation durations.

---

## 5. Calibration & Processing

### Single-Call Video Processing

Use `process_video.py` to create a **single temporary video** containing the frames you want to analyze. The script then calls DLC **once**, parsing the resulting CSV to produce a final landmarks file. This approach reduces overhead and can improve accuracy by avoiding repeated DLC initialization.

**Key Options:**
- **`--skip_frames`**: Skip every other frame for speed (disabled by default).  
- **`--resize_factor`**: Downscale frames (e.g., `0.5` for half resolution). Default is `1.0` for full accuracy.  
- **`--smooth_window`**: Apply a rolling median filter to final landmark coordinates to reduce jitter (e.g., `3`).

#### Example: Full Accuracy (No Skipping, Full Res, No Smoothing)

```bash
python src/process_video.py \
  --video /path/to/calibration_video.mp4 \
  --config /path/to/dlc_config.yaml \
  --output /path/to/landmarks_output.csv
```

#### Example: Skipping & Resizing

```bash
python src/process_video.py \
  --video /path/to/calibration_video.mp4 \
  --config /path/to/dlc_config.yaml \
  --output /path/to/landmarks_output.csv \
  --skip_frames \
  --resize_factor 0.5
```
*(Speeds up processing but reduces resolution and temporal detail.)*

#### Example: Adding Smoothing

```bash
python src/process_video.py \
  --video /path/to/calibration_video.mp4 \
  --config /path/to/dlc_config.yaml \
  --output /path/to/landmarks_output.csv \
  --smooth_window 3
```
*(Applies a rolling median filter over 3 frames to reduce noise in the final CSV.)*

---

### (Optional) Merge Gaze Data with Click Data

If you have touchscreen events for calibration, merge them:

```bash
python src/combine_gaze_click.py \
  --gaze_csv landmarks_output.csv \
  --click_file /path/to/click_data.csv \
  --output_csv calibration_data_for_training.csv \
  --max_time_diff 0.1
```

---

## 6. Gaze Analysis & Visualization

### Train the kNN Mapping Model

After merging gaze and click data, train a kNN model:

```bash
python src/train_knn_mapping.py \
  --data calibration_data_for_training.csv \
  --output data/trained_model/knn_mapping_model.joblib \
  --neighbors 5
```

### Process Experimental Videos

Repeat the **single-call** approach for experimental videos:

```bash
python src/process_video.py \
  --video /path/to/experimental_video.mp4 \
  --config /path/to/dlc_config.yaml \
  --output /path/to/landmarks_output.csv
```

*(Adjust options like `--skip_frames`, `--resize_factor`, or `--smooth_window` as needed.)*

### Analyze Gaze and Generate Heatmaps

```bash
python src/analyze_gaze_knn.py \
  --landmarks_csv /path/to/landmarks_output.csv \
  --model data/trained_model/knn_mapping_model.joblib \
  --screen_width 1920 \
  --screen_height 1080 \
  --n_cols 3 \
  --n_rows 3 \
  --output_heatmap gaze_heatmap.png \
  --output_sections section_durations.csv
```

This script:
1. Loads the landmarks CSV.  
2. Maps eye coordinates → screen coordinates via the kNN model.  
3. Divides the screen (e.g., 1920×1080) into a grid (3×3).  
4. Computes time spent in each grid cell.  
5. Saves a heatmap image and a CSV summarizing fixation durations.
   

![gaze_heatmap](https://github.com/user-attachments/assets/e195b9da-d5e1-4c11-a5d8-59ffae3004e7)


---

## 7. Testing, CI, and Contribution Guidelines

- **Testing:**  
  Unit and integration tests validate calibration, kNN mapping, and other pipeline components.

- **Continuous Integration (CI):**  
  GitHub Actions runs tests and lint checks on every push/pull request.

- **Contribution Guidelines:**  
  Refer to [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on pull requests, issues, and our code of conduct.

---

## 8. Future Improvements

- **Advanced Feature Engineering:**  
  Investigate additional features (e.g., inter-landmark distances) to improve mapping accuracy.
- **Model Comparisons:**  
  Experiment with alternative regression models (e.g., SVR, Random Forests) via cross-validation.
- **Adaptive Calibration:**  
  Develop real-time calibration methods for dynamic gaze mapping.
- **Enhanced Visualization:**  
  Implement interactive dashboards or real-time displays for gaze data.
- **Docker Support:**  
  Provide a Dockerfile for a fully containerized setup.

---

## 9. License

This project is licensed under the [GPL-3.0 License](LICENSE).  
See the license file for details regarding usage and distribution.
