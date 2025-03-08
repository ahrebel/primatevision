# Rhesus Macaque Gaze Tracker (DeepLabCut)

![Animated Eyes](https://github.com/user-attachments/assets/0f245b14-ec20-4a11-868a-ae207a7dfa1d)

This repository implements an eye–tracking system for Rhesus macaques (and humans) using **DeepLabCut (DLC)**. The system detects eye landmarks from video frames, maps raw eye coordinates to screen positions using a trained k–Nearest Neighbors (kNN) regression model, and analyzes gaze/fixation patterns during touchscreen interactions.

> **Key Features:**
>
> - **Offline Video Processing:** Analyze pre–recorded trial videos.
> - **DeepLabCut–Based Landmark Detection:** Use a trained DLC model to detect key eye landmarks (pupils and corners).
> - **Gaze Mapping (kNN):** Convert eye coordinates to screen coordinates using a kNN regressor trained on calibration data.
> - **Visualization:** Generate heatmaps and CSV summaries of time spent in each screen region.
> - **Cross–Platform Support:** Runs on both macOS and Windows (CPU–only supported).
> - **Optional Head–Pose Estimation:** Includes head roll estimation using facial landmarks.

---

## Table of Contents

1. [Installation and Setup](#installation-and-setup)  
2. [Data Preparation](#data-preparation)  
3. [DeepLabCut Model Training](#deeplabcut-model-training)  
4. [Pipeline Overview](#pipeline-overview)  
5. [Step 1: Extract Eye Landmarks for Calibration](#step-1-extract-eye-landmarks-for-calibration)  
6. [Step 2: (Optional) Merge Gaze Data with Click Data](#step-2-optional-merge-gaze-data-with-click-data)  
7. [Step 3: Train the kNN Mapping Model](#step-3-train-the-knn-mapping-model)  
8. [Step 4: Process Experimental Videos (Extract Landmarks)](#step-4-process-experimental-videos-extract-landmarks)  
9. [Step 5: Analyze Gaze (Generate Heatmaps & Time Spent)](#step-5-analyze-gaze-generate-heatmaps--time-spent)  
10. [Fine-Tuning or Retraining the kNN Model](#step-10-fine-tuning-or-retraining-the-knn-model)  
11. [Troubleshooting](#troubleshooting)  
12. [Future Improvements](#future-improvements)

---

## 1. Installation and Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ahrebel/rhesustracking.git
   cd rhesustracking
   ```
2. **Set up your Python environment (using Conda is recommended):**
   ```bash
   conda create -n monkey-gaze-tracker -c conda-forge python=3.8 pytables hdf5 lzo opencv numpy pandas matplotlib scikit-learn scikit-image scipy tqdm statsmodels
   conda activate monkey-gaze-tracker
   ```
3. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Install additional DeepLabCut dependencies:**
   ```bash
   pip install deeplabcut pyyaml tensorflow tensorpack tf-slim 'deeplabcut[gui]'
   ```
   > *If you encounter a `ModuleNotFoundError: No module named 'keras.legacy_tf_layers'`, run:*
   > ```bash
   > pip install --upgrade tensorflow_macos==2.12.0
   > ```
   > *(Adjust for your system as needed.)*

---

## 2. Data Preparation

- **Video Files:**  
  Place your trial videos (e.g., `1.mp4`, `2.mp4`, etc.) in a folder such as `videos/input/`.

- **(Optional) Touch/Click Event Files:**  
  If you use additional calibration data (e.g., from clicks/touches), save them as CSV files with columns like `time, screen_x, screen_y` (or `time, click_x, click_y`).

---

## 3. DeepLabCut Model Training

1. **Launch the DLC GUI:**
   ```bash
   python -m deeplabcut
   ```
2. **Create a New Project:**  
   - Enter your project name and add your video(s).  
   - Label keypoints (e.g., `left_pupil`, `right_pupil`, `corner_left`, `corner_right`).
3. **Label Frames:**  
   Label a diverse set of frames to cover different head positions and lighting.
4. **Train and Evaluate:**  
   Train the network (consider lighter models like `mobilenet_v2_1.0` for CPU-only) and evaluate its performance.
5. **Update Configuration:**  
   Ensure your scripts (e.g., `detect_eye.py`) point to your DLC project’s `config.yaml`.

---

## 4. Pipeline Overview

1. **Calibration:** Extract eye landmarks from a calibration video.
2. **(Optional) Merge Data:** Merge gaze landmarks with click/touch event data to form a calibration CSV.
3. **Mapping Model Training:** Train a kNN regression model to map eye landmarks to screen coordinates.
4. **Experimental Processing:** Process experimental videos to extract raw landmarks.
5. **Gaze Analysis:** Convert eye landmarks to screen coordinates using the kNN model, divide the screen into regions, and generate heatmaps and summaries.

---

## 5. Step 1: Extract Eye Landmarks for Calibration

Run the video processing script to extract eye landmarks:
```bash
python src/process_video.py --video /path/to/calibration_video.mp4 --config /path/to/dlc_config.yaml --output landmarks_output.csv
```
Example:
```bash
python src/process_video.py --video /Users/anthonyrebello/rhesustracking/videos/input/3.mp4 --config /Users/anthonyrebello/rhesustracking/eyetracking-ahrebel-2025-02-26/config.yaml --output landmarks_output.csv
```
The output CSV should contain columns such as:  
`frame, time, left_pupil_x, left_pupil_y, right_pupil_x, right_pupil_y, corner_left_x, corner_left_y, corner_right_x, corner_right_y, roll_angle`.

---

## 6. Step 2: (Optional) Merge Gaze Data with Click Data

If you have click/touch data with known screen coordinates, merge them with your gaze data:
```bash
python src/combine_gaze_click.py --gaze_csv landmarks_output.csv --click_file /path/to/your_click_file.csv --output_csv calibration_data_for_training.csv --max_time_diff 0.1
```
After merging, your calibration CSV should include:
```
time, left_corner_x, left_corner_y, right_corner_x, right_corner_y,
left_pupil_x, left_pupil_y, right_pupil_x, right_pupil_y, screen_x, screen_y
```
*(The script renames and organizes the columns as needed.)*

---

## 7. Step 3: Train the kNN Mapping Model

Train the kNN mapping model using your calibration CSV. This model will learn to predict screen coordinates from the eight eye landmark features. In your terminal, run:
```bash
python src/train_knn_mapping.py --data calibration_data_for_training.csv --output data/trained_model/knn_mapping_model.joblib --neighbors 5
```
- `--neighbors` specifies the number of neighbors to use in kNN (default is 5).  
- The trained model is saved (using joblib) to the specified output file.

---

## 8. Step 4: Process Experimental Videos (Extract Landmarks)

Process your experimental videos with the same processing script used for calibration:
```bash
python src/process_video.py --video /path/to/experimental_video.mp4 --config /path/to/dlc_config.yaml --output landmarks_output.csv
```
This will produce a landmarks CSV (e.g., `landmarks_output.csv`) with the eye landmark positions and a time column.

---

## 9. Step 5: Analyze Gaze (Generate Heatmaps & Time Spent)

Use an analysis script (e.g., `analyze_gaze_knn.py`) to:
- Load the experimental landmarks CSV.
- Convert the eye landmarks to screen coordinates using your trained kNN model.
- Divide the screen into a grid (e.g., 3×3).
- Compute the total time spent in each grid region (using frame duration derived from the `time` column).
- Generate and save a heatmap image and a CSV file with section durations.

Run:
```bash
python src/analyze_gaze_knn.py --landmarks_csv landmarks_output.csv --model data/trained_model/knn_mapping_model.joblib --screen_width 1920 --screen_height 1080 --n_cols 3 --n_rows 3 --output_heatmap gaze_heatmap.png --output_sections section_durations.csv
```

---

## 10. Fine-Tuning or Retraining the kNN Model

When new calibration data is available:
1. **Combine** the new calibration data with your existing CSV (ensure the same column format).
2. **Retrain the kNN model:**  
   ```bash
   python src/train_knn_mapping.py --data combined_calibration_data.csv --output data/trained_model/knn_mapping_model.joblib --neighbors 5
   ```
3. **Tune Hyperparameters:**  
   Experiment with different values for `--neighbors` for optimal performance.

---

## 11. Troubleshooting

- **Uniform Heatmap (All Regions Show the Same Color):**  
  - Verify that the raw landmark data from `process_video.py` shows variability over time.
  - Ensure your calibration data covers a wide range of screen positions.
  - Print sample predicted screen coordinates (e.g., via debug statements) to verify the kNN model output.
  
- **Mapping Accuracy Issues:**  
  - Check that the input features (eye landmarks) are properly normalized or scaled.
  - Adjust the number of neighbors or try additional feature engineering.

- **Data Format Issues:**  
  - Confirm that your CSV files contain the required columns in the correct format.
  - Ensure time values are either numeric or parseable as datetime objects.

---

## 12. Future Improvements

- **Advanced Feature Engineering:**  
  Consider incorporating additional features (e.g., distances or angles between landmarks) that might improve the mapping accuracy.
  
- **Model Comparisons:**  
  Experiment with alternative regression models (e.g., SVR, Random Forests) and compare their performance via cross-validation.
  
- **Adaptive Calibration:**  
  Develop an online or real-time calibration method to continuously update the mapping model during use.
  
- **Enhanced Visualization:**  
  Create interactive dashboards or real-time displays for dynamic gaze visualization.

---

**Happy Tracking!**
