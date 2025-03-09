Below is the reorganized version of your README. It maintains the same content while improving readability and visual structure.

```markdown
# PrimateVision

![Animated Eyes](https://github.com/user-attachments/assets/0f245b14-ec20-4a11-868a-ae207a7dfa1d)

PrimateVision implements an eye–tracking system for Rhesus macaques (and humans) using **DeepLabCut (DLC)**. It detects eye landmarks from video frames, maps raw eye coordinates to screen positions using a trained k–Nearest Neighbors (kNN) regression model, and analyzes gaze/fixation patterns during touchscreen interactions.

> **Key Features:**
> - **Offline Video Processing:** Analyze pre–recorded trial videos.
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

*If a `ModuleNotFoundError: No module named 'keras.legacy_tf_layers'` occurs, run:*

```bash
pip install --upgrade tensorflow_macos==2.12.0
```

---

## 2. Data Preparation

- **Videos:**  
  Place trial videos (e.g., `1.mp4`, `2.mp4`, etc.) in the `videos/input/` folder.
  
- **Touch/Click Event Files (Optional):**  
  Save additional calibration data as CSV files with columns such as `time, screen_x, screen_y`.

---

## 3. DeepLabCut Model Training

1. **Launch the DLC GUI:**

   ```bash
   python -m deeplabcut
   ```

2. **Create a New Project:**  
   Specify the project name, add the video(s), and define keypoints (e.g., `left_pupil`, `right_pupil`, `corner_left`, `corner_right`).

3. **Label Frames:**  
   Label a diverse set of frames covering various head poses and lighting conditions.

4. **Train and Evaluate:**  
   Train the network (consider a lightweight model such as `mobilenet_v2_1.0` for CPU-only systems) and evaluate its performance.

5. **Update Configuration:**  
   Ensure that the scripts (e.g., `detect_eye.py`) correctly reference the DLC project’s `config.yaml`.

---

## 4. Pipeline Overview

The PrimateVision pipeline includes the following steps:

1. **Calibration:**  
   Extract eye landmarks from a calibration video.
2. **(Optional) Data Merging:**  
   Combine gaze data with touch/click events to create a comprehensive calibration CSV.
3. **Mapping Model Training:**  
   Train a kNN regression model to convert raw eye coordinates to screen coordinates.
4. **Experimental Processing:**  
   Process experimental videos to extract raw landmarks.
5. **Gaze Analysis:**  
   Apply the trained kNN model to compute screen positions, segment the screen into regions, and calculate fixation durations.

---

## 5. Calibration & Processing

### Extract Eye Landmarks for Calibration

Run the processing script with multiple workers:

```bash
python src/process_video.py --video /path/to/calibration_video.mp4 --config /path/to/dlc_config.yaml --output landmarks_output.csv --workers 4
```

*Example:*

```bash
python src/process_video.py --video videos/input/3.mp4 --config eyetracking-ahrebel-2025-02-26/config.yaml --output landmarks_output.csv --workers 4
```

The output CSV should include columns like:

```
frame, time, left_pupil_x, left_pupil_y, right_pupil_x, right_pupil_y, corner_left_x, corner_left_y, corner_right_x, corner_right_y, roll_angle
```

### Merge Gaze Data with Click Data (Optional)

```bash
python src/combine_gaze_click.py --gaze_csv landmarks_output.csv --click_file /path/to/your_click_file.csv --output_csv calibration_data_for_training.csv --max_time_diff 0.1
```

---

## 6. Gaze Analysis & Visualization

### Train the kNN Mapping Model

```bash
python src/train_knn_mapping.py --data calibration_data_for_training.csv --output data/trained_model/knn_mapping_model.joblib --neighbors 5
```

### Process Experimental Videos

```bash
python src/process_video.py --video /path/to/experimental_video.mp4 --config /path/to/dlc_config.yaml --output landmarks_output.csv --workers 4
```

### Analyze Gaze and Generate Visualizations

```bash
python src/analyze_gaze_knn.py --landmarks_csv landmarks_output.csv --model data/trained_model/knn_mapping_model.joblib --screen_width 1920 --screen_height 1080 --n_cols 3 --n_rows 3 --output_heatmap gaze_heatmap.png --output_sections section_durations.csv
```

This command converts eye landmarks into screen coordinates, segments the screen into regions, and outputs a heatmap along with a summary CSV.

---

## 7. Testing, CI, and Contribution Guidelines

- **Testing:**  
  Unit and integration tests have been added for key components (e.g., calibration and kNN mapping) to ensure reliable performance.
  
- **Continuous Integration (CI):**  
  GitHub Actions is configured to run tests and lint checks on every push and pull request.
  
- **Contribution Guidelines:**  
  See [CONTRIBUTING.md](CONTRIBUTING.md) for details on submitting pull requests and reporting issues.

---

## 8. Future Improvements

- **Advanced Feature Engineering:**  
  Explore additional features (e.g., inter-landmark distances) to improve mapping accuracy.
- **Model Comparisons:**  
  Experiment with alternative regression models (e.g., SVR, Random Forests) via cross-validation.
- **Adaptive Calibration:**  
  Develop an online calibration method for real-time mapping updates.
- **Enhanced Visualization:**  
  Create interactive dashboards or real-time displays for dynamic gaze analysis.
- **Add Docker Support:**
  Add a Dockerfile for containerized setup.

---

## 9. License

This project is licensed under the [GPL-3.0 License](LICENSE). Please refer to the license file for details.
