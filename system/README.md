# Gaze Tracking and Reading Analysis System

This project is an end-to-end, professional system for real-time gaze tracking and advanced reading analysis using only a standard off-the-shelf RGB webcam.

The system is built on a pipeline combining modern computer vision, a highly-optimized machine learning regressor, and advanced signal processing to transform noisy webcam data into stable, research-grade metrics. It features robust support for multilingual text analysis, including Hebrew (RTL) and English (LTR), by incorporating direction-aware adaptive filtering.

## System Pipeline & Methodology

The core of the system is a feature-based appearance pipeline (as described in ) that maps facial features to on-screen coordinates. This approach avoids expensive, specialized hardware and enables real-time performance on consumer devices.

## Quick Setup

Navigate to the system directory and run the automated setup:

```bash
cd system/
python setup.py
```

For detailed installation instructions and troubleshooting, see `SETUP_README.md`.

## Running the System

Launch the GUI interface:

```bash
python start.py
```

The interface provides a complete workflow with user session management, progress tracking, and automated directory organization.

## Pipeline

The pipeline proceeds as follows:

### 1. Landmark Detection: 
Captures real-time video and uses Google's MediaPipe Face Mesh to extract 468 3D facial landmarks, including precise iris localization.

### 2. Head Pose Estimation: 
Employs OpenCV's solvePnP algorithm to estimate the 3D head orientation (rotation and translation vectors) from a set of 6 stable facial landmarks.

### 3. Calibration:
An optimized calibration process is used to gather a user-specific training dataset. Based on empirical analysis, the optimal balance between accuracy and user fatigue was found to be:
- **30 Calibration Points:** A specific grid layout (see Figure 7) ensures full-screen coverage.
- **11 Frames per Point:** Capturing a burst of 11 frames per target provided the lowest prediction error while minimizing overfitting (see Figure 8).

### 4. Feature Engineering: 
A set of 11 predictive features is engineered from the raw landmark data. This is the final feature vector used for training:
- **Normalized Pupil Coordinates (4 features):** Left and right pupil (x, y) coordinates, normalized relative to the facial bounding box to be position-invariant.
- **Head Pose (2 features):** Baseline-adjusted *Pitch* and *Yaw*. A center-fixation phase at calibration start establishes this baseline.
- **Translation Vector (3 features):** The *tvec* [x, y, z] output from solvePnP, representing the head's position in space
- **Interation Terms (2 features):** Cross-term features that capture the dependency between head position and eye orientation (e.g., tvect-x multiplied by normalized horizontal pupil position).

### 5. Model Training: 
The system was benchmarked against multiple regression models (RandomForest, Neural Networks, SVR). The best-performing model was a Multi-Output XGBoost Regressor.This model outperformed all others, including a two-model (separate X/Y) XGBoost approach, achieving a Cross-Validated best RMSE of 31.17 pixels

## 6. Prediction and Analysis
The prediction pipeline (see Figure 12) uses a two-stage filtering process for maximum stability and analytic precision:
- **Base Kalman Filter:** A simple Kalman filter first stabilizes the raw, noisy (x, y) predictions from the XGBoost model.
- **Adaptive Kalman Filter** A second, more advanced analyzer models the gaze as a 4D state vector ([x, y, v_x, v_y]). This filter is velocity-adaptive: its process noise parameter Q is dynamically adjusted in real-time. Q is lowered for slow movements (fixations) to increase smoothing and raised for rapid movements (saccades) to improve responsiveness.

### Setup Instructions
Position yourself 60-80cm from the screen and follow the red calibration points. Keep your head stable during each measurement phase.



### Performance Characteristics
| Metric | Value |
|--------|--------|
| Euclidean distance error | ~50 pixels (typical) |
| R² Test Score | ~0.97 |
| Overfitting Score | ~0.022 |

### Text Rendering
- **Hebrew RTL rendering** using PIL with proper bidirectional text processing
- **System font integration** with automatic Hebrew font detection
- **RTL word positioning** and layout matching visual text flow
- **Multi-page support** with language-appropriate navigation

### Movement Analysis
- **Fixation detection** with configurable velocity and duration thresholds
- **Saccade classification** with direction-aware parameters for RTL/LTR
- **Reading metrics** including WPM, regression analysis, and line tracking
- **Real-time velocity analysis** and movement classification

## System Structure

```
system/
├── gui.py                    # Professional GUI interface
├── main.py                   # Command line interface
├── calibration.py            # Strategic calibration system
├── model_training.py         # RandomForest training pipeline
├── prediction.py             # RTL-aware prediction and text analysis
├── eye_movement_analyzer.py  # Kalman filtering and movement analysis
├── config.py                 # System configuration parameters
├── requirements.txt          # Python dependencies
├── setup.py                  # System setup script
└── start.py                  # System launcher
```

### Data Organization
```
user_sessions/
└── username_timestamp/
    ├── calibration/          # Session calibration data
    ├── models/              # Session-specific models
    └── analysis/            # Analysis output files
```

## Technical Implementation

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Computer Vision** | MediaPipe | 468 3D facial landmarks detection |
| **Machine Learning** | Multi-output XGBoost | Custom Euclidean optimization |
| **Signal Processing** | Kalman filtering | Reading-direction adaptive parameters |
| **Text Processing** | python-bidi, arabic-reshaper | Bidirectional text rendering |
| **Data Export** | CSV format | Standard eye tracking research tools |

## System Requirements

### Minimum Requirements
- **Python**: 3.8 or higher
- **Camera**: USB webcam (720p+ recommended)
- **Memory**: 8GB RAM minimum, 16GB recommended
- **Storage**: 2GB free space
- **OS**: Windows 10+

### Key Dependencies

| Category | Libraries | Purpose |
|----------|-----------|---------|
| **Computer Vision** | opencv-python, mediapipe | Camera access and face detection |
| **Machine Learning** | scikit-learn, numpy, pandas | Model training and data processing |
| **Text Processing** | pillow, python-bidi, arabic-reshaper | Hebrew RTL text rendering |
| **GUI Framework** | tkinter | User interface |

## Installation and Support

For complete installation details, dependency versions, and troubleshooting guidance, refer to [`SETUP_README.md`](SETUP_README.md).

