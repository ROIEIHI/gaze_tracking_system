# Advanced Eye Movement Tracking System

A professional multilingual eye tracking system with comprehensive support for Hebrew RTL and English LTR text reading analysis. The system combines computer vision, machine learning, and advanced signal processing to provide accurate gaze tracking and movement analysis for reading research.

## System Overview

This system provides real-time eye tracking with specialized support for bidirectional text analysis. It uses MediaPipe for facial landmark detection, implements custom Kalman filtering for smooth gaze prediction, and includes RTL-aware text rendering for Hebrew research applications. The system follows a complete workflow from calibration through model training to real-time analysis with comprehensive data export capabilities.

## Quick Setup

Navigate to the system directory and run the automated setup:

### Windows
```batch
cd system/
setup.bat
```

### All Platforms
```bash
cd system/
python setup.py
```

For detailed installation instructions and troubleshooting, see `SETUP_README.md`.

## Running the System

Launch the GUI interface:

```bash
python gui.py
```

The interface provides a complete workflow with user session management, progress tracking, and automated directory organization.

## Calibration Process

The system uses a strategic calibration approach combining targeted points and comprehensive coverage:

### Calibration Features
- **Strategic Points**: 21 carefully positioned points targeting screen corners, edges, and center regions
- **Grid Calibration**: Configurable density grid (6x6 default) for comprehensive coverage  
- **Face Detection**: Real-time MediaPipe integration for consistent landmark tracking
- **Pitch Baseline**: User-initiated baseline measurement for improved vertical accuracy
- **Quality Control**: Automatic outlier detection and data validation

### Setup Instructions
Position yourself 60-80cm from the screen and follow the red calibration points. Keep your head stable during each measurement phase.

## Model Training and Data Engineering

The system employs a multi-output RandomForest approach optimized for spatial accuracy:

### Feature Engineering
```python
# Raw Features
norm_x_L, norm_y_L    # Left eye normalized coordinates
norm_x_R, norm_y_R    # Right eye normalized coordinates
yaw, pitch, roll      # Head pose parameters

# Engineered Features
pitch = pitch_raw - pitch_baseline    # Baseline-adjusted pitch
x_yaw_interaction = avg_norm_x * yaw  # Eye-head interaction
y_pitch_interaction = avg_norm_y * pitch
```

### Training Process
- **Multi-output RandomForest** for joint X/Y coordinate prediction
- **Custom Euclidean distance scorer** for spatial optimization
- **GridSearchCV hyperparameter tuning** with cross-validation
- **Model evaluation** using spatial accuracy metrics

### Performance Characteristics
| Metric | Value |
|--------|--------|
| Euclidean distance error | ~110 pixels (typical) |
| X-coordinate R² | ~0.90 |
| Y-coordinate R² | ~0.78 |
| Validation | Cross-validated with custom spatial scorer |

## Prediction and Text Analysis

The prediction system includes advanced movement analysis with RTL support:

### Gaze Prediction
- **Real-time coordinate prediction** using trained RandomForest model
- **Optional exponential smoothing** for stable tracking
- **Confidence-based filtering** for robust predictions

### Kalman Filtering
```python
# RTL Mode (Hebrew)
reading_direction_bias = -1.2    # Leftward bias
process_noise = 0.15             # Higher variability
saccade_direction = -1           # Leftward saccades

# LTR Mode (English)  
reading_direction_bias = 1.0     # Rightward bias
process_noise = 0.1              # Standard variability
saccade_direction = 1            # Rightward saccades
```

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
├── setup.bat                 # Windows automated setup
├── setup.py                  # Cross-platform setup
├── start.bat                 # Windows launcher
└── start.py                  # Cross-platform launcher
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
| **Machine Learning** | Multi-output RandomForest | Custom Euclidean optimization |
| **Signal Processing** | Kalman filtering | Reading-direction adaptive parameters |
| **Text Processing** | python-bidi, arabic-reshaper | Bidirectional text rendering |
| **Data Export** | CSV format | Standard eye tracking research tools |

## System Requirements

### Minimum Requirements
- **Python**: 3.8 or higher
- **Camera**: USB webcam (720p+ recommended)
- **Memory**: 8GB RAM minimum, 16GB recommended
- **Storage**: 2GB free space
- **OS**: Windows 10+, macOS 10.15+, or Linux Ubuntu 20.04+

### Key Dependencies

| Category | Libraries | Purpose |
|----------|-----------|---------|
| **Computer Vision** | opencv-python, mediapipe | Camera access and face detection |
| **Machine Learning** | scikit-learn, numpy, pandas | Model training and data processing |
| **Text Processing** | pillow, python-bidi, arabic-reshaper | Hebrew RTL text rendering |
| **GUI Framework** | tkinter | User interface |

## Installation and Support

For complete installation details, dependency versions, and troubleshooting guidance, refer to [`SETUP_README.md`](SETUP_README.md).

