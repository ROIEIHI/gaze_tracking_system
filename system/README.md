# Advanced Eye Movement Tracking System

A professional multilingual eye tracking system with comprehensive support for Hebrew RTL and English LTR text reading analysis. The system combines computer vision, machine learning, and advanced signal processing to provide accurate gaze tracking and movement analysis for reading research.

## System Overview

This system provides real-time eye tracking with specialized support for bidirectional text analysis. It uses MediaPipe for facial landmark detection, implements custom Kalman filtering for smooth gaze prediction, and includes RTL-aware text rendering for Hebrew research applications. The system follows a complete workflow from calibration through model training to real-time analysis with comprehensive data export capabilities.

## Quick Setup

Navigate to the system directory and run the automated setup:

**Windows:**
cd system/
setup.bat

**All Platforms:**
cd system/
python setup.py


For detailed installation instructions and troubleshooting, see SETUP_README.md.

## Running the System

Launch the GUI interface:

python gui.py


The interface provides a complete workflow with user session management, progress tracking, and automated directory organization.

## Calibration Process

The system uses a strategic calibration approach combining targeted points and comprehensive coverage:

- **Strategic Points**: 21 carefully positioned points targeting screen corners, edges, and center regions
- **Grid Calibration**: Configurable density grid (6x6 default) for comprehensive coverage  
- **Face Detection**: Real-time MediaPipe integration for consistent landmark tracking
- **Pitch Baseline**: User-initiated baseline measurement for improved vertical accuracy
- **Quality Control**: Automatic outlier detection and data validation

Position yourself 60-80cm from the screen and follow the red calibration points. Keep your head stable during each measurement phase.

## Model Training and Data Engineering

The system employs a multi-output RandomForest approach optimized for spatial accuracy:

**Feature Engineering:**
- Raw eye positions (left/right normalized coordinates)
- Head pose parameters (yaw, pitch with baseline adjustment, roll)
- Interaction terms (eye position × head orientation)
- Final feature set: 9 engineered features

**Training Process:**
- Multi-output RandomForest for joint X/Y coordinate prediction
- Custom Euclidean distance scorer for spatial optimization
- GridSearchCV hyperparameter tuning with cross-validation
- Model evaluation using spatial accuracy metrics

**Performance Characteristics:**
- Euclidean distance error: ~110 pixels typical
- R² scores: X-coordinate ~0.90, Y-coordinate ~0.78
- Cross-validated training with custom spatial scorer

## Prediction and Text Analysis

The prediction system includes advanced movement analysis with RTL support:

**Gaze Prediction:**
- Real-time coordinate prediction using trained RandomForest model
- Optional exponential smoothing for stable tracking
- Confidence-based filtering for robust predictions

**Kalman Filtering:**
- Dual-mode filtering supporting LTR and RTL reading patterns
- RTL mode: Leftward bias (-1.2), higher process noise for Hebrew variability
- LTR mode: Rightward bias (+1.0), standard parameters for English
- Automatic language detection based on text content analysis

**Text Rendering:**
- Hebrew RTL rendering using PIL with proper bidirectional text processing
- System font integration with automatic Hebrew font detection
- RTL word positioning and layout matching visual text flow
- Multi-page support with language-appropriate navigation

**Movement Analysis:**
- Fixation detection with configurable velocity and duration thresholds
- Saccade classification with direction-aware parameters for RTL/LTR
- Reading metrics including WPM, regression analysis, and line tracking
- Real-time velocity analysis and movement classification

## System Structure

system/
├── gui.py # Professional GUI interface
├── main.py # Command line interface
├── calibration.py # Strategic calibration system
├── model_training.py # RandomForest training pipeline
├── prediction.py # RTL-aware prediction and text analysis
├── eye_movement_analyzer.py # Kalman filtering and movement analysis
├── config.py # System configuration parameters
└── requirements.txt # Python dependencies


**Data Organization:**
- Session-specific directories: `username_timestamp/`
- Calibration data: `username_timestamp/calibration/`
- Trained models: `username_timestamp/models/`
- Analysis output: `username_timestamp/analysis/`

## Technical Implementation

**Computer Vision:** MediaPipe facial landmark detection with 468 3D face landmarks
**Machine Learning:** Multi-output RandomForest with custom Euclidean optimization
**Signal Processing:** Kalman filtering with reading-direction adaptive parameters
**Text Processing:** Bidirectional text rendering with hebrew-reshaper and python-bidi
**Data Export:** CSV format compatible with standard eye tracking research tools

## Requirements

- Python 3.8+
- Webcam (720p+ recommended)
- 8GB RAM minimum, 16GB recommended
- Windows 10+, macOS 10.15+, or Linux Ubuntu 20.04+

## Key Dependencies

- OpenCV, MediaPipe (computer vision)
- scikit-learn, NumPy, pandas (machine learning)
- PIL, python-bidi, arabic-reshaper (text rendering)
- tkinter (GUI framework)

For complete installation details, dependency versions, and troubleshooting guidance, refer to SETUP_README.md.

