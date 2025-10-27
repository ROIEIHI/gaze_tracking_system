# Gaze Tracking System - Code Modules Explanation

This document provides a comprehensive overview of each module in the gaze tracking system, explaining their role, functionality, and how they fit into the overall project architecture.

---

## **Core System Modules**

### **`start.py`** - System Launcher
- **Role**: Entry point and system launcher
- **Functionality**: 
  - Cross-platform launcher script
  - Dependency checking for critical packages
  - Quick system startup with error handling
- **Project Part**: System initialization and validation
- **Dependencies**: Checks for cv2, mediapipe, numpy, pandas, sklearn, bidi.algorithm, arabic_reshaper, tkinter

---

### **`main.py`** - Command Line Interface
- **Role**: Main entry point for console-based system operation
- **Functionality**:
  - Coordinates the complete pipeline (calibration → training → prediction)
  - Command-line interface for users who prefer console over GUI
  - Sequential execution of system phases
  - Mainly used for debugging 
- **Project Part**: Alternative interface to GUI, provides full system workflow
- **Key Features**: User input handling, phase coordination, error management

---

### **`gui.py`** - Professional GUI Interface
- **Role**: Graphical user interface for the entire system
- **Functionality**:
  - Professional tkinter-based interface
  - User session management and progress tracking
  - Integration of all system modules with comprehensive error handling
  - Real-time status updates and user feedback
- **Project Part**: Primary user interface
- **Components**: Session setup, calibration interface, training monitoring, prediction visualization

---

##  **Core Processing Modules**

### **`calibration.py`** - Gaze Calibration System
- **Role**: Handles face detection, feature extraction, and calibration data collection
- **Functionality**:
  - MediaPipe Face Mesh integration for 468 3D facial landmarks
  - Calibration point generation (30-point optimized grid)
  - Head pose estimation using OpenCV's solvePnP algorithm
  - Baseline pitch/yaw calculation for position-invariant features
  - Real-time face stability tracking and boundary warnings
- **Project Part**: Phase 1 of the pipeline - data collection for personalized model training
- **Output**: Calibration CSV files with engineered features ready for ML training

---

### **`model_training.py`** - Machine Learning Training Pipeline
- **Role**: Feature engineering, outlier detection, and model training
- **Functionality**:
  - Advanced feature engineering (11 predictive features)
  - Statistical outlier detection and removal
  - Multi-output XGBoost regression model training
  - Cross-validation and hyperparameter optimization
  - Model performance evaluation and metrics calculation
- **Project Part**: Phase 2 of the pipeline - creates personalized gaze prediction models
- **Output**: Trained models (.pkl files) with performance metrics and scalers

---

### **`prediction.py`** - Real-time Gaze Prediction & Text Analysis
- **Role**: Real-time gaze prediction with comprehensive text reading analysis
- **Functionality**:
  - Real-time gaze coordinate prediction using trained models
  - Hebrew RTL and English LTR text rendering with proper bidirectional support
  - Integration with EyeMovementAnalyzer for reading behavior analysis
  - Multi-page text support with language-appropriate navigation
  - CSV data export for research and analysis
- **Project Part**: Phase 3 of the pipeline - practical application of trained models
- **Key Features**: Text rendering, reading session management, data collection

---

### **`eye_movement_analyzer.py`** - Movement Analysis & Signal Processing
- **Role**: Advanced signal processing and movement classification
- **Functionality**:
  - Dual-stage Kalman filtering (base + adaptive)
  - Velocity-adaptive filtering with dynamic process noise adjustment
  - Fixation and saccade detection with configurable thresholds
  - Reading direction-aware analysis (RTL/LTR support)
  - Real-time movement metrics calculation (WPM, regression analysis)
- **Project Part**: Signal processing layer that converts raw predictions into stable, analyzable data
- **Classes**: `KalmanFilter`, `EyeMovementAnalyzer`, `GazePoint`, `MovementMetrics`

---

##  **Configuration & Setup Modules**

### **`config.py`** - System Configuration
- **Role**: Central configuration management
- **Functionality**:
  - All system constants and settings in one place
  - Screen, camera, and display parameters
  - Calibration process parameters (frames, grid size, margins)
  - Movement analysis thresholds and filtering parameters
  - File paths and directory structure management
- **Project Part**: Configuration layer ensuring consistent parameters across all modules
- **Categories**: System settings, calibration settings, filtering parameters, text analysis settings

---

### **`setup.py`** - Automated Installation
- **Role**: Cross-platform dependency installation and system setup
- **Functionality**:
  - Python version compatibility checking
  - Automated pip package installation
  - System requirements verification
  - Platform-specific dependency handling
  - Installation error handling and reporting
- **Project Part**: System setup and environment preparation
- **Features**: Dependency checking, error handling, platform detection

---

## **Analysis & Visualization**

### **`visualize_reading_session.py`** - Data Visualization Tool
- **Role**: Post-session analysis and visualization (located in parent directory)
- **Functionality**:
  - Comprehensive visualization of CSV eye tracking data
  - Spatial heatmaps showing fixation patterns
  - Reading path visualization with temporal information
  - Velocity analysis and movement classification display
  - Hebrew RTL text support with proper character ordering
  - Word-level reading analysis avoiding duplicate counting
- **Project Part**: Post-processing analysis tool for research and insights
- **Visualizations**: Heatmaps, reading paths, velocity timelines, movement statistics

---

##  **Data Structure & Organization**

### **Directory Structure**
```
system/
├── calibration_data/     # Calibration CSV files
├── models/              # Trained ML models (.pkl files)
├── eye_tracking_data/   # Raw prediction output
├── movement_data/       # Processed movement analysis
└── user_data/          # User sessions with organized subdirectories
    └── username_timestamp/
        ├── calibration/     # Session-specific calibration
        ├── models/         # Session-specific models
        └── analysis/       # Session analysis results
```

---

