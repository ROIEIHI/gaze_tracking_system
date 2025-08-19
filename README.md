# Eye Tracking System - Modular Architecture

A sophisticated eye tracking system built with Python, OpenCV, MediaPipe, and XGBoost for real-time gaze prediction.

## 🎯 Features

- **Modular Architecture**: Clean separation of concerns across multiple files
- **21-Point Calibration**: Comprehensive calibration system with enhanced GUI
- **Real-time Prediction**: Smooth gaze tracking with customizable smoothing factor
- **Boundary Detection**: Automatic face position monitoring with warnings
- **Black Screen Prediction**: Clean prediction interface with red dot visualization
- **Data Export**: CSV export of calibration data for analysis
- **Machine Learning**: XGBoost-based gaze prediction model

## 🏗️ Architecture

```
├── main.py           # Entry point and application launcher
├── tracker.py        # Main EyeTracker class with complete pipeline
├── config.py         # Centralized configuration constants
├── ui_utils.py       # UI drawing and interaction utilities
├── model_utils.py    # Machine learning model utilities
├── feature_ex.py     # Feature extraction from face landmarks
└── requirements.txt  # Python dependencies
```

## 🚀 Quick Start

### Prerequisites

```bash
pip install opencv-python mediapipe numpy pandas xgboost scikit-learn matplotlib seaborn
```

### Running the Application

```bash
python main.py
```

## 📋 Workflow

1. **User Positioning**: Position your face within the boundary box
2. **Calibration**: Click on 21 calibration targets displayed on screen
3. **Model Training**: Automatic training of XGBoost prediction model
4. **Real-time Prediction**: View gaze prediction on black screen with red dot

## ⚙️ Configuration

Edit `config.py` to customize:

- Window dimensions
- Calibration parameters  
- Boundary settings
- Smoothing factors
- Model parameters

## 🎛️ Key Components

### EyeTracker Class (tracker.py)
- Complete pipeline orchestration
- User positioning with boundary checking
- 21-point calibration system
- Real-time prediction with smoothing

### UI Utilities (ui_utils.py)
- Calibration target visualization
- Boundary box drawing
- Warning systems
- Animation effects

### Feature Extraction (feature_ex.py)
- MediaPipe face landmark processing
- Iris position calculation
- Head pose estimation
- Face bounding box detection

### Model Utilities (model_utils.py)
- XGBoost model training
- Data preprocessing
- Performance evaluation
- Prediction smoothing

## 🔧 Technical Details

- **Face Detection**: MediaPipe Face Mesh with 468 landmarks
- **Machine Learning**: XGBoost MultiOutputRegressor
- **Calibration**: 21 strategically placed targets (corners, edges, grid)
- **Smoothing**: Exponential moving average for stable predictions
- **Export**: Timestamped CSV files with calibration data

## 📊 Performance

The system achieves real-time performance with:
- Face detection: ~30 FPS
- Gaze prediction: Sub-millisecond inference
- Calibration: ~2-3 minutes for full setup

## 🛠️ Development

### Original Refactoring

This modular system was refactored from a monolithic `2D_eyeonly.py` file to achieve:
- Better code organization
- Easier maintenance
- Improved testability
- Enhanced readability

### Recent Enhancements

- Enhanced calibration GUI with full-screen targets
- Boundary checking during all phases
- Smoothed prediction display
- Comprehensive error handling

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.
