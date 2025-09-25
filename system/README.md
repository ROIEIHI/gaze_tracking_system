# 🎯 Gaze Tracking Reading Analysis System

A **professional-grade eye tracking system** designed for reading behavior analysis and research. This system provides comprehensive gaze tracking capabilities with real-time analysis, adaptive text rendering, and detailed movement metrics suitable for academic research and commercial applications.

## ✨ Key Features

### 🎯 **Advanced Calibration System**
- **21-point strategic calibration** targeting corners, edges, and center regions
- **Additional grid calibration** with customizable density (6x6 default)
- **Adaptive margins** and intelligent duplicate detection
- **Real-time face detection** with MediaPipe integration
- **Pitch baseline calibration** for improved accuracy

### 📖 **Text Reading Analysis**
- **Adaptive text rendering** that scales to any screen size
- **Word-level fixation tracking** with precise positioning
- **Reading pattern detection** (fixations, saccades, regressions, return sweeps)
- **Multi-page text support** with navigation controls
- **Responsive UI** with configurable margins and font scaling

### 🧠 **Eye Movement Analysis**
- **Kalman filtering** for smooth gaze tracking
- **Movement classification** (fixations, saccades, smooth pursuit)
- **Reading-specific metrics** (WPM, regression rate, line changes)
- **Real-time velocity and direction analysis**
- **EyeMovementAnalyzer** with text reading mode

### 📊 **Data Export & Analysis**
- **CSV export** matching standard research formats
- **Comprehensive fixation data** with timestamps and durations
- **Word-level analysis** with proximity detection
- **Session-based tracking** with detailed metrics
- **Compatible with eye tracking research standards**

## 🚀 Quick Start

### 1. Prerequisites
```bash
# Required Python packages
pip install opencv-python mediapipe pandas numpy scikit-learn joblib
```

### 2. System Setup
```bash
# Run the main system
python main.py

# Follow the interactive menu:
# 1. Calibration (required first time)
# 2. Model Training (after calibration)
# 3. Reading Analysis (after training)
```

### 3. Calibration Process
- **Position yourself** 60-80cm from the screen
- **Follow the red circles** that appear on screen
- **Keep your head stable** during each calibration point
- **21 strategic points + grid points** for comprehensive coverage

### 4. Reading Analysis
- **ESC**: Exit and export data
- **A/D**: Navigate between text pages
- **E**: Export current session data
- **Fullscreen mode** with real-time gaze overlay

## 📁 System Architecture

```
system/
├── main.py                    # Main entry point and system coordinator
├── calibration.py            # Advanced calibration with 21+ points
├── model_training.py         # Multi-output RandomForest training
├── prediction.py             # Text reading analysis system
├── eye_movement_analyzer.py  # Movement analysis and metrics
├── config.py                 # System configuration and constants
└── README.md                 # This documentation
```

### 1. First-Time Setup (New Computer)
```bash
# Download and run the setup script
python setup_environment.py
```
This script will:
- Check Python version compatibility (3.8+)
- Install all required packages automatically
- Test camera and MediaPipe functionality
- Create necessary directory structure
- Verify installation

### 2. Manual Installation (Alternative)
```bash
pip install scikit-learn pandas numpy opencv-python mediapipe joblib
```

### 3. Run the System
```bash
python main.py
```

### 4. Choose Your Workflow
- **Run Calibration**: Collect training data (21-point calibration with pitch baseline)
- **Train Model**: Build optimized RandomForest model
- **Real-Time Prediction**: Use trained model for live gaze tracking
- **Complete Workflow**: Run all steps in sequence

## Setup Scripts

For new installations, the following helper scripts are available in the repository:
- `setup_environment.py` - Comprehensive automated setup for new computers
- `requirements.txt` - Package dependencies list  
- `validate_setup.py` - System validation tool
- `SETUP_SUMMARY.md` - Detailed documentation of the setup system

**Recommended**: Use `python setup_environment.py` for hassle-free installation on new systems.

## Project Structure

```
refactored/
├── main.py                    # Main GUI application
├── calibration.py             # 21-point calibration system with pitch baseline
├── model_training.py          # Multi-output RandomForest training
├── prediction.py              # Real-time gaze prediction
├── eye_movement_analyzer.py   # Movement analysis utilities
├── preview_text.py            # Text preview functionality
├── models/                    # Trained model storage
├── data/                      # Dataset storage
└── assets/                    # UI assets
```

## Core Components

### 1. Calibration System (`calibration.py`)
- MediaPipe-based facial landmark detection
- 21-point screen calibration pattern
- **Pitch Baseline Measurement**: Mouse-click triggered baseline collection (60 frames)
- Head pose estimation (yaw, pitch, roll) with pitch adjustment
- Iris position normalization
- Simplified 9-column feature output
- Outlier detection and data cleaning

### 2. Model Training (`model_training.py`)
- **Multi-Output RandomForest**: Single model for X/Y prediction
- **Custom Euclidean Scorer**: Minimizes spatial distance error
- **Feature Engineering**: 9 features including raw eye positions and interaction terms
- **GridSearchCV Optimization**: Hyperparameter tuning
- **Cross-Validation**: 3-fold validation with custom scorer

### 3. Real-Time Prediction (`prediction.py`)
- **Streamlined Pipeline**: Feature extraction → prediction → smoothing
- **Pitch Baseline Loading**: Uses calibration-time baseline for consistency
- **Exponential Smoothing**: Reduces prediction jitter
- **Eye Movement Analysis**: Optional movement tracking
- **Performance Optimized**: Minimal latency for real-time use

## 📊 Model Performance

### Training Configuration
- **Features**: 9 engineered features (norm_x_L, norm_y_L, norm_x_R, norm_y_R, yaw, pitch, roll, x_yaw_interaction, y_pitch_interaction)
- **Optimization**: Custom Euclidean distance scorer
- **Pitch Adjustment**: Calibration-time baseline measurement and adjustment
- **Cross-Validation**: 3-fold with custom scorer

### Performance Metrics
- **Euclidean Distance Error**: ~110 pixels (typical)
- **R² Score**: ~0.83-0.85 (average)
- **X-coordinate R²**: ~0.90-0.91
- **Y-coordinate R²**: ~0.77-0.78

## 🎛️ Usage Examples

### Programmatic Usage

```python
# Calibration
from calibration import EyeTrackerCalibrator
calibrator = EyeTrackerCalibrator()
csv_file = calibrator.run_calibration()

# Training
from model_training import GazeModelTrainer
trainer = GazeModelTrainer()
model_path = trainer.train_from_csv(csv_file)

# Prediction
from prediction import GazePredictor
predictor = GazePredictor(model_path)
predictor.run_real_time_prediction(enable_smoothing=True)
```

### Feature Engineering
The system uses 9 engineered features from raw MediaPipe outputs:

```python
# Raw features: [norm_x_L, norm_y_L, norm_x_R, norm_y_R, yaw, pitch_raw, roll]
# Engineered features:
pitch = pitch_raw - pitch_baseline  # Baseline-adjusted pitch
avg_norm_x = (norm_x_L + norm_x_R) / 2  # For interaction terms only
avg_norm_y = (norm_y_L + norm_y_R) / 2  # For interaction terms only
x_yaw_interaction = avg_norm_x * yaw
y_pitch_interaction = avg_norm_y * pitch

# Final 9 features: [norm_x_L, norm_y_L, norm_x_R, norm_y_R, yaw, pitch, roll, x_yaw_interaction, y_pitch_interaction]
```

## 🔍 Technical Details

### Multi-Output RandomForest Architecture
- **Single Model**: Predicts both coordinates simultaneously
- **Shared Learning**: Common feature representations for X/Y
- **Euclidean Optimization**: Direct minimization of spatial error
- **Efficient**: One model vs. two separate models

### Custom Scoring Function
```python
def euclidean_distance_scorer(y_true, y_pred):
    euclidean_distances = np.sqrt(np.sum((y_true - y_pred) ** 2, axis=1))
    return -np.mean(euclidean_distances)  # Negative for maximization
```

### Hyperparameter Optimization
- **GridSearchCV** with custom Euclidean scorer
- **Parameters**: n_estimators, max_depth, learning_rate, subsample, colsample_bytree
- **Cross-Validation**: 5-fold with Euclidean distance evaluation

## 🎮 Controls (Real-Time Prediction)

- **ESC**: Exit prediction
- **SPACE**: Toggle movement analysis
- **'s'**: Start/stop analysis session
- **'r'**: Reset smoothing

## ⚙️ Configuration

### Training Parameters
```python
# In model_training.py
param_grid = {
    'n_estimators': [300, 350, 400],
    'max_depth': [8, 10, 12],
    'min_samples_split': [2, 3, 4],
    'min_samples_leaf': [1, 2, 3]
}
```

### Smoothing Parameters
```python
# In prediction.py
predictor.run_real_time_prediction(
    enable_smoothing=True,
    smoothing_alpha=0.3  # 0-1, lower = more smoothing
)
```

## 🔬 Research Notes

### Why Multi-Output RandomForest?
1. **Joint Optimization**: X and Y coordinates learned together
2. **Shared Features**: Common representations reduce overfitting
3. **Euclidean Optimization**: Direct minimization of spatial error
4. **Efficiency**: Single model reduces computational overhead
5. **Robustness**: Ensemble method provides stable predictions

### Feature Engineering Rationale
1. **Individual Eye Positions**: Raw left/right eye coordinates capture fine-grained eye movements
2. **Head Pose Integration**: Captures gaze direction changes with pitch baseline adjustment
3. **Interaction Terms**: Captures complex relationships between eye position and head orientation

## 📈 Performance Optimization

### Training Optimizations
- GridSearchCV with custom scorer
- Pitch baseline calibration for improved accuracy
- Feature scaling for numerical stability
- Cross-validation for reliable evaluation

### Prediction Optimizations
- Streamlined feature engineering
- Efficient NumPy operations
- Optional exponential smoothing
- Minimal memory allocation

## 🚨 Troubleshooting

### Common Issues
1. **Camera Access**: Ensure no other applications are using the camera
2. **Model Loading**: Check that model files exist in `models/` directory
3. **Calibration Quality**: Ensure good lighting and clear face visibility
4. **Performance**: Close unnecessary applications for better real-time performance

### Requirements
- Python 3.7+
- Webcam with good resolution (720p+)
- Good lighting conditions
- Clear face visibility during calibration

## 📝 License

This project is for educational and research purposes.

## 🤝 Contributing

This is a streamlined implementation focused on multi-output RandomForest optimization. The codebase has been simplified for efficiency and maintainability.

---

**Built with**: RandomForest (scikit-learn), MediaPipe, OpenCV, NumPy, pandas
