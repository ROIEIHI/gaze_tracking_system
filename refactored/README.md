# Streamlined Gaze Tracking System

## Multi-Output XGBoost Implementation with Euclidean Distance Optimization

A highly efficient gaze tracking system using advanced machine learning techniques for accurate real-time gaze prediction.

## Features

- **Multi-Output XGBoost Model**: Single model predicting both X and Y coordinates simultaneously
- **Euclidean Distance Optimization**: Custom scorer for minimizing spatial prediction error
- **Advanced Feature Engineering**: 7 engineered features including interaction terms
- **Data Augmentation**: Gaussian noise injection for improved generalization
- **Real-Time Prediction**: Optimized for low-latency live gaze tracking
- **21-Point Calibration**: Comprehensive calibration system using MediaPipe
- **Streamlined Workflow**: Complete pipeline from calibration to prediction

## Quick Start

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
pip install xgboost scikit-learn pandas numpy opencv-python mediapipe joblib
```

### 3. Run the System
```bash
python main.py
```

### 4. Choose Your Workflow
- **Run Calibration**: Collect training data (21-point calibration)
- **Train Model**: Build optimized XGBoost model
- **Real-Time Prediction**: Use trained model for live gaze tracking
- **Complete Workflow**: Run all steps in sequence

## Setup Scripts (Local Use Only)

For new installations, the following helper scripts are available:
- `setup_environment.py` - Comprehensive setup for new computers
- `requirements.txt` - Package dependencies list
- `validate_setup.py` - System validation tool

*Note: These setup scripts are for local use and are not included in the repository.*

## Project Structure

```
refactored/
├── main.py                    # Main GUI application
├── calibration.py             # 21-point calibration system
├── model_training.py          # Multi-output XGBoost training
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
- Head pose estimation (yaw, pitch, roll)
- Iris position normalization
- Outlier detection and data cleaning

### 2. Model Training (`model_training.py`)
- **Multi-Output XGBoost**: Single model for X/Y prediction
- **Custom Euclidean Scorer**: Minimizes spatial distance error
- **Feature Engineering**: Average eye positions + interaction terms
- **GridSearchCV Optimization**: Hyperparameter tuning
- **Data Augmentation**: 5-pixel Gaussian noise injection
- **Cross-Validation**: 5-fold validation with custom scorer

### 3. Real-Time Prediction (`prediction.py`)
- **Streamlined Pipeline**: Feature extraction → prediction → smoothing
- **Exponential Smoothing**: Reduces prediction jitter
- **Eye Movement Analysis**: Optional movement tracking
- **Performance Optimized**: Minimal latency for real-time use

## 📊 Model Performance

### Training Configuration
- **Features**: 7 engineered features (avg_norm_x, avg_norm_y, yaw, pitch, roll, x_yaw_interaction, y_pitch_interaction)
- **Optimization**: Custom Euclidean distance scorer
- **Data Augmentation**: 5-pixel noise level
- **Cross-Validation**: 5-fold with custom scorer

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
model_path = trainer.train_from_csv(csv_file, noise_level=5.0)

# Prediction
from prediction import GazePredictor
predictor = GazePredictor(model_path)
predictor.run_real_time_prediction(enable_smoothing=True)
```

### Feature Engineering
The system automatically engineers features from raw MediaPipe outputs:

```python
# Raw features: [norm_x_L, norm_y_L, norm_x_R, norm_y_R, yaw, pitch, roll]
# Engineered features:
avg_norm_x = (norm_x_L + norm_x_R) / 2
avg_norm_y = (norm_y_L + norm_y_R) / 2
x_yaw_interaction = avg_norm_x * yaw
y_pitch_interaction = avg_norm_y * pitch
# Final: [avg_norm_x, avg_norm_y, yaw, pitch, roll, x_yaw_interaction, y_pitch_interaction]
```

## 🔍 Technical Details

### Multi-Output XGBoost Architecture
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
    'n_estimators': [100, 200, 300],
    'max_depth': [4, 6, 8],
    'learning_rate': [0.05, 0.1, 0.15],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
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

### Why Multi-Output XGBoost?
1. **Joint Optimization**: X and Y coordinates learned together
2. **Shared Features**: Common representations reduce overfitting
3. **Euclidean Optimization**: Direct minimization of spatial error
4. **Efficiency**: Single model reduces computational overhead

### Feature Engineering Rationale
1. **Average Eye Positions**: Reduces noise from individual eye variations
2. **Head Pose Integration**: Captures gaze direction changes
3. **Interaction Terms**: Captures complex relationships between eye position and head orientation

## 📈 Performance Optimization

### Training Optimizations
- GridSearchCV with custom scorer
- Data augmentation for robustness
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

This is a streamlined implementation focused on multi-output XGBoost optimization. The codebase has been simplified for efficiency and maintainability.

---

**Built with**: XGBoost, scikit-learn, MediaPipe, OpenCV, NumPy, pandas
