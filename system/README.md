# 🎯 Hebrew RTL-Aware Gaze Tracking Reading Analysis System

A **professional-grade multilingual eye tracking system** with advanced support for Hebrew right-to-left (RTL) text reading analysis. This system provides comprehensive gaze tracking capabilities with RTL-aware prediction algorithms, adaptive Hebrew text rendering, and detailed movement metrics optimized for both Hebrew and English reading research.

> **🚀 NEW: Plug & Play Setup** - Complete automated installation with one-click deployment scripts. Clone the repository and run `setup.bat` (Windows) or `python setup.py` (all platforms) for instant setup on any computer!

## ✨ Key Features

### 🔄 **RTL-Aware Eye Tracking (NEW)**
- **Hebrew text detection** with automatic RTL mode activation
- **RTL-optimized Kalman filter** with leftward reading bias (-1.2 vs +1.0 for LTR)
- **Hebrew-specific eye movement parameters** (higher process noise, negative saccade direction)
- **Automatic language detection** based on Hebrew keyword analysis
- **Seamless bilingual support** for Hebrew and English text analysis

### 🎯 **Advanced Calibration System**
- **21-point strategic calibration** targeting corners, edges, and center regions
- **Additional grid calibration** with customizable density (6x6 default)
- **Adaptive margins** and intelligent duplicate detection
- **Real-time face detection** with MediaPipe integration
- **Pitch baseline calibration** for improved accuracy

### 📖 **Multilingual Text Reading Analysis**
- **Hebrew RTL text rendering** with PIL and bidi library support
- **Adaptive text rendering** that scales to any screen size for both Hebrew and English
- **RTL word-level fixation tracking** with Hebrew-aligned positioning
- **Reading pattern detection** optimized for RTL patterns (fixations, saccades, regressions)
- **Multi-page text support** with RTL navigation controls
- **Hebrew font integration** with automatic system font detection

### 🧠 **RTL-Aware Eye Movement Analysis**
- **Dual-mode Kalman filtering** (LTR/RTL) for language-optimized gaze tracking
- **RTL movement classification** (Hebrew-specific fixations, leftward saccades, line returns)
- **Hebrew reading-specific metrics** (RTL WPM, Hebrew regression patterns, RTL line changes)
- **Real-time RTL velocity and direction analysis**
- **EyeMovementAnalyzer** with Hebrew reading mode and RTL parameters

### 📊 **Data Export & Analysis**
- **CSV export** matching standard research formats
- **Comprehensive fixation data** with timestamps and durations
- **Word-level analysis** with proximity detection
- **Session-based tracking** with detailed metrics
- **Compatible with eye tracking research standards**

## 🎁 **Plug & Play Deployment**

### ⚡ **One-Click Setup**
- **Automated installation scripts** for Windows and all platforms
- **Dependency auto-detection** with intelligent error handling
- **Project structure creation** with proper directory organization
- **System validation** with comprehensive compatibility checks

### 🚀 **Quick Deployment Features**
- **Clone and run** - minimal manual configuration required
- **Cross-platform support** - Windows, macOS, and Linux ready
- **Session management** - automatic user directory creation
- **Professional GUI** - no command line knowledge needed
- **Comprehensive documentation** - detailed setup guides included

### 📁 **Ready-to-Use Structure**
```bash
git clone [repository]
cd gaze_tracking_system/system
setup.bat          # Windows: Double-click to install
# or
python setup.py    # All platforms: Automated setup
python gui.py      # Start the professional interface
```

## 🚀 Quick Setup (Plug & Play)

### 🎯 **Automated Installation (Recommended)**

**For Windows:**
```batch
# Navigate to system directory
cd system/

# Run automated setup (installs everything)
setup.bat
```

**For All Platforms:**
```bash
# Navigate to system directory
cd system/

# Run cross-platform setup
python setup.py
```

The automated setup will:
- ✅ Verify Python 3.8+ installation
- ✅ Install all required packages automatically
- ✅ Verify Hebrew text processing libraries
- ✅ Create necessary project directories
- ✅ Test camera and system compatibility
- ✅ Provide troubleshooting guidance if needed

### 🎮 **Quick Launch**

After setup, start the system easily:

**Windows Quick Start:**
```batch
# Double-click or run:
start.bat
```

**Cross-Platform:**
```bash
python start.py
# or
python gui.py
```

### 📦 **What Gets Installed**

The system automatically installs:
- **Core Computer Vision**: OpenCV, MediaPipe
- **Machine Learning**: scikit-learn, NumPy, pandas, scipy, joblib  
- **Hebrew RTL Support**: python-bidi, arabic-reshaper, Pillow
- **GUI Framework**: tkinter (usually built-in)
- **Optional**: matplotlib, seaborn for data visualization

### 🆘 **Need Help?**
- See `SETUP_README.md` for detailed troubleshooting
- All setup files are in the `system/` directory
- System tested on Windows 10/11, macOS, and Linux

## 🎯 Using the System

### 1. **Professional GUI Interface**
```bash
python gui.py
```
- Modern tkinter-based interface
- Complete workflow management  
- Session-based directory organization
- Real-time progress tracking
- Hebrew RTL-aware text analysis

### 2. **Command Line Interface** 
```bash
python main.py
```
- Interactive menu system
- Step-by-step workflow
- Manual control over each component

### 3. **Calibration Process**
- **Position yourself** 60-80cm from the screen
- **Follow the red circles** that appear on screen
- **Keep your head stable** during each calibration point
- **21 strategic points + grid points** for comprehensive coverage

### 4. **Hebrew RTL Reading Analysis**
- **Automatic Hebrew detection** and RTL mode activation
- **ESC**: Exit and export Hebrew reading data
- **A/D**: Navigate between Hebrew text pages (RTL-aware)
- **E**: Export current Hebrew reading session data
- **Fullscreen Hebrew text** with RTL-optimized gaze overlay
- **Hebrew font rendering** with proper character support

## 📁 System Architecture

```
system/
├── main.py                    # Main entry point with Hebrew RTL support
├── calibration.py            # Advanced calibration with 21+ points
├── model_training.py         # Multi-output RandomForest training
├── prediction.py             # Hebrew RTL text reading analysis system
├── eye_movement_analyzer.py  # RTL-aware movement analysis and Kalman filtering
├── config.py                 # System configuration with Hebrew RTL settings
└── README.md                 # This documentation
```

### RTL-Aware Components (NEW)
- **Hebrew Text Detection**: Automatic language identification using Hebrew keywords
- **RTL Kalman Filter**: Specialized parameters for Hebrew reading patterns
- **Hebrew Font Rendering**: PIL-based rendering with bidi/arabic-reshaper support
- **RTL Word Positioning**: Hebrew text layout with proper right-to-left alignment

## 🛠 Setup Files Reference

The system includes comprehensive setup automation:

### **Setup Scripts**
- `setup.bat` - Windows automated installation
- `setup.py` - Cross-platform Python setup script
- `start.bat` - Windows quick launcher
- `start.py` - Cross-platform launcher
- `requirements.txt` - Python package dependencies
- `SETUP_README.md` - Detailed setup documentation

### **Manual Installation (If Needed)**
```bash
pip install -r requirements.txt
```

Or install packages individually:
```bash
pip install opencv-python>=4.8.0 mediapipe>=0.10.0
pip install numpy>=1.24.0 pandas>=2.0.0 scikit-learn>=1.3.0
pip install pillow>=10.0.0 python-bidi>=0.4.2 arabic-reshaper>=3.0.0
```

### **Directory Structure Created**
After setup, the system creates:
```
gaze_tracking_system/
├── system/              # Core system files
├── calibration_data/    # Calibration files  
├── eye_tracking_data/   # Session data with user directories
├── models/             # Trained ML models
├── movement_data/      # Movement analysis files
└── user_data/          # User session directories (USERNAME_TIMESTAMP format)
```

## 🇮🇱 Hebrew RTL Features (NEW)

### Hebrew Text Detection
The system automatically detects Hebrew text and switches to RTL mode:
```python
# Automatic Hebrew detection based on keywords
hebrew_keywords = ["של", "את", "על", "אל", "עם", "בין", "אם", "מה", "זה", "הוא"]
rtl_threshold = 2  # Minimum Hebrew words to activate RTL mode
```

### RTL-Aware Kalman Filter
Specialized parameters for Hebrew reading patterns:
```python
# LTR Mode (English)
reading_direction_bias = 1.0    # Rightward movement bias
process_noise = 0.1             # Standard process noise
saccade_direction = 1           # Positive (rightward)

# RTL Mode (Hebrew) 
reading_direction_bias = -1.2   # Leftward movement bias
process_noise = 0.15            # Higher noise for RTL variability  
saccade_direction = -1          # Negative (leftward)
```

### Hebrew Text Rendering
- **PIL Integration**: Proper Hebrew font rendering with system font detection
- **Bidi Processing**: Right-to-left text layout with python-bidi library
- **Character Shaping**: Arabic-reshaper for proper Hebrew character connection
- **RTL Alignment**: Text positioned from right edge with proper word spacing

### Hebrew Reading Analysis
- **RTL Word Detection**: Word positions calculated for right-to-left reading flow
- **Hebrew Fixation Patterns**: Optimized for Hebrew reading behavior
- **RTL Saccade Analysis**: Leftward eye movements and line return detection
- **Hebrew CSV Export**: Reading data formatted for Hebrew text analysis research

### 3. Run the System
```bash
python main.py
```

### 5. **Workflow Options**
- **Complete Workflow** (GUI): Full session with calibration → training → analysis
- **Step-by-Step** (CLI): Individual components with manual control
- **Hebrew RTL Analysis**: Specialized Hebrew text reading analysis
- **Session Management**: User-specific directories with timestamp organization

## 🔧 Installation Troubleshooting

### **Common Setup Issues**

| Issue | Solution |
|-------|----------|
| Python not found | Install Python 3.8+ and add to PATH |
| Camera access denied | Check privacy settings, close other camera apps |
| Package installation fails | Run as Administrator, upgrade pip: `python -m pip install --upgrade pip` |
| MediaPipe errors | Install Visual C++ Redistributable (Windows) |
| Hebrew text not displaying | Verify python-bidi and arabic-reshaper installation |
| tkinter missing | Reinstall Python with tkinter or install python3-tk (Linux) |

### **System Requirements**
- **Python**: 3.8 or higher
- **RAM**: 8GB minimum, 16GB recommended
- **Camera**: USB webcam or built-in (720p+ recommended)
- **OS**: Windows 10+, macOS 10.15+, or Linux Ubuntu 20.04+

### **Verification Test**
After installation, run this quick test:
```python
python -c "
import cv2, mediapipe, numpy, pandas, sklearn
from bidi.algorithm import get_display
import arabic_reshaper, tkinter
print('✅ All dependencies installed successfully!')
"
```

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

### 3. Hebrew RTL Eye Movement Analysis (`eye_movement_analyzer.py`)
- **RTL-Aware KalmanFilter**: Specialized class with reading_direction parameter
- **Automatic RTL Detection**: EyeMovementAnalyzer switches modes based on text language
- **Hebrew Reading Parameters**: Leftward bias, higher process noise, negative saccade direction
- **Bilingual Support**: Seamless switching between LTR and RTL prediction modes

### 4. Hebrew RTL Text Prediction (`prediction.py`)
- **Hebrew Text Detection**: Automatic language identification and RTL mode activation
- **RTL Text Rendering**: PIL-based Hebrew font rendering with proper character shaping
- **RTL Word Positioning**: Right-to-left word layout matching visual Hebrew text flow
- **RTL-Aware Eye Tracking**: Integration with Hebrew-optimized Kalman filter
- **Bilingual Analysis**: Seamless support for both Hebrew and English text analysis
- **Hebrew CSV Export**: Research-compatible data export for Hebrew reading studies

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
# Standard Calibration and Training
from calibration import EyeTrackerCalibrator
from model_training import GazeModelTrainer

calibrator = EyeTrackerCalibrator()
csv_file = calibrator.run_calibration()

trainer = GazeModelTrainer()
model_path = trainer.train_from_csv(csv_file)

# Hebrew RTL-Aware Prediction (NEW)
from prediction import TextReadingGazePredictor

predictor = TextReadingGazePredictor(model_path)
# Automatically detects Hebrew text and activates RTL mode
predictor.run_text_analysis()  # Hebrew text rendering + RTL eye tracking
```

### Hebrew RTL Configuration (NEW)

```python
# Hebrew RTL Configuration in config.py
HEBREW_SUPPORT = True
HEBREW_KEYWORDS = ["של", "את", "על", "אל", "עם", "בין", "אם", "מה", "זה", "הוא"]
RTL_AUTO_DETECT = True
RTL_DETECTION_THRESHOLD = 2

# RTL Kalman Filter Parameters
RTL_READING_BIAS = -1.2        # Leftward movement bias for Hebrew
RTL_PROCESS_NOISE = 0.15       # Higher variability for RTL patterns
RTL_SACCADE_DIRECTION = -1     # Negative for leftward saccades

# Hebrew Text Sample in config.py
READING_TEXT = """
היתרונות של פעילות גופנית מתרחבים הרבה מעבר לכושר גופני בלבד. פעילות גופנית סדירה
משפרת את בריאות הלב וכלי הדם, מחזקת שרירים ועצמות, ועוזרת לשמור על משקל בריא.
"""
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

### Why Multi-Output RandomForest with RTL Support?
1. **Joint Optimization**: X and Y coordinates learned together for both LTR and RTL
2. **Shared Features**: Common representations reduce overfitting across languages
3. **Euclidean Optimization**: Direct minimization of spatial error for multilingual text
4. **Efficiency**: Single model reduces computational overhead for bilingual analysis
5. **Robustness**: Ensemble method provides stable predictions for Hebrew and English
6. **RTL Adaptability**: Same model with RTL-aware post-processing for Hebrew analysis

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

**Built with**: RandomForest (scikit-learn), MediaPipe, OpenCV, NumPy, pandas, PIL (Hebrew rendering), python-bidi (RTL support), arabic-reshaper (Hebrew text processing)
