# Setup Guide - Eye Movement Tracking System

Complete installation and configuration guide for the multilingual gaze tracking system.

## System Requirements

**Minimum Requirements:**
- Python 3.8 or higher
- 8GB RAM (16GB recommended)
- USB webcam or built-in camera (720p+ recommended)
- 2GB free storage space

**Supported Platforms:**
- Windows 10/11
- macOS 10.15+  
- Linux Ubuntu 20.04+

## Quick Installation

### Automated Setup (Recommended)

Navigate to the system directory:
cd path/to/gaze_tracking_system/system

**Windows:**
setup.bat

**All Platforms**
python setup.py 


The automated setup will install all dependencies, create necessary directories, and verify system compatibility.

### Verification

Test installation success:

python -c "import cv2, mediapipe, numpy, pandas, sklearn, bidi.algorithm, arabic_reshaper, tkinter; print('Installation successful')"


## Manual Installation

If automated setup fails, install dependencies manually:

*Upgrade pip*
python -m pip install --upgrade pip

*Install core dependencies*
pip install opencv-python>=4.8.0 mediapipe>=0.10.0
pip install numpy>=1.24.0 pandas>=2.0.0 scikit-learn>=1.3.0
pip install scipy>=1.10.0 joblib>=1.3.0

*Install GUI and text processing*
pip install pillow>=10.0.0 python-bidi>=0.4.2 arabic-reshaper>=3.0.0

*Install from requirements file*
pip install -r requirements.txt


## Dependency Overview

**Core Computer Vision:**
- opencv-python: Camera access and image processing
- mediapipe: Facial landmark detection and tracking

**Machine Learning:**
- scikit-learn: RandomForest model training and prediction
- numpy: Numerical computing and array operations
- pandas: Data manipulation and CSV handling
- scipy: Scientific computing and statistical functions

**Text Processing:**
- pillow: Image rendering and font handling
- python-bidi: Bidirectional text algorithm for RTL support
- arabic-reshaper: Hebrew text character reshaping

**GUI Framework:**
- tkinter: User interface (typically built-in with Python)

## Running the System

### GUI Mode (Recommended)
cd system/
python gui.py


## Directory Structure

After setup, the following structure is created:
gaze_tracking_system/
├── system/ # Core system files
├── calibration_data/ # System-level calibration storage
├── eye_tracking_data/ # Session data directory
├── models/ # System-level model storage
└── user_sessions/ # User-specific session directories
└── username_timestamp/ # Individual session folders
├── calibration/ # Session calibration data
├── models/ # Session-specific models
└── analysis/ # Session analysis output


## Troubleshooting

### Common Issues

**Python Not Found**
- Windows: Reinstall Python with "Add to PATH" option
- macOS/Linux: Add Python to shell profile PATH

**Camera Access Denied**
- Windows: Check privacy settings for camera access
- macOS: Allow camera access in System Preferences > Security & Privacy
- Linux: Add user to video group: `sudo usermod -a -G video $USER`

**Package Installation Failures**
- Run as Administrator/sudo if permission errors occur
- Upgrade pip: `python -m pip install --upgrade pip`
- Use `--no-cache-dir` flag for problematic packages

**MediaPipe Installation Issues**
- Windows: Install Visual C++ Redistributable
- macOS: Install Xcode Command Line Tools: `xcode-select --install`
- Linux: Install system dependencies: `sudo apt-get install python3-dev libgl1-mesa-glx`

**Hebrew Text Not Displaying**
- Verify python-bidi and arabic-reshaper installation
- Check system has Hebrew fonts available
- Test Hebrew processing independently

**GUI Framework Missing**
- Windows: Reinstall Python with tkinter included
- Ubuntu: `sudo apt-get install python3-tk`
- CentOS/RHEL: `sudo yum install python3-tkinter`

### Advanced Troubleshooting

**Clean Installation:**

Remove existing packages
pip freeze | xargs pip uninstall -y

Reinstall from requirements
pip install -r requirements.txt

Create isolated environment
python -m venv gaze_env

Activate environment
Windows: gaze_env\Scripts\activate
macOS/Linux: source gaze_env/bin/activate
Install packages
pip install -r requirements.txt


**System Capability Test:**
Test camera
import cv2
cap = cv2.VideoCapture(0)
print(f"Camera available: {cap.isOpened()}")
cap.release()

Test MediaPipe
import mediapipe as mp
mp_face_mesh = mp.solutions.face_mesh
print("MediaPipe loaded successfully")

Test Hebrew processing
from bidi.algorithm import get_display
import arabic_reshaper
text = arabic_reshaper.reshape("שלום עולם")
display_text = get_display(text)
print("Hebrew processing working")


## Platform-Specific Notes

**Windows:**
- Use PowerShell or Command Prompt as Administrator
- Ensure camera drivers are installed and updated
- Windows Defender may prompt for camera access permission

**macOS:**
- May require Xcode Command Line Tools for compilation
- Grant camera permissions when system prompts
- Some packages may require Homebrew for dependencies


## Performance Optimization

**System Performance:**
- Close unnecessary applications during tracking sessions
- Ensure adequate lighting for camera detection
- Use wired camera connection when possible for stability

**Memory Management:**
- Monitor RAM usage during long sessions
- Restart application between extended analysis sessions
- Clear session data periodically if storage is limited

## Maintenance

**Dependency Updates:**
Update all packages
pip install --upgrade -r requirements.txt

Update specific packages
pip install --upgrade opencv-python mediapipe

**System Validation:**
Periodically verify system functionality with the verification test above, especially after system updates or hardware changes.

## Support

For additional assistance:
- Ensure error messages are captured for troubleshooting
- Note operating system version and Python version
- Test camera functionality independently of the application
- Verify all dependencies are correctly installed before reporting issues
