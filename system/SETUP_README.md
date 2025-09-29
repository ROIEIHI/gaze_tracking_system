# Gaze Tracking System - Setup Guide

## Hebrew RTL-Aware Eye Movement Tracking System

This guide will help you set up the gaze tracking system on any computer after cloning the repository.

---

## 🚀 Quick Setup (Recommended)

### For Windows Users
1. Open Command Prompt or PowerShell as Administrator
2. Navigate to the `system/` directory:
   ```batch
   cd path\to\gaze_tracking_system\system
   ```
3. Run the automated setup:
   ```batch
   setup.bat
   ```

### For All Platforms (Alternative)
1. Open terminal/command prompt
2. Navigate to the `system/` directory:
   ```bash
   cd path/to/gaze_tracking_system/system
   ```
3. Run the Python setup script:
   ```bash
   python setup.py
   ```

---

## 📋 System Requirements

### Minimum Requirements
- **Python**: 3.8 or higher
- **Operating System**: Windows 10+, macOS 10.15+, or Linux Ubuntu 20.04+
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 2GB free space
- **Camera**: USB webcam or built-in camera (720p or higher recommended)

### Python Version Check
```bash
python --version
```
Should return Python 3.8.x or higher.

---

## 🛠 Manual Installation

If the automated setup doesn't work, follow these manual steps:

### Step 1: Install Python Dependencies
```bash
# Upgrade pip first
python -m pip install --upgrade pip

# Install all requirements
pip install -r requirements.txt
```

### Step 2: Verify Critical Packages
Test that key packages are working:

```python
# Test in Python console
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import sklearn
from PIL import Image
import bidi.algorithm
import arabic_reshaper
import tkinter
```

### Step 3: Create Project Directories
The system needs these directories (created automatically by setup scripts):
- `calibration_data/`
- `eye_tracking_data/`
- `models/`
- `movement_data/`
- `user_data/`

---

## 📦 Package Details

### Core Dependencies
| Package | Purpose | Minimum Version |
|---------|---------|----------------|
| opencv-python | Computer vision and camera access | 4.8.0 |
| mediapipe | Face and landmark detection | 0.10.0 |
| numpy | Numerical computing | 1.24.0 |
| pandas | Data manipulation | 2.0.0 |
| scikit-learn | Machine learning models | 1.3.0 |
| scipy | Scientific computing | 1.10.0 |
| joblib | Model serialization | 1.3.0 |

### GUI and Text Processing
| Package | Purpose | Minimum Version |
|---------|---------|----------------|
| tkinter | GUI framework (usually built-in) | Built-in |
| Pillow (PIL) | Image processing | 10.0.0 |
| python-bidi | Hebrew/Arabic text processing | 0.4.2 |
| arabic-reshaper | RTL text reshaping | 3.0.0 |

### Optional Packages
| Package | Purpose | Minimum Version |
|---------|---------|----------------|
| matplotlib | Data visualization | 3.7.0 |
| seaborn | Statistical plotting | 0.12.0 |
| pytest | Unit testing | 7.0.0 |

---

## 🏃‍♂️ Running the System

### GUI Mode (Recommended)
```bash
cd system/
python gui.py
```

### Command Line Mode
```bash
cd system/
python main.py
```

---

## 🔧 Troubleshooting

### Common Issues and Solutions

#### 1. "Python is not recognized"
**Problem**: Python not in system PATH
**Solution**:
- Windows: Reinstall Python with "Add Python to PATH" checked
- macOS/Linux: Add Python to PATH in shell profile

#### 2. "No module named 'cv2'"
**Problem**: OpenCV not installed properly
**Solution**:
```bash
pip uninstall opencv-python
pip install opencv-python
```

#### 3. "ModuleNotFoundError: No module named 'mediapipe'"
**Problem**: MediaPipe installation failed
**Solution**:
```bash
pip install --upgrade pip
pip install mediapipe --no-cache-dir
```

#### 4. Camera Access Issues
**Problem**: Camera not detected or permission denied
**Solution**:
- Windows: Check camera privacy settings
- macOS: Allow camera access in System Preferences
- Linux: Add user to video group: `sudo usermod -a -G video $USER`

#### 5. Hebrew Text Not Displaying Properly
**Problem**: RTL text processing packages missing
**Solution**:
```bash
pip install python-bidi arabic-reshaper
```

#### 6. "tkinter module not found"
**Problem**: GUI framework not available
**Solution**:
- Windows: Reinstall Python with tkinter
- Ubuntu: `sudo apt-get install python3-tk`
- macOS: Usually included with Python

### Advanced Troubleshooting

#### Clean Installation
If you encounter persistent issues:
```bash
# Remove all packages
pip freeze | grep -v "^-e" | xargs pip uninstall -y

# Reinstall from requirements
pip install -r requirements.txt
```

#### Virtual Environment Setup
For isolated installation:
```bash
# Create virtual environment
python -m venv gaze_env

# Activate (Windows)
gaze_env\Scripts\activate

# Activate (macOS/Linux)  
source gaze_env/bin/activate

# Install packages
pip install -r requirements.txt
```

#### Check System Capabilities
Test your system:
```python
# Test camera access
import cv2
cap = cv2.VideoCapture(0)
print(f"Camera available: {cap.isOpened()}")
cap.release()

# Test MediaPipe
import mediapipe as mp
mp_face_mesh = mp.solutions.face_mesh
print("MediaPipe face mesh loaded successfully")

# Test Hebrew processing
from bidi.algorithm import get_display
import arabic_reshaper
text = "שלום עולם"
reshaped = arabic_reshaper.reshape(text)
display_text = get_display(reshaped) 
print("Hebrew text processing working")
```

---

## 🌐 Platform-Specific Notes

### Windows
- Use PowerShell or Command Prompt as Administrator
- Ensure camera drivers are installed
- Windows Defender may flag camera access

### macOS
- Install Xcode Command Line Tools: `xcode-select --install`
- Grant camera permissions when prompted
- May need to install Homebrew for some dependencies

### Linux (Ubuntu/Debian)
Install system dependencies:
```bash
sudo apt-get update
sudo apt-get install python3-dev python3-pip python3-tk
sudo apt-get install libgl1-mesa-glx libglib2.0-0
```

### Linux (CentOS/RHEL)
```bash
sudo yum install python3-devel python3-pip python3-tkinter
sudo yum install mesa-libGL glib2
```

---

## 📁 Directory Structure

After successful setup, your directory should look like:
```
gaze_tracking_system/
├── system/
│   ├── gui.py              # Main GUI application
│   ├── main.py             # Command line interface
│   ├── requirements.txt    # Package dependencies
│   ├── setup.py           # Cross-platform setup
│   ├── setup.bat          # Windows setup script
│   ├── SETUP_README.md    # This file
│   └── [other system files]
├── calibration_data/       # Calibration files
├── eye_tracking_data/      # Session data
├── models/                 # Trained ML models
├── movement_data/          # Movement analysis
└── user_data/             # User session directories
```

---

## 🆘 Getting Help

If you continue to have issues:

1. **Check System Requirements**: Ensure your system meets minimum requirements
2. **Update Python**: Make sure you're using Python 3.8+
3. **Try Virtual Environment**: Create a clean environment
4. **Check Error Messages**: Read error messages carefully
5. **System Logs**: Check system logs for camera/permission issues

### Contact Information
- Create an issue in the repository
- Include error messages and system information
- Specify your operating system and Python version

---

## ✅ Verification Checklist

After setup, verify these work:
- [ ] Python 3.8+ installed and in PATH
- [ ] All packages from requirements.txt installed
- [ ] Camera access works
- [ ] GUI launches without errors: `python gui.py`
- [ ] Hebrew text renders properly
- [ ] Directory structure created

---

## 🔄 Updates and Maintenance

### Updating Dependencies
```bash
# Update all packages to latest versions
pip install --upgrade -r requirements.txt

# Update specific package
pip install --upgrade opencv-python
```

### System Compatibility
The system has been tested on:
- Windows 10/11
- macOS Big Sur and later
- Ubuntu 20.04 LTS and later

---

## 📝 License and Credits

This gaze tracking system includes Hebrew RTL support and advanced eye movement analysis capabilities. See the main README for full feature documentation.