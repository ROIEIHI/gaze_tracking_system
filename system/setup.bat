@echo off
REM ====================================================================
REM Gaze Tracking System - Automated Setup Script for Windows
REM Hebrew RTL-Aware Eye Movement Tracking System
REM ====================================================================

echo.
echo ========================================
echo   Gaze Tracking System Setup
echo   Hebrew RTL-Aware Eye Tracking
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH!
    echo Please install Python 3.8+ from https://python.org
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

echo [1/6] Python detected successfully
python --version

REM Check Python version (requires 3.8+)
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo [2/6] Checking Python version: %PYTHON_VERSION%

REM Upgrade pip to latest version
echo [3/6] Upgrading pip to latest version...
python -m pip install --upgrade pip

REM Install all required packages
echo [4/6] Installing required packages...
echo Installing core dependencies (this may take a few minutes)...

REM Install packages from requirements.txt
python -m pip install -r requirements.txt

REM Verify critical installations
echo [5/6] Verifying installations...

python -c "import cv2; print(f'✓ OpenCV {cv2.__version__} installed successfully')" || (
    echo ✗ OpenCV installation failed
    goto :error
)

python -c "import mediapipe as mp; print(f'✓ MediaPipe {mp.__version__} installed successfully')" || (
    echo ✗ MediaPipe installation failed
    goto :error
)

python -c "import numpy as np; print(f'✓ NumPy {np.__version__} installed successfully')" || (
    echo ✗ NumPy installation failed
    goto :error
)

python -c "import pandas as pd; print(f'✓ Pandas {pd.__version__} installed successfully')" || (
    echo ✗ Pandas installation failed
    goto :error
)

python -c "import sklearn; print(f'✓ Scikit-learn {sklearn.__version__} installed successfully')" || (
    echo ✗ Scikit-learn installation failed
    goto :error
)

python -c "from PIL import Image; print('✓ Pillow (PIL) installed successfully')" || (
    echo ✗ Pillow installation failed
    goto :error
)

python -c "import bidi.algorithm; print('✓ python-bidi installed successfully')" || (
    echo ✗ python-bidi installation failed
    goto :error
)

python -c "import arabic_reshaper; print('✓ arabic-reshaper installed successfully')" || (
    echo ✗ arabic-reshaper installation failed
    goto :error
)

python -c "import tkinter; print('✓ Tkinter (GUI) available')" || (
    echo ✗ Tkinter not available - may need to reinstall Python with tkinter
    goto :error
)

REM Create necessary directories
echo [6/6] Creating project directories...
if not exist "..\calibration_data" mkdir "..\calibration_data"
if not exist "..\eye_tracking_data" mkdir "..\eye_tracking_data"
if not exist "..\models" mkdir "..\models"
if not exist "..\movement_data" mkdir "..\movement_data"
if not exist "..\user_data" mkdir "..\user_data"

echo.
echo ========================================
echo   Setup Complete!
echo ========================================
echo.
echo All dependencies have been installed successfully.
echo You can now run the system using:
echo.
echo   python gui.py
echo.
echo Or for command-line interface:
echo   python main.py
echo.
echo Project directories created:
echo   - calibration_data/  (for calibration files)
echo   - eye_tracking_data/ (for session data)
echo   - models/           (for trained models)
echo   - movement_data/    (for movement analysis)
echo   - user_data/        (for user sessions)
echo.
echo For troubleshooting, see SETUP_README.md
echo.
pause
exit /b 0

:error
echo.
echo ========================================
echo   Setup Failed!
echo ========================================
echo.
echo An error occurred during installation.
echo Please check the error messages above.
echo.
echo Common solutions:
echo 1. Make sure Python 3.8+ is installed
echo 2. Run this script as Administrator
echo 3. Check your internet connection
echo 4. Try: python -m pip install --upgrade pip
echo.
echo For detailed troubleshooting, see SETUP_README.md
echo.
pause
exit /b 1