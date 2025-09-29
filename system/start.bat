@echo off
REM ====================================================================
REM Gaze Tracking System - Quick Launcher
REM Hebrew RTL-Aware Eye Movement Tracking System
REM ====================================================================

echo.
echo ========================================
echo   Starting Gaze Tracking System
echo   Hebrew RTL-Aware Eye Tracking
echo ========================================
echo.

REM Change to system directory
cd /d "%~dp0"

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH!
    echo Please run setup.bat first to install dependencies
    pause
    exit /b 1
)

REM Check if dependencies are installed by testing key imports
echo Checking system dependencies...
python -c "import cv2, mediapipe, numpy, pandas, sklearn, bidi.algorithm, arabic_reshaper" >nul 2>&1
if %errorlevel% neq 0 (
    echo WARNING: Dependencies may not be installed properly
    echo Please run setup.bat to install all required packages
    echo.
    echo Attempting to start anyway...
    echo.
)

REM Start the GUI application
echo Starting GUI application...
echo.
python gui.py

REM If we get here, the application has closed
echo.
echo Application closed.
pause