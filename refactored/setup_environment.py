#!/usr/bin/env python3
"""
Gaze Tracking System - Environment Setup Script
===============================================

This script sets up all necessary libraries and environment for the gaze tracking system.
Run this script ONCE before using the main gaze tracking application.

Requirements:
- Python 3.8 or higher
- Internet connection for package downloads
- Camera access permissions

Usage:
    python setup_environment.py

Author: Gaze Tracking System Team
Date: September 2025
"""

import sys
import subprocess
import os
import platform
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    print("[PYTHON] Checking Python version...")
    
    version = sys.version_info
    print(f"   Current Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major != 3 or version.minor < 8:
        print("[ERROR] Python 3.8 or higher is required")
        print("   Please upgrade Python and try again")
        return False
    
    print("[OK] Python version is compatible")
    return True

def install_package(package_name, import_name=None, version=None):
    """Install a Python package using pip"""
    if import_name is None:
        import_name = package_name
    
    # Check if package is already installed
    try:
        __import__(import_name)
        print(f"[OK] {package_name} is already installed")
        return True
    except ImportError:
        pass
    
    # Install the package
    print(f"[INSTALL] Installing {package_name}...")
    
    install_cmd = [sys.executable, "-m", "pip", "install"]
    
    if version:
        install_cmd.append(f"{package_name}{version}")
    else:
        install_cmd.append(package_name)
    
    try:
        result = subprocess.run(install_cmd, capture_output=True, text=True, check=True)
        print(f"[OK] {package_name} installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to install {package_name}")
        print(f"   Error: {e.stderr}")
        return False

def upgrade_pip():
    """Upgrade pip to latest version"""
    print("[PIP] Upgrading pip to latest version...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], 
                      capture_output=True, text=True, check=True)
        print("[OK] pip upgraded successfully")
        return True
    except subprocess.CalledProcessError as e:
        print("[WARN] Could not upgrade pip")
        print(f"   Error: {e.stderr}")
        return False

def create_directory_structure():
    """Create necessary directories for the system"""
    print("[DIRS] Creating directory structure...")
    
    base_dir = Path.cwd()
    directories = [
        "calibration_data",
        "models",
        "movement_data",
        "utils"
    ]
    
    for directory in directories:
        dir_path = base_dir / directory
        dir_path.mkdir(exist_ok=True)
        print(f"   Created/verified: {directory}/")
    
    print("[OK] Directory structure ready")
    return True

def check_camera_access():
    """Check if camera is accessible"""
    print("[CAMERA] Checking camera access...")
    
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            
            if ret and frame is not None:
                print("[OK] Camera is accessible")
                return True
            else:
                print("[WARN] Camera detected but cannot capture frames")
                return False
        else:
            print("[WARN] Cannot access camera")
            print("   Please ensure:")
            print("   - Camera is connected")
            print("   - Camera permissions are granted")
            print("   - No other application is using the camera")
            return False
            
    except Exception as e:
        print(f"[WARN] Error checking camera: {e}")
        return False

def install_core_packages():
    """Install all required packages for the gaze tracking system"""
    print("[INSTALL] Installing core packages...")
    
    # Define all required packages with their import names and version constraints
    packages = [
        # Computer Vision and MediaPipe
        ("opencv-python", "cv2", ">=4.5.0"),
        ("mediapipe", "mediapipe", ">=0.10.0"),
        
        # Data Science and Machine Learning
        ("numpy", "numpy", ">=1.21.0"),
        ("pandas", "pandas", ">=1.3.0"),
        ("scikit-learn", "sklearn", ">=1.0.0"),
        ("joblib", "joblib", ">=1.1.0"),
        
        # GUI Framework (usually pre-installed with Python)
        ("tk", None, None),  # tkinter - usually comes with Python
        
        # Optional but recommended packages
        ("matplotlib", "matplotlib", ">=3.5.0"),  # For visualization
        ("seaborn", "seaborn", ">=0.11.0"),       # For better plots
    ]
    
    successful_installs = 0
    total_packages = len([p for p in packages if p[1] is not None])
    
    for package_name, import_name, version in packages:
        if import_name is None:
            continue  # Skip packages that don't need installation (like tkinter)
            
        success = install_package(package_name, import_name, version)
        if success:
            successful_installs += 1
    
    print(f"\n[SUMMARY] Installation Summary: {successful_installs}/{total_packages} packages installed successfully")
    
    if successful_installs == total_packages:
        print("[OK] All core packages installed successfully")
        return True
    else:
        print("[WARN] Some packages failed to install - system may still work with limited functionality")
        return False

def verify_installation():
    """Verify that all critical components are working"""
    print("[VERIFY] Verifying installation...")
    
    # Test critical imports
    critical_imports = [
        ("cv2", "OpenCV"),
        ("mediapipe", "MediaPipe"),
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("sklearn", "Scikit-learn"),
        ("joblib", "Joblib"),
        ("tkinter", "Tkinter")
    ]
    
    failed_imports = []
    
    for module_name, display_name in critical_imports:
        try:
            __import__(module_name)
            print(f"[OK] {display_name} import successful")
        except ImportError as e:
            print(f"[ERROR] {display_name} import failed: {e}")
            failed_imports.append(display_name)
    
    if not failed_imports:
        print("[OK] All critical components verified")
        return True
    else:
        print(f"[ERROR] Failed to import: {', '.join(failed_imports)}")
        return False

def test_mediapipe():
    """Test MediaPipe face mesh functionality"""
    print("[TEST] Testing MediaPipe face mesh...")
    
    try:
        import mediapipe as mp
        import cv2
        import numpy as np
        
        # Initialize MediaPipe face mesh
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Create a test image (dummy face-like pattern)
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Process the test image
        rgb_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_image)
        
        print("[OK] MediaPipe face mesh initialized successfully")
        return True
        
    except Exception as e:
        print(f"[ERROR] MediaPipe test failed: {e}")
        return False

def display_system_info():
    """Display system information"""
    print("[SYSTEM] System Information:")
    print(f"   Operating System: {platform.system()} {platform.release()}")
    print(f"   Python Version: {sys.version}")
    print(f"   Python Executable: {sys.executable}")
    print(f"   Current Directory: {os.getcwd()}")

def create_readme():
    """Create a README file with setup information"""
    readme_content = '''# Gaze Tracking System - Setup Complete

## Installation Summary
Your gaze tracking system environment has been set up successfully!

## Directory Structure
- `calibration_data/` - Stores calibration data files
- `models/` - Stores trained machine learning models
- `movement_data/` - Stores eye movement analysis data
- `utils/` - Utility functions and feature engineering

## Next Steps
1. Run the main application: `python main.py`
2. Start with calibration to create your personal gaze model
3. Train the model using your calibration data
4. Use real-time gaze prediction

## System Requirements Met
[OK] Python 3.8+
[OK] OpenCV for computer vision
[OK] MediaPipe for face/eye detection
[OK] Scikit-learn for machine learning
[OK] NumPy and Pandas for data processing
[OK] Tkinter for GUI

## Troubleshooting
If you encounter issues:
1. Ensure camera permissions are granted
2. Check that no other application is using the camera
3. Restart the application if face detection fails
4. Re-run setup if packages are missing

## Camera Requirements
- Built-in webcam or USB camera
- Minimum 640x480 resolution
- Good lighting conditions recommended

Generated by setup_environment.py
'''
    
    with open("README_SETUP.md", "w") as f:
        f.write(readme_content)
    
    print("[INFO] Created README_SETUP.md with setup information")

def main():
    """Main setup function"""
    print("[SETUP] Gaze Tracking System - Environment Setup")
    print("=" * 50)
    
    # Step 1: Check Python version
    if not check_python_version():
        return False
    
    print()
    display_system_info()
    print()
    
    # Step 2: Upgrade pip
    upgrade_pip()
    print()
    
    # Step 3: Install packages
    packages_ok = install_core_packages()
    print()
    
    # Step 4: Create directories
    create_directory_structure()
    print()
    
    # Step 5: Verify installation
    verification_ok = verify_installation()
    print()
    
    # Step 6: Test MediaPipe
    mediapipe_ok = test_mediapipe()
    print()
    
    # Step 7: Check camera
    camera_ok = check_camera_access()
    print()
    
    # Step 8: Create README
    create_readme()
    print()
    
    # Final summary
    print("=" * 50)
    if packages_ok and verification_ok and mediapipe_ok:
        print("[SUCCESS] SETUP COMPLETE!")
        print("[OK] Your gaze tracking system is ready to use")
        print()
        print("[NEXT] What to do next:")
        print("   1. Run: python main.py")
        print("   2. Start with calibration")
        print("   3. Train your personal gaze model")
        print("   4. Enjoy real-time gaze tracking!")
        
        if not camera_ok:
            print()
            print("[WARN] Note: Camera test failed - please check camera setup before using the system")
        
        return True
    else:
        print("[ERROR] SETUP INCOMPLETE")
        print("Some components failed to install or verify.")
        print("Please check the error messages above and try running the setup again.")
        return False

if __name__ == "__main__":
    try:
        success = main()
        input("\\nPress Enter to continue...")
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\\n\\n[WARN] Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\\n\\n[ERROR] Unexpected error during setup: {e}")
        print("Please try running the setup again or contact support")
        sys.exit(1)
