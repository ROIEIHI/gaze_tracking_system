#!/usr/bin/env python3
"""
Gaze Tracking System - Setup Script
Hebrew RTL-Aware Eye Movement Tracking System

Cross-platform setup script for automatic dependency installation
"""

import subprocess
import sys
import os
import platform
from pathlib import Path

def run_command(command, description):
    """Run a shell command and handle errors"""
    print(f"[INFO] {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print(f"[SUCCESS] {description}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] {description} failed:")
        print(f"  Command: {command}")
        print(f"  Exit code: {e.returncode}")
        print(f"  Output: {e.stdout}")
        print(f"  Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    print(f"[INFO] Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major != 3 or version.minor < 8:
        print("[ERROR] Python 3.8 or higher is required!")
        print("Please install Python 3.8+ from https://python.org")
        return False
    
    print("[SUCCESS] Python version is compatible")
    return True

def install_package(package_name):
    """Install a single package with error handling"""
    command = f'"{sys.executable}" -m pip install {package_name}'
    return run_command(command, f"Installing {package_name}")

def verify_installation(package_name, import_name=None):
    """Verify that a package was installed correctly"""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        print(f"[SUCCESS] ✓ {package_name} verified")
        return True
    except ImportError as e:
        print(f"[ERROR] ✗ {package_name} verification failed: {e}")
        return False

def create_directories():
    """Create necessary project directories"""
    print("[INFO] Creating project directories...")
    
    base_dir = Path(__file__).parent.parent
    directories = [
        "calibration_data",
        "eye_tracking_data", 
        "models",
        "movement_data",
        "user_data"
    ]
    
    for dir_name in directories:
        dir_path = base_dir / dir_name
        dir_path.mkdir(exist_ok=True)
        print(f"[SUCCESS] ✓ {dir_path}")
    
    return True

def main():
    """Main setup function"""
    print("=" * 60)
    print("   Gaze Tracking System - Automated Setup")
    print("   Hebrew RTL-Aware Eye Movement Tracking")
    print("=" * 60)
    print()
    
    # Step 1: Check Python version
    print("[STEP 1/6] Checking Python version...")
    if not check_python_version():
        return 1
    
    # Step 2: Upgrade pip
    print("\n[STEP 2/6] Upgrading pip...")
    if not run_command(f'"{sys.executable}" -m pip install --upgrade pip', "Upgrading pip"):
        print("[WARNING] Pip upgrade failed, continuing anyway...")
    
    # Step 3: Install requirements from file
    print("\n[STEP 3/6] Installing packages from requirements.txt...")
    requirements_file = Path(__file__).parent / "requirements.txt"
    
    if requirements_file.exists():
        if not run_command(f'"{sys.executable}" -m pip install -r "{requirements_file}"', 
                          "Installing requirements"):
            print("[ERROR] Failed to install requirements")
            return 1
    else:
        print("[WARNING] requirements.txt not found, installing packages individually...")
        
        # Core packages - all actually used by the system
        packages = [
            "opencv-python>=4.8.0",
            "mediapipe>=0.10.0", 
            "numpy>=1.24.0",
            "pandas>=2.0.0",
            "scikit-learn>=1.3.0",
            "xgboost>=1.7.0",
            "scipy>=1.10.0",
            "joblib>=1.3.0",
            "Pillow>=10.0.0",
            "python-bidi>=0.4.2",
            "arabic-reshaper>=3.0.0"
        ]
        
        failed_packages = []
        for package in packages:
            if not install_package(package):
                failed_packages.append(package)
        
        if failed_packages:
            print(f"[ERROR] Failed to install: {', '.join(failed_packages)}")
            return 1
    
    # Step 4: Verify critical installations
    print("\n[STEP 4/6] Verifying installations...")
    
    verifications = [
        ("opencv-python", "cv2"),
        ("mediapipe", "mediapipe"),
        ("numpy", "numpy"), 
        ("pandas", "pandas"),
        ("scikit-learn", "sklearn"),
        ("xgboost", "xgboost"),
        ("scipy", "scipy"),
        ("joblib", "joblib"),
        ("Pillow", "PIL"),
        ("python-bidi", "bidi"),
        ("arabic-reshaper", "arabic_reshaper")
    ]
    
    failed_verifications = []
    for package_name, import_name in verifications:
        if not verify_installation(package_name, import_name):
            failed_verifications.append(package_name)
    
    # Check tkinter separately (usually built-in)
    try:
        import tkinter
        print("[SUCCESS] ✓ Tkinter (GUI) available")
    except ImportError:
        print("[ERROR] ✗ Tkinter not available - GUI may not work")
        print("  Try reinstalling Python with tkinter support")
        failed_verifications.append("tkinter")
    
    if failed_verifications:
        print(f"[ERROR] Verification failed for: {', '.join(failed_verifications)}")
        return 1
    
    # Step 5: Create directories
    print("\n[STEP 5/6] Creating project directories...")
    create_directories()
    
    # Step 6: Final success message
    print("\n[STEP 6/6] Setup complete!")
    print()
    print("=" * 60)
    print("   Setup Successful!")
    print("=" * 60)
    print()
    print("All dependencies have been installed successfully.")
    print("You can now run the system using:")
    print()
    print("  GUI Mode:         python gui.py")
    print("  Command Line:     python main.py")
    print()
    print("Project directories created:")
    print("  • calibration_data/  (for calibration files)")
    print("  • eye_tracking_data/ (for session data)")  
    print("  • models/           (for trained models)")
    print("  • movement_data/    (for movement analysis)")
    print("  • user_data/        (for user sessions)")
    print()
    
    # System-specific instructions
    system = platform.system()
    if system == "Windows":
        print("Windows users: Run 'python start.py' to launch the system")
    elif system == "Darwin":
        print("macOS users: Make sure Xcode command line tools are installed")
    elif system == "Linux":
        print("Linux users: Make sure python3-dev and python3-tkinter are installed")
    
    print("\nFor troubleshooting, see SETUP_README.md")
    return 0

if __name__ == "__main__":
    exit_code = main()
    if exit_code != 0:
        print("\n[SETUP FAILED] Please check the error messages above")
        print("For help, see SETUP_README.md or contact support")
    sys.exit(exit_code)