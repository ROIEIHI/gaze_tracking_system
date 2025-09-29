#!/usr/bin/env python3
"""
Gaze Tracking System - Quick Launcher
Hebrew RTL-Aware Eye Movement Tracking System

Cross-platform launcher script for the gaze tracking system
"""

import subprocess
import sys
import os
from pathlib import Path

def check_dependencies():
    """Check if critical dependencies are available"""
    critical_packages = [
        'cv2', 'mediapipe', 'numpy', 'pandas', 'sklearn', 
        'bidi.algorithm', 'arabic_reshaper', 'tkinter'
    ]
    
    missing = []
    for package in critical_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    return missing

def main():
    """Main launcher function"""
    print("=" * 50)
    print("   Gaze Tracking System Launcher")
    print("   Hebrew RTL-Aware Eye Tracking")
    print("=" * 50)
    print()
    
    # Change to system directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("[ERROR] Python 3.8 or higher is required!")
        print(f"Current version: {sys.version}")
        input("Press Enter to exit...")
        return 1
    
    print(f"[INFO] Python version: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    
    # Check dependencies
    print("[INFO] Checking dependencies...")
    missing = check_dependencies()
    
    if missing:
        print("[WARNING] Missing dependencies:")
        for pkg in missing:
            print(f"  • {pkg}")
        print()
        print("Please run the setup script first:")
        print("  Windows: setup.bat")
        print("  All platforms: python setup.py")
        print()
        
        response = input("Continue anyway? (y/N): ").lower().strip()
        if response != 'y':
            return 1
    else:
        print("[SUCCESS] All dependencies available")
    
    # Launch the GUI application
    print()
    print("[INFO] Starting GUI application...")
    print("Close this window to exit the system.")
    print()
    
    try:
        # Start gui.py
        result = subprocess.run([sys.executable, "gui.py"], cwd=script_dir)
        return result.returncode
    except KeyboardInterrupt:
        print("\n[INFO] Application interrupted by user")
        return 0
    except Exception as e:
        print(f"[ERROR] Failed to start application: {e}")
        input("Press Enter to exit...")
        return 1

if __name__ == "__main__":
    exit_code = main()
    
    if exit_code == 0:
        print("\n[INFO] Application closed normally")
    else:
        print(f"\n[ERROR] Application exited with code {exit_code}")
        input("Press Enter to exit...")
    
    sys.exit(exit_code)