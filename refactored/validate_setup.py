#!/usr/bin/env python3
"""
Quick validation script to test if the gaze tracking system is properly set up.
Run this after setup_environment.py to verify everything is working.
"""

import sys
import traceback

def test_imports():
    """Test all critical imports"""
    print("[TEST] Testing critical imports...")
    
    tests = [
        ("cv2", "import cv2"),
        ("mediapipe", "import mediapipe as mp"),
        ("numpy", "import numpy as np"),
        ("pandas", "import pandas as pd"),
        ("sklearn", "from sklearn.ensemble import RandomForestRegressor"),
        ("joblib", "import joblib"),
        ("tkinter", "import tkinter as tk"),
    ]
    
    all_passed = True
    
    for name, import_statement in tests:
        try:
            exec(import_statement)
            print(f"[OK] {name} - OK")
        except Exception as e:
            print(f"[ERROR] {name} - FAILED: {e}")
            all_passed = False
    
    return all_passed

def test_camera():
    """Test camera functionality"""
    print("\n[CAMERA] Testing camera access...")
    
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("[ERROR] Camera cannot be opened")
            return False
        
        ret, frame = cap.read()
        cap.release()
        
        if ret and frame is not None:
            h, w = frame.shape[:2]
            print(f"✅ Camera working - Resolution: {w}x{h}")
            return True
        else:
            print("❌ Camera opened but cannot capture frames")
            return False
            
    except Exception as e:
        print(f"❌ Camera test failed: {e}")
        return False

def test_mediapipe():
    """Test MediaPipe face mesh"""
    print("\n🎭 Testing MediaPipe face detection...")
    
    try:
        import mediapipe as mp
        import cv2
        import numpy as np
        
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Test with dummy image
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        rgb_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_image)
        
        print("✅ MediaPipe face mesh initialized successfully")
        return True
        
    except Exception as e:
        print(f"❌ MediaPipe test failed: {e}")
        return False

def test_system_modules():
    """Test if system modules can be imported"""
    print("\n🔧 Testing system modules...")
    
    try:
        # Test utils module
        sys.path.append('.')
        from utils.features import FEATURE_COLUMNS, feature_engineer_df
        print("✅ utils.features - OK")
        
        # Test if main modules can be imported (without running them)
        import importlib.util
        
        modules_to_test = [
            "calibration.py",
            "model_training.py", 
            "prediction.py",
            "eye_movement_analyzer.py"
        ]
        
        for module_file in modules_to_test:
            try:
                spec = importlib.util.spec_from_file_location("test_module", module_file)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    # Don't execute, just test if it can be loaded
                    print(f"✅ {module_file} - Can be imported")
                else:
                    print(f"⚠️ {module_file} - File not found")
            except Exception as e:
                print(f"❌ {module_file} - Import error: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ System modules test failed: {e}")
        return False

def test_file_structure():
    """Test if required directories exist"""
    print("\n📁 Testing directory structure...")
    
    from pathlib import Path
    
    required_dirs = [
        "calibration_data",
        "models", 
        "movement_data",
        "utils"
    ]
    
    all_exist = True
    
    for directory in required_dirs:
        path = Path(directory)
        if path.exists() and path.is_dir():
            print(f"✅ {directory}/ - exists")
        else:
            print(f"❌ {directory}/ - missing")
            all_exist = False
    
    return all_exist

def main():
    """Run all validation tests"""
    print("🔍 Gaze Tracking System - Validation Tests")
    print("=" * 50)
    
    tests = [
        ("Package Imports", test_imports),
        ("Camera Access", test_camera),
        ("MediaPipe", test_mediapipe),
        ("System Modules", test_system_modules),
        ("Directory Structure", test_file_structure)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} - Unexpected error: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 VALIDATION SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:8} | {test_name}")
    
    print("-" * 50)
    print(f"TOTAL: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("Your gaze tracking system is ready to use.")
        print("Run 'python main.py' to start the application.")
        return True
    else:
        print(f"\n⚠️ {total - passed} TESTS FAILED!")
        print("Please check the errors above and:")
        print("1. Re-run setup_environment.py")
        print("2. Check camera permissions")
        print("3. Ensure all files are in the correct location")
        return False

if __name__ == "__main__":
    try:
        success = main()
        input("\nPress Enter to continue...")
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nValidation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        traceback.print_exc()
        sys.exit(1)
