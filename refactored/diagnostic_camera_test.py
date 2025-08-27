#!/usr/bin/env python3
"""
Diagnostic test to find exactly where camera initialization hangs
"""

import cv2
import sys
import signal
import time

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print('\n[SIGNAL] Received interrupt signal. Exiting...')
    sys.exit(0)

def test_step_by_step():
    """Test camera initialization step by step to find the hanging point"""
    signal.signal(signal.SIGINT, signal_handler)
    
    print("=== Step-by-Step Camera Initialization Test ===")
    cap = None
    
    try:
        print("[STEP 1] Testing basic VideoCapture creation...")
        cap = cv2.VideoCapture(0)
        print("[STEP 1] ✅ VideoCapture created successfully")
        
        print("[STEP 2] Testing isOpened()...")
        if not cap.isOpened():
            print("[STEP 2] ❌ Camera not opened")
            return False
        print("[STEP 2] ✅ Camera is opened")
        
        print("[STEP 3] Testing frame width property...")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
        print("[STEP 3] ✅ Frame width set")
        
        print("[STEP 4] Testing frame height property...")
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        print("[STEP 4] ✅ Frame height set")
        
        print("[STEP 5] Testing first frame read...")
        ret, frame = cap.read()
        if not ret:
            print("[STEP 5] ❌ Failed to read first frame")
            return False
        print(f"[STEP 5] ✅ First frame read successfully: {frame.shape}")
        
        print("[STEP 6] Testing second frame read...")
        ret, frame = cap.read()
        if not ret:
            print("[STEP 6] ❌ Failed to read second frame")
            return False
        print(f"[STEP 6] ✅ Second frame read successfully: {frame.shape}")
        
        print("[STEP 7] Testing camera release...")
        cap.release()
        print("[STEP 7] ✅ Camera released successfully")
        
        print("\n✅ ALL STEPS PASSED - Camera initialization working correctly!")
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR at current step: {e}")
        return False
        
    finally:
        if cap is not None:
            try:
                cap.release()
                print("[CLEANUP] Camera released in finally block")
            except:
                pass

def test_mediapipe_integration():
    """Test MediaPipe integration which might be causing the hang"""
    signal.signal(signal.SIGINT, signal_handler)
    
    print("\n=== MediaPipe Integration Test ===")
    
    try:
        print("[MP-STEP 1] Importing MediaPipe...")
        import mediapipe as mp
        print("[MP-STEP 1] ✅ MediaPipe imported")
        
        print("[MP-STEP 2] Creating FaceMesh...")
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        print("[MP-STEP 2] ✅ FaceMesh created")
        
        print("[MP-STEP 3] Creating camera...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[MP-STEP 3] ❌ Camera not opened")
            return False
        print("[MP-STEP 3] ✅ Camera created")
        
        print("[MP-STEP 4] Reading frame...")
        ret, frame = cap.read()
        if not ret:
            print("[MP-STEP 4] ❌ Failed to read frame")
            return False
        print(f"[MP-STEP 4] ✅ Frame read: {frame.shape}")
        
        print("[MP-STEP 5] Converting to RGB...")
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        print("[MP-STEP 5] ✅ RGB conversion successful")
        
        print("[MP-STEP 6] Processing with MediaPipe...")
        results = face_mesh.process(rgb_frame)
        print("[MP-STEP 6] ✅ MediaPipe processing successful")
        
        print("[MP-STEP 7] Cleaning up...")
        cap.release()
        print("[MP-STEP 7] ✅ Cleanup successful")
        
        print("\n✅ MEDIAPIPE INTEGRATION TEST PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR in MediaPipe test: {e}")
        return False

def test_calibration_import():
    """Test importing our calibration module"""
    signal.signal(signal.SIGINT, signal_handler)
    
    print("\n=== Calibration Module Import Test ===")
    
    try:
        print("[CAL-STEP 1] Importing calibration module...")
        sys.path.append('.')
        from calibration import EyeTrackerCalibrator
        print("[CAL-STEP 1] ✅ Calibration module imported")
        
        print("[CAL-STEP 2] Creating calibrator instance...")
        calibrator = EyeTrackerCalibrator()
        print("[CAL-STEP 2] ✅ Calibrator instance created")
        
        print("[CAL-STEP 3] Testing setup_camera method...")
        cap = calibrator.setup_camera()
        print("[CAL-STEP 3] ✅ Camera setup successful")
        
        print("[CAL-STEP 4] Testing frame read...")
        ret, frame = cap.read()
        if not ret:
            print("[CAL-STEP 4] ❌ Failed to read frame")
            return False
        print(f"[CAL-STEP 4] ✅ Frame read: {frame.shape}")
        
        print("[CAL-STEP 5] Releasing camera...")
        cap.release()
        print("[CAL-STEP 5] ✅ Camera released")
        
        print("\n✅ CALIBRATION MODULE TEST PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR in calibration test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Starting diagnostic tests...")
    print("Press Ctrl+C at any time to abort\n")
    
    # Test 1: Basic camera
    test1 = test_step_by_step()
    
    # Test 2: MediaPipe integration
    test2 = test_mediapipe_integration() if test1 else False
    
    # Test 3: Our calibration module
    test3 = test_calibration_import() if test2 else False
    
    print("\n" + "="*50)
    print("DIAGNOSTIC RESULTS:")
    print(f"Basic Camera: {'✅ PASS' if test1 else '❌ FAIL'}")
    print(f"MediaPipe Integration: {'✅ PASS' if test2 else '❌ FAIL'}")
    print(f"Calibration Module: {'✅ PASS' if test3 else '❌ FAIL'}")
    
    if test3:
        print("\n✅ All tests passed! Camera should work in main app.")
    else:
        print("\n❌ Found the problem! Check the failed test above.")
