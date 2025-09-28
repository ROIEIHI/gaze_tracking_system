"""
Main entry point for the Gaze Tracking Reading Analysis System
Coordinates calibration and model training phases
"""

import sys
import os
from calibration import GazeCalibrator
from model_training import train_model_from_file
from config import *
from prediction import TextReadingGazePredictor

def main():
    """Main system entry point"""
    print("="*60)
    print("GAZE TRACKING READING ANALYSIS SYSTEM")
    print("="*60)
    print("This system will:")
    print("1. Calibrate your gaze tracking")
    print("2. Train a machine learning model")
    print("3. Start text reading analysis") 
    print("\nPress ENTER to continue or CTRL+C to exit...")
    
    try:
        input()
    except (KeyboardInterrupt, EOFError):
        print("\nSystem cancelled by user or no input available")
        return

    # Phase 1-2: Calibration
    print("\n" + "="*60)
    print("STARTING CALIBRATION")
    print("="*60)
    calibrator = GazeCalibrator()
    calibration_file = calibrator.run_full_calibration()
    
    if not calibration_file:
        print("Calibration failed or was cancelled")
        return

    # Phase 3: Model Training
    print("\n" + "="*60)
    print("STARTING MODEL TRAINING")
    print("="*60)
    model_path = train_model_from_file(calibration_file)
    
    if not model_path:
        print("Model training failed")
        return

    # Phase 4: Text Reading Analysis
    print("\n" + "="*60)
    print("STARTING TEXT READING ANALYSIS")
    print("="*60)
    print(f"Calibration completed: {os.path.basename(calibration_file)}")
    print(f"Model trained: {os.path.basename(model_path)}")
    print("\nStarting text reading analysis system...")
    print("This will show text on screen and track your reading patterns.")
    print("Press ENTER to continue or CTRL+C to skip...")

    try:
        input()
        predictor = TextReadingGazePredictor(model_path)
        predictor.run_text_reading_analysis()
    except (KeyboardInterrupt, EOFError):
        print("Text reading analysis skipped")

    # System Complete
    print("\n" + "="*60)
    print("SYSTEM COMPLETE!")
    print("="*60)
    print("Your gaze tracking system is fully operational!")
    print("Calibration data saved")
    print("Model trained and saved")  
    print("Text reading analysis completed")
    print("\nThank you for using the Gaze Tracking Reading Analysis System!")

def run_calibration_only():
    """Run only the calibration phase"""
    print("Running calibration only...")
    calibrator = GazeCalibrator()
    return calibrator.run_full_calibration()

def run_training_only(calibration_file: str):
    """Run only the model training phase"""
    if not os.path.exists(calibration_file):
        print(f"Calibration file not found: {calibration_file}")
        return ""
    
    print(f"Training model from: {calibration_file}")
    return train_model_from_file(calibration_file)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "calibrate":
            # Run calibration only
            calibration_file = run_calibration_only()
            if calibration_file:
                print(f"Calibration completed: {calibration_file}")
        elif sys.argv[1] == "train" and len(sys.argv) > 2:
            # Run training only
            model_path = run_training_only(sys.argv[2])
            if model_path:
                print(f"Training completed: {model_path}")
        else:
            print("Usage:")
            print("  python main.py                    # Run full system")
            print("  python main.py calibrate          # Run calibration only")
            print("  python main.py train <file.csv>   # Run training only")
    else:
        # Run full system
        main()
