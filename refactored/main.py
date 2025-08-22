import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from calibration import EyeTrackerCalibrator
from model_training import GazeModelTrainer
from prediction import GazePredictor

class ModeSelector:
    def __init__(self):
        self.selected_mode = None
        self.root = None
    
    def select_mode(self, mode):
        """Select mode and close window"""
        self.selected_mode = mode
        if self.root:
            self.root.destroy()
    
    def get_selection(self):
        """Display mode selection window and return selected mode"""
        self.root = tk.Tk()
        self.root.title("Gaze Tracking System - Mode Selection")
        self.root.geometry("450x350")
        self.root.resizable(False, False)
        
        # Center the window
        self.root.eval('tk::PlaceWindow . center')
        
        # Title
        title_label = tk.Label(self.root, text="Gaze Tracking System", 
                              font=("Arial", 16, "bold"))
        title_label.pack(pady=10)
        
        # Description
        desc_label = tk.Label(self.root, 
                             text="Select the mode you want to run:",
                             font=("Arial", 12))
        desc_label.pack(pady=5)
        
        # Info about automatic analysis
        info_label = tk.Label(self.root, 
                             text="💡 Text Reading mode automatically collects\neye movement data for model training",
                             font=("Arial", 9),
                             fg="blue")
        info_label.pack(pady=2)
        
        # Buttons frame
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=10)
        
        # Standard mode button
        standard_btn = tk.Button(button_frame, 
                               text="Standard Gaze Tracking", 
                               command=lambda: self.select_mode("standard"),
                               width=25, height=2,
                               font=("Arial", 10))
        standard_btn.pack(pady=5)
        
        # Text analysis mode button (with automatic movement analysis)
        text_btn = tk.Button(button_frame, 
                           text="Text Reading Analysis\n(Auto Eye Movement Data Collection)", 
                           command=lambda: self.select_mode("text_analysis"),
                           width=35, height=3,
                           font=("Arial", 9),
                           bg="#e8f5e8")  # Light green background to highlight
        text_btn.pack(pady=5)
        
        # Cancel button
        cancel_btn = tk.Button(button_frame, 
                             text="Cancel", 
                             command=lambda: self.select_mode(None),
                             width=25, height=1,
                             font=("Arial", 10))
        cancel_btn.pack(pady=10)
        
        # Run the window
        self.root.mainloop()
        
        return self.selected_mode

class GazeTrackingSystem:
    def __init__(self):
        self.calibrator = None
        self.trainer = None
        self.predictor = None
        
        # Create necessary directories
        self.setup_directories()
    
    def setup_directories(self):
        """Create necessary directories for the system"""
        base_dir = os.path.dirname(__file__)
        
        directories = [
            os.path.join(base_dir, 'data'),
            os.path.join(base_dir, 'models'),
            os.path.join(base_dir, 'assets')
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def run_full_pipeline(self, mode="standard"):
        """Run the complete gaze tracking pipeline"""
        print("=== Full Gaze Tracking Pipeline ===")
        
        # Step 1: Calibration
        print("\nStep 1: Calibration")
        self.calibrator = EyeTrackerCalibrator()
        csv_file = self.calibrator.run_calibration()
        
        if not csv_file:
            print("Calibration failed. Pipeline aborted.")
            return
        
        # Step 2: Model Training
        print("\nStep 2: Model Training")
        self.trainer = GazeModelTrainer()
        model_path = self.trainer.train_from_csv(csv_file, create_visualizations=False)
        
        if not model_path:
            print("Model training failed. Pipeline aborted.")
            return
        
        # Step 3: Real-time Prediction
        print("\nStep 3: Real-time Prediction")
        self.predictor = GazePredictor()
        self.predictor.run_prediction(model_path, mode=mode)
        
        print("=== Full Pipeline Complete ===")
    
    def run_system(self):
        """Main system entry point"""
        print("Welcome to the Gaze Tracking System!")
        
        # Mode selection
        mode_selector = ModeSelector()
        selected_mode = mode_selector.get_selection()
        
        if selected_mode is None:
            print("No mode selected. Exiting.")
            return
        
        # Execute based on selected mode
        if selected_mode == "standard":
            self.run_full_pipeline(mode="standard")
        elif selected_mode == "text_analysis":
            self.run_full_pipeline(mode="text_analysis")
        else:
            print(f"Unknown mode: {selected_mode}")

def main():
    """Main application entry point"""
    try:
        system = GazeTrackingSystem()
        system.run_system()
    except KeyboardInterrupt:
        print("\nApplication interrupted by user.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
