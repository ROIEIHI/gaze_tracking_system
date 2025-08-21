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
        self.root.geometry("400x300")
        self.root.resizable(False, False)
        
        # Center the window
        self.root.eval('tk::PlaceWindow . center')
        
        # Title
        title_label = tk.Label(self.root, text="Gaze Tracking System", 
                              font=("Arial", 16, "bold"))
        title_label.pack(pady=20)
        
        # Description
        desc_label = tk.Label(self.root, 
                             text="Select the mode you want to run:",
                             font=("Arial", 12))
        desc_label.pack(pady=10)
        
        # Buttons frame
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=20)
        
        # Standard mode button
        standard_btn = tk.Button(button_frame, 
                               text="Standard Gaze Tracking", 
                               command=lambda: self.select_mode("standard"),
                               width=20, height=2,
                               font=("Arial", 10))
        standard_btn.pack(pady=5)
        
        # Text analysis mode button
        text_btn = tk.Button(button_frame, 
                           text="Text Reading Analysis", 
                           command=lambda: self.select_mode("text_analysis"),
                           width=20, height=2,
                           font=("Arial", 10))
        text_btn.pack(pady=5)
        
        # Calibration only button
        calib_btn = tk.Button(button_frame, 
                            text="Calibration Only", 
                            command=lambda: self.select_mode("calibration_only"),
                            width=20, height=2,
                            font=("Arial", 10))
        calib_btn.pack(pady=5)
        
        # Training only button
        train_btn = tk.Button(button_frame, 
                            text="Model Training Only", 
                            command=lambda: self.select_mode("training_only"),
                            width=20, height=2,
                            font=("Arial", 10))
        train_btn.pack(pady=5)
        
        # Prediction only button
        pred_btn = tk.Button(button_frame, 
                           text="Prediction Only", 
                           command=lambda: self.select_mode("prediction_only"),
                           width=20, height=2,
                           font=("Arial", 10))
        pred_btn.pack(pady=5)
        
        # Cancel button
        cancel_btn = tk.Button(button_frame, 
                             text="Cancel", 
                             command=lambda: self.select_mode(None),
                             width=20, height=1,
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
    
    def run_calibration_only(self):
        """Run calibration process only"""
        print("=== Calibration Mode ===")
        
        self.calibrator = EyeTrackerCalibrator()
        csv_file = self.calibrator.run_calibration()
        
        if csv_file:
            print(f"Calibration completed successfully!")
            print(f"Data saved to: {csv_file}")
        else:
            print("Calibration failed or was cancelled")
        
        return csv_file
    
    def run_training_only(self):
        """Run model training only"""
        print("=== Training Mode ===")
        
        # File selection dialog
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        
        csv_file = filedialog.askopenfilename(
            title="Select Calibration Data CSV File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        root.destroy()
        
        if not csv_file:
            print("No file selected. Training cancelled.")
            return None
        
        self.trainer = GazeModelTrainer()
        model_path = self.trainer.train_from_csv(csv_file)
        
        if model_path:
            print(f"Model training completed successfully!")
            print(f"Model saved to: {model_path}")
        else:
            print("Model training failed")
        
        return model_path
    
    def run_prediction_only(self):
        """Run prediction only"""
        print("=== Prediction Mode ===")
        
        # Model selection dialog
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        
        model_file = filedialog.askopenfilename(
            title="Select Trained Model File",
            filetypes=[("Joblib files", "*.joblib"), ("All files", "*.*")]
        )
        
        if not model_file:
            print("No model file selected. Prediction cancelled.")
            root.destroy()
            return
        
        # Mode selection for prediction
        mode_selector = ModeSelector()
        root.destroy()
        
        # Create a simple mode selector for prediction
        prediction_root = tk.Tk()
        prediction_root.title("Prediction Mode Selection")
        prediction_root.geometry("300x200")
        prediction_root.eval('tk::PlaceWindow . center')
        
        selected_prediction_mode = tk.StringVar(value="standard")
        
        tk.Label(prediction_root, text="Select Prediction Mode:", font=("Arial", 12)).pack(pady=20)
        
        tk.Radiobutton(prediction_root, text="Standard Gaze Tracking", 
                      variable=selected_prediction_mode, value="standard",
                      font=("Arial", 10)).pack(pady=5)
        
        tk.Radiobutton(prediction_root, text="Text Reading Analysis", 
                      variable=selected_prediction_mode, value="text_analysis",
                      font=("Arial", 10)).pack(pady=5)
        
        def start_prediction():
            prediction_root.quit()
            prediction_root.destroy()
        
        tk.Button(prediction_root, text="Start Prediction", 
                 command=start_prediction, width=15, height=2).pack(pady=20)
        
        prediction_root.mainloop()
        
        # Run prediction
        self.predictor = GazePredictor()
        self.predictor.run_prediction(model_file, mode=selected_prediction_mode.get())
    
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
        elif selected_mode == "calibration_only":
            self.run_calibration_only()
        elif selected_mode == "training_only":
            self.run_training_only()
        elif selected_mode == "prediction_only":
            self.run_prediction_only()
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
