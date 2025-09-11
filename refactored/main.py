"""
Streamlined Gaze Tracking System Main Interface
Multi-Output RandomForest Regressor Implementation
"""

import os
import tkinter as tk
from tkinter import messagebox, filedialog
from calibration import EyeTrackerCalibrator
from model_training import GazeModelTrainer
from prediction import GazePredictor

class GazeTrackingSystem:
    """Main interface for the streamlined gaze tracking system."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Streamlined Gaze Tracking System")
        self.root.geometry("500x400")
        self.root.resizable(False, False)
        
        # Center the window
        self.root.eval('tk::PlaceWindow . center')
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the user interface."""
        # Title
        title_label = tk.Label(self.root, 
                              text="Gaze Tracking System", 
                              font=("Arial", 18, "bold"),
                              fg="navy")
        title_label.pack(pady=15)
        
        # Subtitle
        subtitle_label = tk.Label(self.root, 
                                 text="Multi-Output RandomForest Regressor Implementation", 
                                 font=("Arial", 12),
                                 fg="gray")
        subtitle_label.pack(pady=5)
        
        # Main buttons frame
        main_frame = tk.Frame(self.root)
        main_frame.pack(pady=20)
        
        # Calibration button
        calib_btn = tk.Button(main_frame,
                             text="📍 Run Calibration",
                             command=self.run_calibration,
                             font=("Arial", 12, "bold"),
                             bg="#4CAF50",
                             fg="white",
                             width=20,
                             height=2)
        calib_btn.pack(pady=10)
        
        # Training button
        train_btn = tk.Button(main_frame,
                             text="🤖 Train Model",
                             command=self.train_model,
                             font=("Arial", 12, "bold"),
                             bg="#2196F3",
                             fg="white",
                             width=20,
                             height=2)
        train_btn.pack(pady=10)
        
        # Prediction button
        predict_btn = tk.Button(main_frame,
                               text="Real-Time Prediction",
                               command=self.run_prediction,
                               font=("Arial", 12, "bold"),
                               bg="#FF9800",
                               fg="white",
                               width=20,
                               height=2)
        predict_btn.pack(pady=10)
        
        # Quick workflow button
        workflow_btn = tk.Button(main_frame,
                                text="⚡ Complete Workflow",
                                command=self.run_complete_workflow,
                                font=("Arial", 12, "bold"),
                                bg="#9C27B0",
                                fg="white",
                                width=20,
                                height=2)
        workflow_btn.pack(pady=10)
        
        # Status frame
        status_frame = tk.Frame(self.root)
        status_frame.pack(side=tk.BOTTOM, pady=10)
        
        # Status label
        self.status_label = tk.Label(status_frame,
                                    text="Ready",
                                    font=("Arial", 10),
                                    fg="green")
        self.status_label.pack()
        
        # Info text
        info_text = tk.Text(self.root, height=6, width=60, font=("Arial", 9))
        info_text.pack(pady=10)
        info_text.insert(tk.END, 
                        "🔹 Calibration: Collect training data using 21-point calibration\n"
                        "🔹 Train Model: Build multi-output RandomForest Regressor model with Euclidean optimization\n"
                        "🔹 Real-Time Prediction: Use trained model for live gaze tracking\n"
                        "🔹 Complete Workflow: Run calibration → training → prediction in sequence\n\n"
                        "💡 The system uses advanced feature engineering and data augmentation\n"
                        "   for optimal gaze tracking accuracy with minimal error.")
        info_text.config(state=tk.DISABLED)
    
    def update_status(self, message, color="black"):
        """Update status message."""
        self.status_label.config(text=message, fg=color)
        self.root.update()
    
    def run_calibration(self):
        """Run the calibration process."""
        self.update_status("Running calibration...", "blue")
        
        try:
            calibrator = EyeTrackerCalibrator()
            csv_file = calibrator.run_calibration()
            
            if csv_file:
                self.update_status(f"Calibration completed: {os.path.basename(csv_file)}", "green")
                messagebox.showinfo("Success", f"Calibration completed!\nData saved to: {os.path.basename(csv_file)}")
            else:
                self.update_status("Calibration failed", "red")
                messagebox.showerror("Error", "Calibration failed. Please try again.")
                
        except Exception as e:
            self.update_status("Calibration error", "red")
            messagebox.showerror("Error", f"Calibration error: {str(e)}")
    
    def train_model(self):
        """Train the gaze model."""
        self.update_status("Training model...", "blue")
        
        # Get the current script directory and calibration data directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        calibration_dir = os.path.join(os.path.dirname(script_dir), 'calibration_data')
        
        # Check for calibration data in calibration_data directory first
        csv_files = []
        if os.path.exists(calibration_dir):
            csv_files = [f for f in os.listdir(calibration_dir) if f.startswith("calibration_data_") and f.endswith(".csv")]
            csv_files = [os.path.join(calibration_dir, f) for f in csv_files]  # Full paths
        
        # If no files in calibration_data directory, check script directory for backward compatibility
        if not csv_files:
            legacy_files = [f for f in os.listdir(script_dir) if f.startswith("calibration_data_") and f.endswith(".csv")]
            csv_files = [os.path.join(script_dir, f) for f in legacy_files]
        
        if not csv_files:
            # If no calibration files found, let user browse for one
            self.update_status("No calibration data found - please select file", "orange")
            initial_dir = calibration_dir if os.path.exists(calibration_dir) else script_dir
            csv_file = filedialog.askopenfilename(
                title="Select Calibration Data File",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                initialdir=initial_dir
            )
            
            if not csv_file:
                self.update_status("No calibration data selected", "red")
                messagebox.showerror("Error", "No calibration data selected. Please run calibration first or select a calibration file.")
                return
        else:
            # Use the most recent calibration file (csv_files already contains full paths)
            csv_files.sort(reverse=True)
            csv_file = csv_files[0]
            print(f"Using calibration file: {csv_file}")
        
        try:
            trainer = GazeModelTrainer()
            model_path = trainer.train_from_csv(csv_file, save_model=True, noise_level=5.0)
            
            if model_path:
                error = trainer.training_history.get('mean_euclidean_error', 'N/A')
                r2 = trainer.training_history.get('avg_r2_score', 'N/A')
                
                self.update_status(f"Model trained: {error:.1f}px error", "green")
                messagebox.showinfo("Success", 
                                   f"Model training completed!\n"
                                   f"Using data: {os.path.basename(csv_file)}\n"
                                   f"Model saved to: {os.path.basename(model_path)}\n"
                                   f"Test Error: {error:.2f} pixels\n"
                                   f"R² Score: {r2:.4f}")
            else:
                self.update_status("Training failed", "red")
                messagebox.showerror("Error", "Model training failed. Please try again.")
                
        except Exception as e:
            self.update_status("Training error", "red")
            messagebox.showerror("Error", f"Training error: {str(e)}")
    
    def run_prediction(self):
        """Run real-time prediction with choice between simple and advanced modes."""
        self.update_status("Starting prediction...", "blue")
        
        try:
            # Hide main window
            self.root.withdraw()
            
            # Create predictor and attempt to load default model
            predictor = GazePredictor()
            if predictor.load_default_model():
                self.update_status("Model loaded successfully", "green")
                
                # Show prediction mode selection in terminal
                print("\n" + "="*50)
                print("GAZE PREDICTION MODE SELECTION")
                print("="*50)
                print("1. Simple Gaze Prediction (Black screen with red dot)")
                print("2. Advanced Text Analysis (Text reading with analysis)")
                print("="*50)
                
                while True:
                    choice = input("Choose prediction mode (1 or 2): ").strip()
                    if choice == "1":
                        print("Starting Simple Gaze Prediction...")
                        self.update_status("Running simple gaze prediction", "green")
                        predictor.run_prediction(mode="standard")
                        break
                    elif choice == "2":
                        print("Starting Advanced Text Analysis...")
                        self.update_status("Running advanced text analysis", "green")
                        predictor.run_prediction_with_analysis(mode="text_analysis")
                        break
                    else:
                        print("Invalid choice. Please enter 1 or 2.")
                        
            else:
                # Fallback: let user browse for model file
                self.update_status("Default model not found - please select file", "orange")
                script_dir = os.path.dirname(os.path.abspath(__file__))
                models_dir = os.path.join(script_dir, "models")
                
                model_path = filedialog.askopenfilename(
                    title="Select Trained Model File",
                    filetypes=[("Joblib files", "*.joblib"), ("All files", "*.*")],
                    initialdir=models_dir if os.path.exists(models_dir) else script_dir
                )
                
                if model_path and predictor.load_model(model_path):
                    # Show prediction mode selection in terminal
                    print("\n" + "="*50)
                    print("GAZE PREDICTION MODE SELECTION")
                    print("="*50)
                    print("1. Simple Gaze Prediction (Black screen with red dot)")
                    print("2. Advanced Text Analysis (Text reading with analysis)")
                    print("="*50)
                    
                    while True:
                        choice = input("Choose prediction mode (1 or 2): ").strip()
                        if choice == "1":
                            print("Starting Simple Gaze Prediction...")
                            self.update_status("Running simple gaze prediction", "green")
                            predictor.run_prediction(mode="standard")
                            break
                        elif choice == "2":
                            print("Starting Advanced Text Analysis...")
                            self.update_status("Running advanced text analysis", "green")
                            predictor.run_prediction_with_analysis(mode="text_analysis")
                            break
                        else:
                            print("Invalid choice. Please enter 1 or 2.")
                else:
                    self.update_status("Failed to load model", "red")
                    messagebox.showerror("Error", "No trained models found. Please train a model first.")
                    
        except Exception as e:
            self.update_status(f"Prediction error: {str(e)}", "red")
            messagebox.showerror("Error", f"Prediction error: {str(e)}")
        finally:
            # Show main window again
            self.root.deiconify()
            self.update_status("Ready", "green")
    
    def run_complete_workflow(self):
        """Run the complete workflow: calibration → training → prediction."""
        if messagebox.askyesno("Complete Workflow", 
                              "This will run:\n"
                              "1. Calibration (collect data)\n"
                              "2. Model training\n"
                              "3. Real-time prediction\n\n"
                              "Continue?"):
            
            # Step 1: Calibration
            self.update_status("Step 1/3: Running calibration...", "blue")
            
            try:
                calibrator = EyeTrackerCalibrator()
                csv_file = calibrator.run_calibration()
                
                if not csv_file:
                    self.update_status("Workflow failed at calibration", "red")
                    messagebox.showerror("Error", "Calibration failed. Workflow stopped.")
                    return
                
                # Step 2: Training
                self.update_status("Step 2/3: Training model...", "blue")
                
                trainer = GazeModelTrainer()
                model_path = trainer.train_from_csv(csv_file, save_model=True, noise_level=5.0)
                
                if not model_path:
                    self.update_status("Workflow failed at training", "red")
                    messagebox.showerror("Error", "Model training failed. Workflow stopped.")
                    return
                
                # Step 3: Prediction
                self.update_status("Step 3/3: Starting prediction...", "blue")
                
                # Hide main window
                self.root.withdraw()
                
                predictor = GazePredictor(model_path)
                if predictor.model is not None:
                    error = trainer.training_history.get('mean_euclidean_error', 'N/A')
                    messagebox.showinfo("Workflow Complete", 
                                       f"Complete workflow finished!\n"
                                       f"Model Error: {error:.2f} pixels\n"
                                       f"Ready for prediction...")
                    
                    # Show prediction mode selection in terminal
                    print("\n" + "="*50)
                    print("COMPLETE WORKFLOW - PREDICTION MODE SELECTION")
                    print("="*50)
                    print("1. Simple Gaze Prediction (Black screen with red dot)")
                    print("2. Advanced Text Analysis (Text reading with analysis)")
                    print("="*50)
                    
                    while True:
                        choice = input("Choose prediction mode (1 or 2): ").strip()
                        if choice == "1":
                            print("Starting Simple Gaze Prediction...")
                            self.update_status("Running simple gaze prediction", "green")
                            predictor.run_prediction(mode="standard")
                            break
                        elif choice == "2":
                            print("Starting Advanced Text Analysis...")
                            self.update_status("Running advanced text analysis", "green")
                            predictor.run_prediction_with_analysis(mode="text_analysis")
                            break
                        else:
                            print("Invalid choice. Please enter 1 or 2.")
                else:
                    messagebox.showerror("Error", "Failed to load trained model.")
                
            except Exception as e:
                messagebox.showerror("Error", f"Workflow error: {str(e)}")
            
            finally:
                # Show main window again
                self.root.deiconify()
                self.update_status("Ready", "green")
    
    def run(self):
        """Start the application."""
        self.root.mainloop()

def main():
    """Main function."""
    print("Starting Streamlined Gaze Tracking System...")
    
    try:
        app = GazeTrackingSystem()
        app.run()
    except Exception as e:
        print(f"Application error: {str(e)}")

if __name__ == "__main__":
    main()
