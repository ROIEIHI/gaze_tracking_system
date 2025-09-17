#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlined Gaze Tracking System Main Interface
Multi-Output RandomForest Regressor Implementation
"""

import os
import tkinter as tk
from tkinter import messagebox, filedialog, simpledialog
from calibration import EyeTrackerCalibrator
from model_training import GazeModelTrainer
from prediction import GazePredictor

class GazeTrackingSystem:
    """Main interface for the streamlined gaze tracking system."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Gaze Tracking System")
        self.root.geometry("500x400")
        self.root.resizable(False, False)
        
        # Center the window
        self.root.eval('tk::PlaceWindow . center')
        
        # Show welcome screen first
        self.setup_welcome_screen()
    
    def setup_welcome_screen(self):
        """Setup the welcome screen with name input."""
        # Clear any existing widgets
        for widget in self.root.winfo_children():
            widget.destroy()
        
        # Welcome title
        welcome_label = tk.Label(self.root, 
                                text="Welcome to Gaze Tracking System", 
                                font=("Arial", 20, "bold"),
                                fg="navy")
        welcome_label.pack(pady=40)
        
        # Subtitle
        subtitle_label = tk.Label(self.root, 
                                 text="Advanced Eye Tracking with Machine Learning", 
                                 font=("Arial", 12),
                                 fg="gray")
        subtitle_label.pack(pady=10)
        
        # Name input frame
        name_frame = tk.Frame(self.root)
        name_frame.pack(pady=30)
        
        # Name input label
        name_label = tk.Label(name_frame, 
                             text="Please enter your name:", 
                             font=("Arial", 14),
                             fg="black")
        name_label.pack(pady=10)
        
        # Name input field
        self.name_entry = tk.Entry(name_frame, 
                                  font=("Arial", 12),
                                  width=25,
                                  justify='center')
        self.name_entry.pack(pady=10)
        self.name_entry.focus()
        
        # Bind Enter key to start session
        self.name_entry.bind('<Return>', lambda event: self.start_session())
        
        # Start button
        start_btn = tk.Button(name_frame,
                             text="Start Session",
                             command=self.start_session,
                             font=("Arial", 12, "bold"),
                             bg="#4CAF50",
                             fg="white",
                             width=15,
                             height=2)
        start_btn.pack(pady=20)
        
        # Info text
        info_label = tk.Label(self.root, 
                             text="Your personalized gaze tracking session will begin\nwith calibration, training, and real-time prediction", 
                             font=("Arial", 10),
                             fg="gray",
                             justify='center')
        info_label.pack(pady=20)
    
    def start_session(self):
        """Start the session after getting user name."""
        name = self.name_entry.get().strip()
        
        if not name:
            messagebox.showerror("Invalid Name", "Please enter your name to continue.")
            return
        
        # Clean the name (remove special characters, spaces)
        clean_name = "".join(c for c in name if c.isalnum() or c in (' ', '-', '_')).strip()
        
        if not clean_name:
            messagebox.showerror("Invalid Name", "Please enter a valid name.")
            return
        
        self.user_name = clean_name.replace(' ', '_')
        
        # Create user-specific directories
        self.setup_user_directories()
        
        # Save user name to file
        self.save_user_session()
        
        # Update window title
        self.root.title(f"Gaze Tracking System - {self.user_name}")
        
        # Show main interface
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the main user interface."""
        # Clear any existing widgets
        for widget in self.root.winfo_children():
            widget.destroy()
            
        # Title with user name
        title_label = tk.Label(self.root, 
                              text=f"Gaze Tracking System - {self.user_name}", 
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
        main_frame.pack(pady=40)
        
        # Complete workflow button (only option)
        workflow_btn = tk.Button(main_frame,
                                text="Start Gaze Tracking Session",
                                command=self.run_complete_workflow,
                                font=("Arial", 14, "bold"),
                                bg="#9C27B0",
                                fg="white",
                                width=25,
                                height=3)
        workflow_btn.pack(pady=20)
        
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
        info_text = tk.Text(self.root, height=4, width=60, font=("Arial", 9))
        info_text.pack(pady=10)
        info_text.insert(tk.END, 
                        "Streamlined Gaze Tracking Session:\n"
                        "   • Enter your name for personalized data organization\n"
                        "   • Automated calibration -> training -> text analysis\n"
                        "   • All session data saved in your personal folder\n\n"
                        "Complete end-to-end gaze tracking with automatic text analysis")
        info_text.config(state=tk.DISABLED)
    
    def setup_user_directories(self):
        """Create user-specific directories for data organization."""
        # Get the project root directory (parent of refactored)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        
        # Create user-specific directory structure
        self.user_dir = os.path.join(project_root, 'user_sessions', self.user_name)
        self.user_calibration_dir = os.path.join(self.user_dir, 'calibration_data')
        self.user_movement_dir = os.path.join(self.user_dir, 'movement_data')
        self.user_models_dir = os.path.join(self.user_dir, 'models')
        
        # Create directories if they don't exist
        for directory in [self.user_dir, self.user_calibration_dir, self.user_movement_dir, self.user_models_dir]:
            os.makedirs(directory, exist_ok=True)
    
    def save_user_session(self):
        """Save user session information to file."""
        from datetime import datetime
        
        session_info = {
            'user_name': self.user_name,
            'session_start': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'user_directory': self.user_dir
        }
        
        # Save to user directory
        session_file = os.path.join(self.user_dir, 'session_info.txt')
        with open(session_file, 'w', encoding='utf-8') as f:
            for key, value in session_info.items():
                f.write(f"{key}: {value}\n")
        
        print(f"Session started for user: {self.user_name}")
        print(f"Data will be saved to: {self.user_dir}")
    
    def update_status(self, message, color="black"):
        """Update status message."""
        self.status_label.config(text=message, fg=color)
        self.root.update()
    
    def run_complete_workflow(self):
        """Run the complete workflow: calibration -> training -> prediction with user-specific data organization."""
        if messagebox.askyesno("Start Gaze Tracking Session", 
                              f"Ready to start gaze tracking session for {self.user_name}?\n\n"
                              "This will run:\n"
                              "1. Calibration (collect training data)\n"
                              "2. Model training\n"
                              "3. Automatic text analysis prediction\n\n"
                              f"All data will be saved in: user_sessions/{self.user_name}/\n\n"
                              "Continue?"):
            
            # Hide main window immediately after user confirms
            self.root.withdraw()
            
            try:
                # Step 1: Calibration with user-specific directory
                self.update_status("Step 1/3: Running calibration...", "blue")
                
                calibrator = EyeTrackerCalibrator()
                # Run calibration (we'll move the file to user directory afterward)
                csv_file = calibrator.run_calibration()
                
                if csv_file:
                    # Move the calibration file to user-specific directory
                    import shutil
                    filename = os.path.basename(csv_file)
                    user_csv_file = os.path.join(self.user_calibration_dir, filename)
                    shutil.move(csv_file, user_csv_file)
                    csv_file = user_csv_file
                
                if not csv_file:
                    self.update_status("Workflow failed at calibration", "red")
                    messagebox.showerror("Error", "Calibration failed. Workflow stopped.")
                    return
                
                # Step 2: Training with user-specific model directory
                self.update_status("Step 2/3: Training model...", "blue")
                
                trainer = GazeModelTrainer()
                # Train model (we'll move the model file to user directory afterward)
                model_path = trainer.train_from_csv(csv_file, save_model=True, noise_level=5.0)
                
                if model_path:
                    # Move the model file to user-specific directory
                    import shutil
                    filename = os.path.basename(model_path)
                    user_model_path = os.path.join(self.user_models_dir, filename)
                    shutil.move(model_path, user_model_path)
                    model_path = user_model_path
                
                if not model_path:
                    self.update_status("Workflow failed at training", "red")
                    messagebox.showerror("Error", "Model training failed. Workflow stopped.")
                    return
                
                # Step 3: Automatic Text Analysis Prediction
                self.update_status("Step 3/3: Starting text analysis...", "blue")
                
                predictor = GazePredictor(model_path)
                # Set user-specific movement data directory
                predictor.movement_data_dir = self.user_movement_dir
                
                if predictor.model is not None:
                    error = trainer.training_history.get('mean_euclidean_error', 'N/A')
                    print(f"\n{'='*60}")
                    print(f"GAZE TRACKING SESSION COMPLETE - {self.user_name.upper()}")
                    print(f"{'='*60}")
                    print(f"Model Error: {error:.2f} pixels")
                    print(f"Data saved in: user_sessions/{self.user_name}/")
                    print(f"Starting automatic text analysis...")
                    print(f"{'='*60}")
                    
                    # Automatically start Advanced Text Analysis
                    self.update_status("Running text analysis", "green")
                    predictor.run_prediction_with_analysis(mode="text_analysis")
                else:
                    messagebox.showerror("Error", "Failed to load trained model.")
                
            except Exception as e:
                messagebox.showerror("Error", f"Workflow error: {str(e)}")
            
            finally:
                # Close the application after prediction is complete
                self.root.quit()
                self.root.destroy()
    
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