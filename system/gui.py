#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gaze Tracking System - Professional GUI Interface
Properly integrates all system modules with comprehensive error handling
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import time
import os
import sys
from datetime import datetime
from typing import Optional

# Import all system modules
from config import *
from calibration import GazeCalibrator
from model_training import GazeModelTrainer
from prediction import TextReadingGazePredictor
from eye_movement_analyzer import EyeMovementAnalyzer

class GazeTrackingGUI:
    """Professional GUI for the Gaze Tracking System"""
    
    def __init__(self):
        """Initialize the GUI with proper error handling"""
        self.root = tk.Tk()
        self.setup_main_window()
        
        # System components
        self.calibrator: Optional[GazeCalibrator] = None
        self.trainer: Optional[GazeModelTrainer] = None
        self.predictor: Optional[TextReadingGazePredictor] = None
        
        # Session data
        self.user_name = ""
        self.session_dir = ""
        self.calibration_file = ""
        self.model_file = ""
        
        # GUI state
        self.current_step = 0
        self.is_running = False
        
        self.setup_interface()
        
    def setup_main_window(self):
        """Configure main window properties"""
        self.root.title("Gaze Tracking System - Main")
        
        # ADD FULLSCREEN CAPABILITY
        self.is_fullscreen = True
        
        # Set initial window size and configure
        self.root.configure(bg='black')
        
        if self.is_fullscreen:
            # Start in fullscreen mode
            self.root.attributes('-fullscreen', True)
            self.root.attributes('-topmost', True)
        else:
            # Start in windowed mode
            self.root.geometry("1200x800")
            # Center window
            self.root.update_idletasks()
            x = (self.root.winfo_screenwidth() // 2) - (1200 // 2)
            y = (self.root.winfo_screenheight() // 2) - (800 // 2)
            self.root.geometry(f"1200x800+{x}+{y}")
        
        # ADD FULLSCREEN TOGGLE KEYBINDINGS
        self.root.bind('<F11>', self.toggle_fullscreen)
        self.root.bind('<Escape>', self.exit_fullscreen)
        
    def setup_interface(self):
        """Create the main interface layout"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill="both", expand=True)
        
        # Header
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill="x", pady=(0, 20))
        
        title_label = ttk.Label(
            header_frame,
            text="Gaze Tracking and Reading Analysis System",
            font=("Arial", 24, "bold")
        )
        title_label.pack()
        
        # ADD FULLSCREEN INSTRUCTIONS
        instruction_label = ttk.Label(
            header_frame,
            text="Press F11 for fullscreen mode | ESC to exit fullscreen",
            font=("Arial", 10, "italic"),
            foreground="gray"
        )
        instruction_label.pack(pady=(5, 0))
        
        # User input section
        self.setup_user_section(main_frame)
        
        # Workflow section
        self.setup_workflow_section(main_frame)
        
        # Status section
        self.setup_status_section(main_frame)
        
        # Control buttons
        self.setup_control_buttons(main_frame)
        
    def setup_user_section(self, parent):
        """Create user input section"""
        user_frame = ttk.LabelFrame(parent, text="User Information", padding="10")
        user_frame.pack(fill="x", pady=(0, 10))
        
        # Name input
        name_frame = ttk.Frame(user_frame)
        name_frame.pack(fill="x")
        
        ttk.Label(name_frame, text="Participant Name:").pack(side="left")
        self.name_var = tk.StringVar()
        self.name_entry = ttk.Entry(name_frame, textvariable=self.name_var, width=30)
        self.name_entry.pack(side="left", padx=(10, 0))
        
        # Session info
        info_frame = ttk.Frame(user_frame)
        info_frame.pack(fill="x", pady=(10, 0))
        
        self.session_info_label = ttk.Label(
            info_frame,
            text="Session: Not Started",
            font=("Arial", 10, "italic")
        )
        self.session_info_label.pack(side="left")
        
    def setup_workflow_section(self, parent):
        """Create workflow progress section"""
        workflow_frame = ttk.LabelFrame(parent, text="System Workflow", padding="10")
        workflow_frame.pack(fill="both", expand=True, pady=(0, 10))
        
        # Workflow steps
        self.steps = [
            ("1. Face Detection & Calibration", "Collect gaze training data"),
            ("2. Model Training", "Build personalized ML model"),
            ("3. Text Reading Analysis", "Real-time reading tracking"),
            ("4. Data Export", "Generate analysis reports")
        ]
        
        self.step_frames = []
        self.step_labels = []
        self.step_progress = []
        
        for i, (title, description) in enumerate(self.steps):
            step_frame = ttk.Frame(workflow_frame)
            step_frame.pack(fill="x", pady=5)
            
            # Step indicator
            indicator = ttk.Label(
                step_frame,
                text="●",
                font=("Arial", 16),
                foreground="gray"
            )
            indicator.pack(side="left", padx=(0, 10))
            
            # Step content
            content_frame = ttk.Frame(step_frame)
            content_frame.pack(side="left", fill="x", expand=True)
            
            title_label = ttk.Label(
                content_frame,
                text=title,
                font=("Arial", 12, "bold")
            )
            title_label.pack(anchor="w")
            
            desc_label = ttk.Label(
                content_frame,
                text=description,
                font=("Arial", 10),
                foreground="gray"
            )
            desc_label.pack(anchor="w")
            
            # Progress bar
            progress = ttk.Progressbar(
                step_frame,
                length=100,
                mode='indeterminate'
            )
            progress.pack(side="right", padx=(10, 0))
            
            self.step_frames.append(step_frame)
            self.step_labels.append(indicator)
            self.step_progress.append(progress)
            
    def setup_status_section(self, parent):
        """Create status display section"""
        status_frame = ttk.LabelFrame(parent, text="System Status", padding="10")
        status_frame.pack(fill="x", pady=(0, 10))
        
        # Status text
        initial_status = "Ready to start... (Fullscreen mode active - Press F11 or ESC to toggle)"
        self.status_var = tk.StringVar(value=initial_status)
        self.status_label = ttk.Label(
            status_frame,
            textvariable=self.status_var,
            font=("Arial", 11)
        )
        self.status_label.pack(anchor="w")
        
        # Configuration info
        config_frame = ttk.Frame(status_frame)
        config_frame.pack(fill="x", pady=(5, 0))
        
        config_text = f"""Configuration: {SCREEN_WIDTH}×{SCREEN_HEIGHT} | Camera: {CAMERA_WIDTH}×{CAMERA_HEIGHT} | Reading: {'Hebrew RTL' if RTL_READING_DIRECTION == 'rtl' else 'English LTR'}"""
        
        ttk.Label(
            config_frame,
            text=config_text,
            font=("Arial", 9),
            foreground="gray"
        ).pack(anchor="w")
        
    def setup_control_buttons(self, parent):
        """Create control buttons section"""
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill="x")
        
        # Start button
        self.start_button = ttk.Button(
            button_frame,
            text="Start Session ▶",
            command=self.start_workflow,
            style="Accent.TButton"
        )
        self.start_button.pack(side="left", padx=(0, 10))
        
        # Stop button
        self.stop_button = ttk.Button(
            button_frame,
            text="Stop & Export",
            command=self.stop_workflow,
            state="disabled"
        )
        self.stop_button.pack(side="left", padx=(0, 10))
        
        # Settings button
        self.settings_button = ttk.Button(
            button_frame,
            text="Settings",
            command=self.show_settings
        )
        self.settings_button.pack(side="right")
        
        # Exit button
        self.exit_button = ttk.Button(
            button_frame,
            text="Exit",
            command=self.exit_application
        )
        self.exit_button.pack(side="right", padx=(0, 10))
        
    def update_status(self, message: str, step: int = None):
        """Update status message and workflow progress"""
        self.status_var.set(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
        
        if step is not None:
            # Update step indicators
            for i, (indicator, progress) in enumerate(zip(self.step_labels, self.step_progress)):
                if i < step:
                    indicator.config(foreground="green", text="✓")
                    progress.stop()
                elif i == step:
                    indicator.config(foreground="blue", text="●")
                    progress.start(10)
                else:
                    indicator.config(foreground="gray", text="●")
                    progress.stop()
                    
        self.root.update()
        
    def start_workflow(self):
        """Start the complete workflow in a separate thread"""
        if not self.validate_inputs():
            return
            
        self.is_running = True
        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")
        
        # OPTIONAL: Auto-enter fullscreen for workflow
        if not self.is_fullscreen:
            auto_fullscreen = messagebox.askyesno(
                "Fullscreen Mode", 
                "Would you like to enter fullscreen mode for the session?\n\n"
                "This provides a better experience for calibration and analysis."
            )
            if auto_fullscreen:
                self.toggle_fullscreen()
        
        # Create session directory
        self.create_session_directory()
        
        # Bring window to front
        self.bring_to_front()
        
        # Run workflow in separate thread to prevent GUI freezing
        workflow_thread = threading.Thread(target=self.run_workflow, daemon=True)
        workflow_thread.start()
        
    def validate_inputs(self) -> bool:
        """Validate user inputs"""
        if not self.name_var.get().strip():
            messagebox.showerror("Invalid Input", "Please enter participant name")
            self.name_entry.focus()
            return False
            
        # Check camera availability
        import cv2
        camera = cv2.VideoCapture(CAMERA_INDEX)
        if not camera.isOpened():
            messagebox.showerror("Camera Error", f"Cannot access camera {CAMERA_INDEX}")
            camera.release()
            return False
        camera.release()
        
        return True
        
    def create_session_directory(self):
        """Create user session directory"""
        self.user_name = self.name_var.get().strip().replace(" ", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(USER_DATA_DIR, f"{self.user_name}_{timestamp}")
        
        # Create directories
        for subdir in ["calibration", "models", "analysis"]:
            os.makedirs(os.path.join(self.session_dir, subdir), exist_ok=True)
            
        self.session_info_label.config(text=f"Session: {self.user_name} - {timestamp}")
        
    def run_workflow(self):
        """Execute the complete workflow"""
        try:
            # Step 1: Calibration
            self.update_status("Starting face detection and calibration...", 0)
            if not self.run_calibration():
                return
                
            # Step 2: Model Training  
            self.update_status("Training personalized gaze model...", 1)
            if not self.run_model_training():
                return
                
            # Step 3: Text Reading Analysis
            self.update_status("Running text reading analysis...", 2)
            if not self.run_text_analysis():
                return
                
            # Step 4: Data Export
            self.update_status("Exporting analysis data...", 3)
            self.export_final_data()
            
            # Complete
            self.update_status("Workflow completed successfully!", 4)
            
            # Show completion message and exit
            messagebox.showinfo(
                "Session Complete", 
                "Gaze tracking analysis completed successfully!\n\n"
                "All data has been saved.\n"
                "The application will now close."
            )
            
            # Exit the application
            self.exit_application()
            
        except Exception as e:
            self.update_status(f"Error: {str(e)}", None)
            messagebox.showerror("Workflow Error", f"An error occurred: {str(e)}")
        finally:
            # Only reset UI state if the application is still running
            # (i.e., user didn't choose to exit)
            if hasattr(self, 'root') and self.root.winfo_exists():
                self.is_running = False
                self.start_button.config(state="normal")
                self.stop_button.config(state="disabled")
            
    def run_calibration(self) -> bool:
        """Execute calibration step"""
        try:
            calib_dir = os.path.join(self.session_dir, "calibration")
            
            # Create calibrator with session-specific output directory
            self.calibrator = GazeCalibrator(output_dir=calib_dir)
            
            # Run calibration - it will save to session directory
            calibration_result = self.calibrator.run_full_calibration()

            if not calibration_result:
                raise Exception("Calibration was cancelled or failed")
                
            # The calibration module returns the full absolute path
            self.calibration_file = calibration_result
            
            if not os.path.exists(self.calibration_file):
                raise Exception(f"Calibration file not found: {self.calibration_file}")
        
            self.update_status(f"Calibration completed: {os.path.basename(self.calibration_file)}")
            return True
            
        except Exception as e:
            self.update_status(f"Calibration failed: {str(e)}")
            messagebox.showerror("Calibration Error", f"Calibration failed: {str(e)}")
            return False
            
    def run_model_training(self) -> bool:
        """Execute model training step"""
        try:
            # Create model directory
            model_dir = os.path.join(self.session_dir, "models")
            
            # Initialize trainer with session-specific paths
            self.trainer = GazeModelTrainer(
                calibration_file=self.calibration_file,
                output_dir=model_dir
            )
            
            # Train model - it will save to session directory
            self.model_file = self.trainer.train_complete_model()
            
            if not self.model_file:
                raise Exception("Model training failed")
                
            self.update_status(f"Model trained: {os.path.basename(self.model_file)}")
            return True
            
        except Exception as e:
            self.update_status(f"Model training failed: {str(e)}")
            messagebox.showerror("Training Error", f"Model training failed: {str(e)}")
            return False
            
    def run_text_analysis(self) -> bool:
        """Execute text reading analysis step"""
        try:
            analysis_dir = os.path.join(self.session_dir, "analysis")
            
            # Initialize predictor with session-specific model and output paths
            self.predictor = TextReadingGazePredictor(
                model_path=self.model_file,
                output_dir=analysis_dir
            )
            
            # Run analysis - it will save to session directory
            self.predictor.run_text_reading_analysis()
            
            self.update_status("Text reading analysis completed")
            
            # Show analysis summary GUI
            try:
                if self.predictor and hasattr(self.predictor, 'movement_analyzer'):
                    movement_analyzer = self.predictor.movement_analyzer
                    if movement_analyzer and hasattr(movement_analyzer, 'show_analysis_summary_gui'):
                        movement_analyzer.show_analysis_summary_gui()
            except Exception as e:
                print(f"Error showing analysis summary: {e}")
                messagebox.showinfo("Analysis Complete", 
                                  f"Reading analysis completed successfully!\n"
                                  f"Data saved to: {analysis_dir}")
            
            return True
            
        except Exception as e:
            self.update_status(f"Text analysis failed: {str(e)}")
            messagebox.showerror("Analysis Error", f"Text analysis failed: {str(e)}")
            return False
            
    def export_final_data(self):
        """Export final analysis data"""
        try:
            # Create summary report
            summary_file = os.path.join(self.session_dir, "session_summary.txt")
            
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write("GAZE TRACKING SESSION SUMMARY\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Participant: {self.user_name}\n")
                f.write(f"Session Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Calibration File: {os.path.basename(self.calibration_file)}\n")
                f.write(f"Model File: {os.path.basename(self.model_file)}\n")
                f.write(f"Reading Mode: {'Hebrew RTL' if RTL_READING_DIRECTION == 'rtl' else 'English LTR'}\n")
                f.write(f"Screen Resolution: {SCREEN_WIDTH}×{SCREEN_HEIGHT}\n")
                f.write(f"Text Font Size: {TEXT_FONT_SIZE}\n\n")
                f.write("Session completed successfully!\n")
                
            self.update_status(f"Session data exported to: {self.session_dir}")
            
        except Exception as e:
            self.update_status(f"Export warning: {str(e)}")
            
    def stop_workflow(self):
        """Stop the current workflow"""
        self.is_running = False
        self.update_status("Workflow stopped by user", None)
        
        # Reset GUI state
        for progress in self.step_progress:
            progress.stop()
        for indicator in self.step_labels:
            indicator.config(foreground="gray", text="●")
            
        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")
        
    def show_settings(self):
        """Show settings dialog"""
        settings_window = tk.Toplevel(self.root)
        settings_window.title("System Settings")
        settings_window.geometry("500x400")
        settings_window.transient(self.root)
        settings_window.grab_set()
        
        # Settings content
        ttk.Label(
            settings_window,
            text="System Configuration",
            font=("Arial", 14, "bold")
        ).pack(pady=10)
        
        # Display current settings
        settings_text = f"""
Current Configuration:
• Screen Resolution: {SCREEN_WIDTH} × {SCREEN_HEIGHT}
• Camera Resolution: {CAMERA_WIDTH} × {CAMERA_HEIGHT}
• Camera Index: {CAMERA_INDEX}
• Reading Direction: {RTL_READING_DIRECTION.upper()}
• Text Font Size: {TEXT_FONT_SIZE}
• Calibration Points: {CALIBRATION_GRID_SIZE}×{CALIBRATION_GRID_SIZE}
• Face Detection Confidence: {FACE_DETECTION_CONFIDENCE}
• Text Rows Per Page: {TEXT_ROWS_PER_PAGE}
• Words Per Row: {TEXT_WORDS_PER_ROW}
        """
        
        text_widget = tk.Text(settings_window, wrap="word", height=15)
        text_widget.pack(fill="both", expand=True, padx=20, pady=10)
        text_widget.insert("1.0", settings_text.strip())
        text_widget.config(state="disabled")
        
        # Close button
        ttk.Button(
            settings_window,
            text="Close",
            command=settings_window.destroy
        ).pack(pady=10)
        
    def exit_application(self):
        """Exit the application safely"""
        if self.is_running:
            if messagebox.askyesno("Confirm Exit", "Workflow is running. Do you want to stop and exit?"):
                self.stop_workflow()
            else:
                return
                
        self.root.quit()
        self.root.destroy()
        
    def toggle_fullscreen(self, event=None):
        """Toggle fullscreen mode"""
        self.is_fullscreen = not self.is_fullscreen
        
        if self.is_fullscreen:
            # Enter fullscreen
            self.root.attributes('-fullscreen', True)
            self.root.attributes('-topmost', True)
            self.update_status("Entered fullscreen mode (Press F11 or ESC to exit)")
        else:
            # Exit fullscreen
            self.root.attributes('-fullscreen', False)
            self.root.attributes('-topmost', False)
            self.root.geometry("1200x800")
            # Re-center window
            self.root.update_idletasks()
            x = (self.root.winfo_screenwidth() // 2) - (1200 // 2)
            y = (self.root.winfo_screenheight() // 2) - (800 // 2)
            self.root.geometry(f"1200x800+{x}+{y}")
            self.update_status("Exited fullscreen mode")

    def exit_fullscreen(self, event=None):
        """Exit fullscreen mode"""
        if self.is_fullscreen:
            self.is_fullscreen = False
            self.root.attributes('-fullscreen', False)
            self.root.attributes('-topmost', False)
            self.root.geometry("1200x800")
            # Re-center window
            self.root.update_idletasks()
            x = (self.root.winfo_screenwidth() // 2) - (1200 // 2)
            y = (self.root.winfo_screenheight() // 2) - (800 // 2)
            self.root.geometry(f"1200x800+{x}+{y}")
            self.update_status("Exited fullscreen mode")

    def bring_to_front(self):
        """Bring window to front and focus"""
        self.root.lift()
        self.root.focus_force()
        self.root.attributes('-topmost', True)
        self.root.after_idle(lambda: self.root.attributes('-topmost', False))
        
    def run(self):
        """Start the GUI application"""
        self.root.protocol("WM_DELETE_WINDOW", self.exit_application)
        self.root.mainloop()

def main():
    """Main function"""
    print("Starting Professional Gaze Tracking System...")
    
    # Create data directories
    os.makedirs(CALIBRATION_DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    try:
        app = GazeTrackingGUI()
        app.run()
    except Exception as e:
        print(f"Application error: {str(e)}")
        messagebox.showerror("Application Error", f"Failed to start application: {str(e)}")

if __name__ == "__main__":
    main()
