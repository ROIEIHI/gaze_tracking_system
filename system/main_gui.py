#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""        # Welcome title
        welcome_label = tk.Label(self.root, 
                                text="Welcome to Gaze Tracking System",
                                font=("Arial", 48, "bold"),
                                fg="white",
                                bg="black")
        welcome_label.pack(pady=100) Tracking System - GUI Interface
Provides a user-friendly graphical interface for the gaze tracking system
while preserving all existing functionality from the system directory.
"""

import os
import sys
import tkinter as tk

from tkinter import messagebox, simpledialog
from datetime import datetime

# Import existing system modules
from calibration import GazeCalibrator
from model_training import train_model_from_file
from prediction import TextReadingGazePredictor

class GazeTrackingSystemGUI:
    """GUI interface for the gaze tracking system."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Gaze Tracking System")
        
        # Set full screen mode
        self.root.attributes('-fullscreen', True)
        self.root.configure(bg='black')
        
        # Bind Escape key to exit fullscreen (for debugging/emergency exit)
        self.root.bind('<Escape>', self.exit_fullscreen)
        
        # Initialize variables
        self.user_name = None
        self.calibration_file = None
        self.model_path = None
        
        # Show welcome screen first
        self.setup_welcome_screen()
    
    def exit_fullscreen(self, event=None):
        """Exit fullscreen mode (for debugging)."""
        self.root.attributes('-fullscreen', False)
    
    def setup_welcome_screen(self):
        """Setup the welcome screen with name input."""
        # Clear any existing widgets
        for widget in self.root.winfo_children():
            widget.destroy()
        
        # Welcome title
        welcome_label = tk.Label(self.root, 
                                text="Welcome to Gaze Tracking System", 
                                font=("Arial", 48, "bold"),
                                fg="white",
                                bg="black")
        welcome_label.pack(pady=100)
        
        # Subtitle
        subtitle_label = tk.Label(self.root, 
                                 text="Advanced Eye Tracking with Reading Analysis", 
                                 font=("Arial", 24),
                                 fg="lightgray",
                                 bg="black")
        subtitle_label.pack(pady=30)
        
        # Name input frame
        name_frame = tk.Frame(self.root, bg="black")
        name_frame.pack(pady=50)
        
        # Name input label
        name_label = tk.Label(name_frame, 
                             text="Please enter your name:", 
                             font=("Arial", 24),
                             fg="white",
                             bg="black")
        name_label.pack(pady=30)
        
        # Name input field
        self.name_entry = tk.Entry(name_frame, 
                                  font=("Arial", 20),
                                  width=30,
                                  justify='center')
        self.name_entry.pack(pady=20)
        self.name_entry.focus()
        
        # Bind Enter key to start session
        self.name_entry.bind('<Return>', lambda event: self.start_session())
        
        # Start button
        start_btn = tk.Button(name_frame,
                             text="Start Gaze Tracking Session",
                             command=self.start_session,
                             font=("Arial", 20, "bold"),
                             bg="#4CAF50",
                             fg="white",
                             width=30,
                             height=3)
        start_btn.pack(pady=40)
    
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
        
        # Create a minimal status label for status updates
        self.status_label = tk.Label(self.root, text="Initializing...", font=("Arial", 12))
        
        # Start workflow immediately (skip confirmation screen)
        self.run_complete_workflow()
    
    def setup_user_directories(self):
        """Create user-specific directories for data organization."""
        # Get the system directory (current directory)
        system_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(system_dir)
        
        # Create user-specific directory structure
        self.user_dir = os.path.join(project_root, 'user_data', self.user_name)
        self.user_calibration_dir = os.path.join(self.user_dir, 'calibration')
        self.user_models_dir = os.path.join(self.user_dir, 'models')
        self.user_analysis_dir = os.path.join(self.user_dir, 'analysis')
        
        # Create directories if they don't exist
        for directory in [self.user_dir, self.user_calibration_dir, 
                         self.user_models_dir, self.user_analysis_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Save session info
        session_info = {
            'user_name': self.user_name,
            'session_start': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'user_directory': self.user_dir
        }
        
        session_file = os.path.join(self.user_dir, 'session_info.txt')
        with open(session_file, 'w', encoding='utf-8') as f:
            for key, value in session_info.items():
                f.write(f"{key}: {value}\\n")
        
        print(f"Session started for user: {self.user_name}")
        print(f"Data will be saved to: {self.user_dir}")
    
    def setup_main_interface(self):
        """Setup the main system interface."""
        # Clear any existing widgets
        for widget in self.root.winfo_children():
            widget.destroy()
            
        # Update window title
        self.root.title(f"Gaze Tracking System - {self.user_name}")
        
        # Main title
        title_label = tk.Label(self.root, 
                              text=f"Gaze Tracking System - {self.user_name}", 
                              font=("Arial", 20, "bold"),
                              fg="navy")
        title_label.pack(pady=20)
        
        # System info
        info_frame = tk.Frame(self.root)
        info_frame.pack(pady=20)
        
        system_info = tk.Label(info_frame, 
                              text="Complete Gaze Tracking Workflow", 
                              font=("Arial", 16, "bold"),
                              fg="darkgreen")
        system_info.pack()
        
        # Workflow steps
        steps_frame = tk.Frame(self.root)
        steps_frame.pack(pady=30)
        
        steps_text = """
1. CALIBRATION: Collect gaze training data
2. MODEL TRAINING: Build machine learning model  
3. TEXT ANALYSIS: Real-time reading analysis

Click "Start Complete Workflow" to begin the automated process.
All data will be saved in your personal folder.
        """
        
        steps_label = tk.Label(steps_frame, 
                              text=steps_text,
                              font=("Arial", 12),
                              justify='left',
                              fg="black")
        steps_label.pack()
        
        # Start workflow button
        workflow_btn = tk.Button(self.root,
                                text="Start Complete Workflow",
                                command=self.run_complete_workflow,
                                font=("Arial", 16, "bold"),
                                bg="#2196F3",
                                fg="white",
                                width=25,
                                height=3)
        workflow_btn.pack(pady=30)
        
        # Status label
        self.status_label = tk.Label(self.root,
                                    text="Ready to start workflow",
                                    font=("Arial", 12),
                                    fg="green")
        self.status_label.pack(pady=10)
    
    def update_status(self, message, color="black"):
        """Update status message."""
        self.status_label.config(text=message, fg=color)
        self.root.update()
    
    def show_wait_window(self):
        """Show a simple wait window during model training."""
        print("DEBUG: Creating wait window")
        # Create wait window
        self.wait_window = tk.Toplevel()  # Don't parent to root since root might be minimized
        self.wait_window.title("Training in Progress")
        
        # Make wait window fullscreen
        self.wait_window.attributes('-fullscreen', True)
        self.wait_window.configure(bg='black')
        
        # Make window modal and always on top
        self.wait_window.grab_set()
        self.wait_window.attributes('-topmost', True)
        
        # Message centered on screen
        message_label = tk.Label(self.wait_window, 
                               text="Please wait until the training process is complete",
                               font=("Arial", 48, "bold"),
                               fg="white",
                               bg="black")
        message_label.pack(expand=True)
        
        # Additional instruction text
        instruction_label = tk.Label(self.wait_window, 
                                   text="The system is training your personalized gaze model...\nThis may take a few moments.",
                                   font=("Arial", 24),
                                   fg="lightgray",
                                   bg="black",
                                   justify='center')
        instruction_label.pack(pady=50)
        
        # Force the window to display immediately
        self.wait_window.update_idletasks()
        self.wait_window.update()
        self.wait_window.lift()
        self.wait_window.focus_force()
        self.wait_window.tkraise()
        print("DEBUG: Wait window should now be visible")
    

    
    def close_wait_window(self):
        """Close the wait window."""
        print("DEBUG: Attempting to close wait window")
        if hasattr(self, 'wait_window') and self.wait_window:
            try:
                self.wait_window.grab_release()
                self.wait_window.destroy()
                print("DEBUG: Wait window destroyed successfully")
            except Exception as e:
                print(f"DEBUG: Error closing wait window: {e}")
        else:
            print("DEBUG: No wait window to close")
    

    
    def run_complete_workflow(self):
        """Run the complete workflow using existing system functionality."""
        # Start workflow immediately without confirmation
        print(f"Starting gaze tracking workflow for {self.user_name}")
        print(f"Data will be saved in: user_data/{self.user_name}/")
        
        # Keep main window but hide it during workflow
        self.root.withdraw()  # Hide completely instead of minimize
        
        try:
            # Step 1: Calibration using existing system
            self.update_status("Step 1/3: Running calibration...", "blue")
            print("\\n" + "="*60)
            print("STARTING CALIBRATION")
            print("="*60)
            
            calibrator = GazeCalibrator(output_dir=self.user_calibration_dir)
            self.calibration_file = calibrator.run_full_calibration()
            
            if not self.calibration_file:
                messagebox.showerror("Error", "Calibration failed or was cancelled.")
                return
            
            # Step 2: Model Training with wait window
            self.update_status("Step 2/3: Training model...", "blue")
            print("\\n" + "="*60)
            print("STARTING MODEL TRAINING")
            print("="*60)
            
            # Show simple wait window
            print("DEBUG: About to show wait window")
            try:
                self.show_wait_window()
                print("DEBUG: Wait window created, forcing display")
                
                # Give the window time to fully render
                import time
                time.sleep(0.5)  # Half second delay to ensure window is visible
                
                print("DEBUG: Starting training")
                # Train model directly
                self.model_path = train_model_from_file(self.calibration_file)
                print("DEBUG: Training completed")
                
            except Exception as e:
                print(f"Error during training: {e}")
                self.model_path = None
            finally:
                # Close wait window
                print("DEBUG: Closing wait window")
                self.close_wait_window()
            
            if not self.model_path:
                messagebox.showerror("Error", "Model training failed.")
                return
            
            # Move model file to user directory
            if self.model_path:
                import shutil
                filename = os.path.basename(self.model_path)
                user_model_file = os.path.join(self.user_models_dir, filename)
                shutil.move(self.model_path, user_model_file)
                self.model_path = user_model_file
            
            # Step 3: Text Reading Analysis using existing system
            self.update_status("Step 3/3: Starting text analysis...", "blue")
            print("\\n" + "="*60)
            print("STARTING TEXT READING ANALYSIS")
            print("="*60)
            print(f"Calibration completed: {os.path.basename(self.calibration_file)}")
            print(f"Model trained: {os.path.basename(self.model_path)}")
            print("\\nStarting text reading analysis system...")
            
            # Run text reading analysis using existing system
            # Create user eye_tracking_data directory
            user_eye_data_dir = os.path.join(self.user_dir, "eye_tracking_data")
            os.makedirs(user_eye_data_dir, exist_ok=True)
            
            predictor = TextReadingGazePredictor(self.model_path, output_dir=user_eye_data_dir)
            predictor.run_text_reading_analysis()
            
            # System Complete
            print("\\n" + "="*60)
            print("SYSTEM COMPLETE!")
            print("="*60)
            print("Your gaze tracking system is fully operational!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Workflow error: {str(e)}")
            print(f"Error during workflow: {e}")
        
        finally:
            # Close application after completion
            self.root.quit()
            self.root.destroy()
    
    def run(self):
        """Start the application."""
        self.root.mainloop()

def main():
    """Main function."""
    print("Starting Gaze Tracking System GUI...")
    
    try:
        app = GazeTrackingSystemGUI()
        app.run()
    except Exception as e:
        print(f"Application error: {str(e)}")

if __name__ == "__main__":
    main()