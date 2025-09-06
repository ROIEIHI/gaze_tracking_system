#!/usr/bin/env python3
"""
Complete Gaze Tracking System with Custom Euclidean Distance Scorer
==================================================================

This script implements a complete gaze tracking system using:
- Custom Euclidean distance scorer for model evaluation
- Multi-output RandomForestRegressor with GridSearchCV
- MediaPipe for facial landmark detection
- Professional reporting without emojis

Author: Refactored Gaze Tracking System
Date: August 31, 2025
"""

import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import time
import os
import datetime
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, r2_score, mean_squared_error
import math


def euclidean_distance_scorer(y_true, y_pred):
    """
    Custom scorer for Euclidean distance between true and predicted coordinates.
    
    Args:
        y_true: True (x, y) coordinates, shape (n_samples, 2)
        y_pred: Predicted (x, y) coordinates, shape (n_samples, 2)
    
    Returns:
        float: Negative mean Euclidean distance (for maximization in GridSearchCV)
    """
    distances = np.sqrt(np.sum((y_true - y_pred) ** 2, axis=1))
    return -np.mean(distances)


# Create scorer object for use with GridSearchCV
euclidean_scorer = make_scorer(euclidean_distance_scorer, greater_is_better=True)


class FacialLandmarkExtractor:
    """Handles facial landmark detection and feature engineering."""
    
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
    def extract_features(self, frame):
        """
        Extract facial landmarks and engineered features from a frame.
        
        Args:
            frame: OpenCV image frame
            
        Returns:
            dict: Dictionary containing extracted features or None if no face detected
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return None
            
        landmarks = results.multi_face_landmarks[0]
        h, w = frame.shape[:2]
        
        # Convert normalized landmarks to pixel coordinates
        points = []
        for lm in landmarks.landmark:
            x = int(lm.x * w)
            y = int(lm.y * h)
            points.append([x, y])
        points = np.array(points)
        
        # Extract key facial points
        nose_tip = points[1]  # Nose tip
        left_eye_center = points[33]  # Left eye inner corner
        right_eye_center = points[362]  # Right eye inner corner
        left_mouth = points[61]  # Left mouth corner
        right_mouth = points[291]  # Right mouth corner
        chin = points[17]  # Chin point
        forehead = points[10]  # Forehead center
        
        # Calculate engineered features
        features = {}
        
        # 1. Eye center coordinates (normalized)
        eye_center_x = (left_eye_center[0] + right_eye_center[0]) / 2 / w
        eye_center_y = (left_eye_center[1] + right_eye_center[1]) / 2 / h
        features['eye_center_x'] = eye_center_x
        features['eye_center_y'] = eye_center_y
        
        # 2. Head pose estimation using key points
        # Yaw: Left-right head rotation
        nose_x_normalized = nose_tip[0] / w
        face_center_x = (left_eye_center[0] + right_eye_center[0]) / 2 / w
        yaw = nose_x_normalized - face_center_x
        features['head_yaw'] = yaw
        
        # Pitch: Up-down head rotation
        nose_y_normalized = nose_tip[1] / h
        eye_y_normalized = (left_eye_center[1] + right_eye_center[1]) / 2 / h
        pitch = nose_y_normalized - eye_y_normalized
        features['head_pitch'] = pitch
        
        # 3. Face size (distance between eyes, normalized)
        eye_distance = np.linalg.norm(left_eye_center - right_eye_center) / w
        features['face_size'] = eye_distance
        
        # 4. Nose position relative to face center
        face_center = (left_eye_center + right_eye_center) / 2
        nose_offset_x = (nose_tip[0] - face_center[0]) / w
        nose_offset_y = (nose_tip[1] - face_center[1]) / h
        features['nose_offset_x'] = nose_offset_x
        features['nose_offset_y'] = nose_offset_y
        
        return features


class GazeModelTrainer:
    """Handles model training with custom Euclidean distance scorer."""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_extractor = FacialLandmarkExtractor()
        self.training_history = {}
        
    def prepare_data_from_csv(self, csv_file):
        """
        Load and prepare training data from CSV file.
        
        Args:
            csv_file: Path to calibration CSV file
            
        Returns:
            tuple: (features_array, targets_array) or (None, None) if failed
        """
        try:
            df = pd.read_csv(csv_file)
            print(f"Loaded {len(df)} samples from {csv_file}")
            
            features_list = []
            targets_list = []
            
            for _, row in df.iterrows():
                # Parse features (assuming they're stored as strings)
                try:
                    features = eval(row['features'])  # Convert string to dict
                    target_x = float(row['target_x'])
                    target_y = float(row['target_y'])
                    
                    # Convert features dict to array
                    feature_array = [
                        features['eye_center_x'],
                        features['eye_center_y'],
                        features['head_yaw'],
                        features['head_pitch'],
                        features['face_size'],
                        features['nose_offset_x'],
                        features['nose_offset_y']
                    ]
                    
                    features_list.append(feature_array)
                    targets_list.append([target_x, target_y])
                    
                except (ValueError, KeyError) as e:
                    print(f"Skipping invalid row: {e}")
                    continue
            
            if len(features_list) == 0:
                print("No valid data found in CSV file")
                return None, None
                
            X = np.array(features_list)
            y = np.array(targets_list)
            
            print(f"Prepared {len(X)} valid samples")
            print(f"Features shape: {X.shape}")
            print(f"Targets shape: {y.shape}")
            
            return X, y
            
        except Exception as e:
            print(f"Error preparing data: {e}")
            return None, None
    
    def train_model(self, X, y, noise_level=0):
        """
        Train RandomForestRegressor with custom Euclidean distance scorer.
        
        Args:
            X: Feature matrix, shape (n_samples, n_features)
            y: Target matrix, shape (n_samples, 2) for (x, y) coordinates
            noise_level: Data augmentation noise level (pixels)
            
        Returns:
            RandomForestRegressor: Trained model or None if failed
        """
        print("Starting model training with RandomForestRegressor...")
        print(f"Feature matrix shape: {X.shape}")
        print(f"Target matrix shape: {y.shape}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"Training set size: {len(X_train)}")
        print(f"Test set size: {len(X_test)}")
        
        # Apply data augmentation to training set
        if noise_level > 0:
            print(f"Applying data augmentation to training targets...")
            noise = np.random.normal(0, noise_level, y_train.shape)
            y_train_augmented = y_train + noise
        else:
            y_train_augmented = y_train
        
        # Scale features
        print("Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Define parameter grid for GridSearchCV
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }
        
        print("Performing GridSearchCV with custom Euclidean distance scorer...")
        
        # Create RandomForestRegressor
        rf = RandomForestRegressor(random_state=42, n_jobs=-1)
        
        # Perform grid search with custom scorer
        grid_search = GridSearchCV(
            rf,
            param_grid,
            scoring=euclidean_scorer,
            cv=3,
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train_scaled, y_train_augmented)
        
        # Get best model
        self.model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        
        print(f"Best model parameters: {best_params}")
        print(f"Best CV score (negative mean Euclidean distance): {best_score:.4f}")
        
        # Evaluate on test set
        print("Evaluating model on clean test set...")
        y_pred = self.model.predict(X_test_scaled)
        
        # Calculate Euclidean distance error
        euclidean_distances = np.sqrt(np.sum((y_test - y_pred) ** 2, axis=1))
        mean_euclidean_error = np.mean(euclidean_distances)
        
        # Calculate R² scores
        r2_x = r2_score(y_test[:, 0], y_pred[:, 0])
        r2_y = r2_score(y_test[:, 1], y_pred[:, 1])
        avg_r2_score = (r2_x + r2_y) / 2
        
        # Calculate RMSE for each coordinate
        rmse_x = np.sqrt(mean_squared_error(y_test[:, 0], y_pred[:, 0]))
        rmse_y = np.sqrt(mean_squared_error(y_test[:, 1], y_pred[:, 1]))
        
        # Cross-validation with Euclidean scorer
        print("Performing cross-validation with Euclidean distance scorer...")
        cv_scores = cross_val_score(
            self.model, X_train_scaled, y_train, 
            scoring=euclidean_scorer, cv=5
        )
        
        # Store training history
        self.training_history = {
            'mean_euclidean_error': mean_euclidean_error,
            'avg_r2_score': avg_r2_score,
            'r2_x': r2_x,
            'r2_y': r2_y,
            'rmse_x': rmse_x,
            'rmse_y': rmse_y,
            'cv_euclidean_mean': np.mean(cv_scores),
            'cv_euclidean_std': np.std(cv_scores),
            'best_params': best_params,
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'noise_level': noise_level
        }
        
        # Print performance report
        self._print_performance_report()
        
        return self.model
    
    def _print_performance_report(self):
        """Print detailed performance report."""
        # Calculate percentage error based on fixed resolution 1080x720
        screen_width = 1080
        screen_height = 720
        screen_diagonal = np.sqrt(screen_width**2 + screen_height**2)
        percentage_error = (self.training_history['mean_euclidean_error'] / screen_diagonal) * 100
        
        print("\n" + "=" * 80)
        print("RANDOM FOREST REGRESSOR MODEL PERFORMANCE REPORT")
        print("=" * 80)
        print(f"EUCLIDEAN DISTANCE ERROR: {self.training_history['mean_euclidean_error']:.2f} pixels")
        print(f"PERCENTAGE ERROR: {percentage_error:.2f}% (relative to {screen_width}x{screen_height} diagonal)")
        print(f"Average R² Score: {self.training_history['avg_r2_score']:.4f}")
        print(f"X-coordinate R²: {self.training_history['r2_x']:.4f}")
        print(f"Y-coordinate R²: {self.training_history['r2_y']:.4f}")
        print(f"X-coordinate RMSE: {self.training_history['rmse_x']:.2f} pixels")
        print(f"Y-coordinate RMSE: {self.training_history['rmse_y']:.2f} pixels")
        print(f"Cross-validation Score (Euclidean): {self.training_history['cv_euclidean_mean']:.4f} ± {self.training_history['cv_euclidean_std']:.4f}")
        print(f"Training samples: {self.training_history['training_samples']}")
        print(f"Test samples: {self.training_history['test_samples']}")
        print(f"Noise augmentation level: {self.training_history['noise_level']} pixels")
        print(f"Best parameters: {self.training_history['best_params']}")
        print("=" * 80)
    
    def save_model(self, custom_filename=None):
        """Save the trained model and associated data."""
        if self.model is None:
            print("No trained model to save.")
            return None
        
        # Create models directory if it doesn't exist
        models_dir = "models"
        os.makedirs(models_dir, exist_ok=True)
        
        # Use fixed filename to overwrite previous models
        model_filename = "gaze_prediction_model.joblib"
        model_path = os.path.join(models_dir, model_filename)
        
        # Prepare model data
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'training_history': self.training_history,
            'model_type': 'RandomForestRegressor'
        }
        
        # Save model
        joblib.dump(model_data, model_path)
        print(f"Model saved to: {model_path}")
        
        return model_path
    
    def train_from_csv(self, csv_file, noise_level=5):
        """Complete training pipeline from CSV file."""
        try:
            X, y = self.prepare_data_from_csv(csv_file)
            if X is None or y is None:
                return None
            
            model = self.train_model(X, y, noise_level)
            if model is not None:
                model_path = self.save_model()
                return model_path
            else:
                return None
                
        except Exception as e:
            print(f"Training failed: {e}")
            return None


class GazePredictor:
    """Handles real-time gaze prediction."""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_extractor = FacialLandmarkExtractor()
        
    def load_model(self, model_path):
        """Load a trained model."""
        try:
            model_data = joblib.load(model_path)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            print(f"Model loaded successfully: {model_path}")
            return True
        except Exception as e:
            print(f"Failed to load model: {e}")
            return False
    
    def predict_gaze_point(self, frame):
        """Predict gaze point from a single frame."""
        if self.model is None:
            return None
        
        features = self.feature_extractor.extract_features(frame)
        if features is None:
            return None
        
        # Convert features to array
        feature_array = np.array([[
            features['eye_center_x'],
            features['eye_center_y'],
            features['head_yaw'],
            features['head_pitch'],
            features['face_size'],
            features['nose_offset_x'],
            features['nose_offset_y']
        ]])
        
        # Scale features
        feature_array_scaled = self.scaler.transform(feature_array)
        
        # Predict
        prediction = self.model.predict(feature_array_scaled)
        
        return prediction[0]  # Return (x, y) coordinates
    
    def run_real_time_prediction(self):
        """Run real-time gaze prediction."""
        if self.model is None:
            print("No model loaded. Please load a model first.")
            return
        
        print("Starting real-time gaze prediction...")
        print("Press 'q' to quit")
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Failed to initialize camera")
            return
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Predict gaze point
                gaze_point = self.predict_gaze_point(frame)
                
                # Draw prediction on frame
                if gaze_point is not None:
                    x, y = gaze_point
                    cv2.circle(frame, (int(x), int(y)), 10, (0, 255, 0), -1)
                    cv2.putText(frame, f"Gaze: ({int(x)}, {int(y)})", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "No face detected", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                cv2.imshow('Gaze Tracking', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        finally:
            cap.release()
            cv2.destroyAllWindows()


class CalibrationSystem:
    """Handles calibration data collection."""
    
    def __init__(self):
        self.feature_extractor = FacialLandmarkExtractor()
        
    def run_calibration(self, screen_width=1080, screen_height=720, points_per_target=3):
        """Run calibration data collection."""
        print("Starting calibration...")
        
        # Define calibration points (grid)
        calibration_points = [
            (screen_width * 0.1, screen_height * 0.1),  # Top-left
            (screen_width * 0.5, screen_height * 0.1),  # Top-center
            (screen_width * 0.9, screen_height * 0.1),  # Top-right
            (screen_width * 0.1, screen_height * 0.5),  # Middle-left
            (screen_width * 0.5, screen_height * 0.5),  # Center
            (screen_width * 0.9, screen_height * 0.5),  # Middle-right
            (screen_width * 0.1, screen_height * 0.9),  # Bottom-left
            (screen_width * 0.5, screen_height * 0.9),  # Bottom-center
            (screen_width * 0.9, screen_height * 0.9),  # Bottom-right
        ]
        
        calibration_data = []
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Failed to initialize camera")
            return None
        
        try:
            for i, (target_x, target_y) in enumerate(calibration_points):
                print(f"Calibrating point {i+1}/9: ({target_x:.0f}, {target_y:.0f})")
                
                # Create calibration display
                calib_frame = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
                cv2.circle(calib_frame, (int(target_x), int(target_y)), 20, (0, 0, 255), -1)
                cv2.putText(calib_frame, f"Look at the red dot ({i+1}/9)", 
                          (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(calib_frame, "Press SPACE when ready", 
                          (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                cv2.imshow('Calibration', calib_frame)
                cv2.waitKey(1)
                
                collected_points = 0
                while collected_points < points_per_target:
                    ret, frame = cap.read()
                    if not ret:
                        continue
                    
                    cv2.imshow('Camera', frame)
                    key = cv2.waitKey(1) & 0xFF
                    
                    if key == ord(' '):  # Space to capture
                        features = self.feature_extractor.extract_features(frame)
                        if features is not None:
                            calibration_data.append({
                                'target_x': target_x,
                                'target_y': target_y,
                                'features': features
                            })
                            collected_points += 1
                            print(f"Collected {collected_points}/{points_per_target} points for target {i+1}")
                        else:
                            print("No face detected, try again")
                    
                    elif key == ord('q'):
                        break
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
        
        if len(calibration_data) == 0:
            print("No calibration data collected")
            return None
        
        # Save calibration data
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"calibration_data_{timestamp}.csv"
        
        df = pd.DataFrame(calibration_data)
        df.to_csv(csv_filename, index=False)
        
        print(f"Calibration completed! Data saved to: {csv_filename}")
        print(f"Collected {len(calibration_data)} calibration points")
        
        return csv_filename


class GazeTrackingGUI:
    """Main GUI application."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Gaze Tracking System - RandomForest with Custom Scorer")
        self.root.geometry("600x500")
        
        self.calibration_system = CalibrationSystem()
        self.trainer = GazeModelTrainer()
        self.predictor = GazePredictor()
        
        self.setup_gui()
    
    def setup_gui(self):
        """Setup the GUI interface."""
        # Title
        title_label = tk.Label(self.root, 
                              text="Gaze Tracking System", 
                              font=("Arial", 16, "bold"))
        title_label.pack(pady=10)
        
        # Status display
        self.status_text = tk.Text(self.root, height=8, width=70)
        self.status_text.pack(pady=10)
        
        # Buttons frame
        buttons_frame = tk.Frame(self.root)
        buttons_frame.pack(pady=20)
        
        # Calibration button
        calib_btn = tk.Button(buttons_frame, 
                             text="Run Calibration", 
                             command=self.run_calibration,
                             width=20, height=2)
        calib_btn.grid(row=0, column=0, padx=10, pady=5)
        
        # Training button
        train_btn = tk.Button(buttons_frame, 
                             text="Train Model", 
                             command=self.train_model,
                             width=20, height=2)
        train_btn.grid(row=0, column=1, padx=10, pady=5)
        
        # Prediction button
        pred_btn = tk.Button(buttons_frame, 
                            text="Real-Time Prediction", 
                            command=self.run_prediction,
                            width=20, height=2)
        pred_btn.grid(row=1, column=0, padx=10, pady=5)
        
        # Complete workflow button
        complete_btn = tk.Button(buttons_frame, 
                                text="Complete Workflow", 
                                command=self.run_complete_workflow,
                                width=20, height=2)
        complete_btn.grid(row=1, column=1, padx=10, pady=5)
        
        self.update_status("Gaze Tracking System Ready", "green")
    
    def update_status(self, message, color="black"):
        """Update status display."""
        self.status_text.insert(tk.END, f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {message}\n")
        self.status_text.see(tk.END)
        self.root.update()
    
    def run_calibration(self):
        """Run calibration in separate thread."""
        def calibration_thread():
            try:
                self.update_status("Starting calibration...", "blue")
                csv_file = self.calibration_system.run_calibration()
                if csv_file:
                    self.update_status(f"Calibration completed: {csv_file}", "green")
                else:
                    self.update_status("Calibration failed", "red")
            except Exception as e:
                self.update_status(f"Calibration error: {str(e)}", "red")
        
        thread = threading.Thread(target=calibration_thread)
        thread.daemon = True
        thread.start()
    
    def train_model(self):
        """Train model using calibration data."""
        def training_thread():
            try:
                # Find calibration data
                script_dir = os.path.dirname(os.path.abspath(__file__))
                calibration_files = [f for f in os.listdir(script_dir) 
                                   if f.startswith('calibration_data_') and f.endswith('.csv')]
                
                if not calibration_files:
                    # Fallback: open file browser
                    csv_file = filedialog.askopenfilename(
                        title="Select calibration data",
                        filetypes=[("CSV files", "*.csv")]
                    )
                    if not csv_file:
                        self.update_status("No calibration data selected", "red")
                        return
                else:
                    # Use most recent calibration file
                    csv_file = os.path.join(script_dir, max(calibration_files))
                
                self.update_status(f"Training model with: {os.path.basename(csv_file)}", "blue")
                
                model_path = self.trainer.train_from_csv(csv_file)
                if model_path:
                    # Get error from training history
                    error = self.trainer.training_history.get('mean_euclidean_error', 0)
                    self.update_status(f"Model trained: {error:.1f}px error", "green")
                    self.update_status(f"Model saved: {os.path.basename(model_path)}", "green")
                    
                    # Display detailed results
                    history = self.trainer.training_history
                    self.update_status(f"R² Score: {history['avg_r2_score']:.3f}", "blue")
                    self.update_status(f"CV Score: {history['cv_euclidean_mean']:.3f}", "blue")
                else:
                    self.update_status("Training failed", "red")
                    
            except Exception as e:
                self.update_status(f"Training error: {str(e)}", "red")
        
        thread = threading.Thread(target=training_thread)
        thread.daemon = True
        thread.start()
    
    def run_prediction(self):
        """Run real-time prediction."""
        def prediction_thread():
            try:
                # Find trained models
                models_dir = "models"
                if not os.path.exists(models_dir):
                    self.update_status("No models directory found", "red")
                    return
                
                model_files = [f for f in os.listdir(models_dir) if f.endswith('.joblib')]
                
                if not model_files:
                    # Fallback: open file browser
                    model_path = filedialog.askopenfilename(
                        title="Select trained model",
                        filetypes=[("Joblib files", "*.joblib")]
                    )
                    if not model_path:
                        self.update_status("No model selected", "red")
                        return
                else:
                    # Use most recent model
                    model_path = os.path.join(models_dir, max(model_files))
                
                self.update_status(f"Loading model: {os.path.basename(model_path)}", "blue")
                
                if self.predictor.load_model(model_path):
                    self.update_status("Running real-time prediction", "green")
                    self.predictor.run_real_time_prediction()
                    self.update_status("Prediction session ended", "blue")
                else:
                    self.update_status("Failed to load model", "red")
                    
            except Exception as e:
                self.update_status(f"Prediction error: {str(e)}", "red")
        
        thread = threading.Thread(target=prediction_thread)
        thread.daemon = True
        thread.start()
    
    def run_complete_workflow(self):
        """Run complete calibration -> training -> prediction workflow."""
        def workflow_thread():
            try:
                # Step 1: Calibration
                self.update_status("Step 1/3: Running calibration...", "blue")
                csv_file = self.calibration_system.run_calibration()
                if not csv_file:
                    self.update_status("Workflow failed at calibration", "red")
                    return
                
                self.update_status(f"Calibration completed: {csv_file}", "green")
                
                # Step 2: Training
                self.update_status("Step 2/3: Training model...", "blue")
                model_path = self.trainer.train_from_csv(csv_file)
                if not model_path:
                    self.update_status("Workflow failed at training", "red")
                    return
                
                error = self.trainer.training_history.get('mean_euclidean_error', 0)
                self.update_status(f"Model trained: {error:.1f}px error", "green")
                
                # Step 3: Prediction
                self.update_status("Step 3/3: Loading model for prediction...", "blue")
                if self.predictor.load_model(model_path):
                    self.update_status("Complete workflow finished!\n"
                                     "Starting real-time prediction...", "green")
                    self.predictor.run_real_time_prediction()
                else:
                    self.update_status("Failed to load trained model", "red")
                    
            except Exception as e:
                self.update_status(f"Workflow error: {str(e)}", "red")
        
        thread = threading.Thread(target=workflow_thread)
        thread.daemon = True
        thread.start()
    
    def run(self):
        """Start the GUI application."""
        self.root.mainloop()


def main():
    """Main function to run the application."""
    print("Starting Gaze Tracking System with RandomForest and Custom Euclidean Scorer...")
    
    try:
        # Create and run GUI
        app = GazeTrackingGUI()
        app.run()
        
    except Exception as e:
        print(f"Application error: {str(e)}")
        

def test_euclidean_scorer():
    """Test the custom Euclidean distance scorer."""
    print("Testing custom Euclidean distance scorer...")
    
    # Create test data
    y_true = np.array([[100, 200], [300, 400], [500, 600]])
    y_pred = np.array([[110, 190], [290, 410], [495, 605]])
    
    # Test the scorer
    score = euclidean_distance_scorer(y_true, y_pred)
    
    # Calculate expected result manually
    distances = np.sqrt(np.sum((y_true - y_pred) ** 2, axis=1))
    expected_score = -np.mean(distances)
    
    print(f"Scorer result: {score:.4f}")
    print(f"Expected result: {expected_score:.4f}")
    print(f"Test passed: {abs(score - expected_score) < 1e-10}")
    

if __name__ == "__main__":
    # Uncomment to test the scorer
    # test_euclidean_scorer()
    
    # Run main application
    main()
