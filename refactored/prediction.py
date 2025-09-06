"""
Streamlined Gaze Prediction with Multi-Output RandomForest Model
Optimized for real-time performance
"""

import cv2
import numpy as np
import joblib
import os
from calibration import EyeTrackerCalibrator

# Optional import for eye movement analysis
try:
    from eye_movement_analyzer import EyeMovementAnalyzer
    MOVEMENT_ANALYZER_AVAILABLE = True
except ImportError:
    MOVEMENT_ANALYZER_AVAILABLE = False
    print("Eye movement analyzer not available (optional feature)")

class GazePredictor:
    """Streamlined gaze prediction using multi-output XGBoost model."""
    
    def __init__(self, model_path=None):
        """Initialize the gaze predictor."""
        # Initialize the calibrator for feature extraction
        self.calibrator = EyeTrackerCalibrator()
        
        # Prediction-specific attributes
        self.model = None
        self.scaler = None
        self.model_path = model_path
        self.smoothed_x = None
        self.smoothed_y = None
        
        # Initialize eye movement analyzer (optional)
        self.movement_analyzer = None
        self.analysis_enabled = False
        
        # Load model if path provided
        if model_path:
            self.load_model(model_path)
    
    def load_default_model(self):
        """Load the default model with fixed filename."""
        # Try multiple possible locations for the model
        possible_paths = [
            "models/gaze_prediction_model.joblib",  # Local models directory
            "../models/gaze_prediction_model.joblib",  # Parent directory models
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "gaze_prediction_model.joblib")  # Absolute path to parent models
        ]
        
        for model_path in possible_paths:
            if os.path.exists(model_path):
                print(f"Found model at: {model_path}")
                return self.load_model(model_path)
        
        print("Default model not found in any of these locations:")
        for path in possible_paths:
            print(f"  - {os.path.abspath(path)}")
        print("Please train a model first.")
        return False
    
    def load_model(self, model_path):
        """Load a trained multi-output model from file."""
        try:
            model_data = joblib.load(model_path)
            
            # Validate model format - accept both XGBoost and any model type
            model_type = model_data.get('model_type', 'unknown')
            print(f"Loading model type: {model_type}")
            
            # Extract model components
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            
            print(f"Model loaded successfully from: {model_path}")
            
            # Print model performance info if available
            training_history = model_data.get('training_history', {})
            if training_history:
                error = training_history.get('mean_euclidean_error', 'N/A')
                r2 = training_history.get('avg_r2_score', 'N/A')
                if isinstance(error, float):
                    print(f"Model test error: {error:.2f} pixels")
                if isinstance(r2, float):
                    print(f"Model R² score: {r2:.4f}")
            
            self.model_path = model_path
            return True
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
    
    def enable_movement_analysis(self, window_width=None, window_height=None):
        """Enable eye movement analysis during prediction."""
        if not MOVEMENT_ANALYZER_AVAILABLE:
            print("Eye movement analysis not available")
            return
            
        if window_width is None:
            window_width = self.calibrator.WINDOW_WIDTH
        if window_height is None:
            window_height = self.calibrator.WINDOW_HEIGHT
            
        self.movement_analyzer = EyeMovementAnalyzer(window_width, window_height)
        self.analysis_enabled = True
        print("Eye movement analysis enabled")
    
    def disable_movement_analysis(self):
        """Disable eye movement analysis."""
        self.analysis_enabled = False
        self.movement_analyzer = None
        print("Eye movement analysis disabled")
    
    def start_analysis_session(self):
        """Start a new movement analysis session."""
        if self.movement_analyzer:
            self.movement_analyzer.start_reading_session()
            print("Started eye movement analysis session")
    
    def finish_analysis_session(self, auto_export=True):
        """Finish movement analysis session and export data."""
        if self.movement_analyzer:
            csv_file = self.movement_analyzer.finish_analysis_session(auto_export)
            return csv_file
        return None
    
    def predict_gaze_point(self, features):
        """
        Predict gaze coordinates from extracted features using multi-output XGBoost model.
        
        Args:
            features: List of features [norm_x_L, norm_y_L, norm_x_R, norm_y_R, yaw, pitch, roll]
            
        Returns:
            Tuple of (x, y) coordinates or None if prediction fails
        """
        if self.model is None or self.scaler is None:
            print("No model loaded. Cannot make predictions.")
            return None
        
        try:
            # Engineer features from raw input
            if len(features) >= 7:
                # Raw features: [norm_x_L, norm_y_L, norm_x_R, norm_y_R, yaw, pitch, roll]
                avg_norm_x = (features[0] + features[2]) / 2
                avg_norm_y = (features[1] + features[3]) / 2
                yaw, pitch, roll = features[4], features[5], features[6]
                
                # Create engineered features
                engineered_features = np.array([[
                    avg_norm_x, avg_norm_y, yaw, pitch, roll,
                    avg_norm_x * yaw,      # x_yaw_interaction
                    avg_norm_y * pitch     # y_pitch_interaction
                ]])
            else:
                # Assume features are already engineered
                engineered_features = np.array([features])
            
            # Scale features
            scaled_features = self.scaler.transform(engineered_features)
            
            # Make prediction using multi-output model
            prediction = self.model.predict(scaled_features)[0]
            pred_x, pred_y = int(prediction[0]), int(prediction[1])
            
            return pred_x, pred_y
            
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            return None
    
    def smooth_prediction(self, x, y, alpha=0.3):
        """
        Apply exponential smoothing to predictions for stability.
        
        Args:
            x, y: Current prediction coordinates
            alpha: Smoothing factor (0-1, lower = more smoothing)
            
        Returns:
            Tuple of smoothed (x, y) coordinates
        """
        if self.smoothed_x is None or self.smoothed_y is None:
            self.smoothed_x = x
            self.smoothed_y = y
        else:
            self.smoothed_x = alpha * x + (1 - alpha) * self.smoothed_x
            self.smoothed_y = alpha * y + (1 - alpha) * self.smoothed_y
        
        return int(self.smoothed_x), int(self.smoothed_y)
    
    def run_real_time_prediction(self, enable_smoothing=True, smoothing_alpha=0.3):
        """
        Run real-time gaze prediction loop.
        
        Args:
            enable_smoothing: Whether to apply smoothing to predictions
            smoothing_alpha: Smoothing factor for exponential smoothing
        """
        if self.model is None:
            print("No model loaded. Attempting to load default model...")
            if not self.load_default_model():
                print("No model loaded. Please load a model first.")
                return
        
        print("Starting real-time gaze prediction...")
        print("Controls:")
        print("  - ESC: Exit")
        print("  - SPACE: Toggle movement analysis")
        print("  - 's': Start/stop analysis session")
        print("  - 'r': Reset smoothing")
        
        # Initialize camera directly
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Failed to initialize camera")
            return
        
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("Camera initialized successfully")
        print("Press ESC to exit prediction...")
        
        try:
            while True:
                # Capture frame
                ret, frame = cap.read()
                if not ret:
                    print("Failed to read from camera")
                    break
                
                # Convert frame to RGB and process with MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                landmarks = self.calibrator.face_mesh.process(rgb_frame)
                
                # Extract features using calibrator's feature extraction
                result = self.calibrator.extract_iris_features(frame, landmarks)
                
                if result:
                    features, visualization_data = result
                
                if features:
                    # Make prediction
                    prediction = self.predict_gaze_point(features)
                    
                    if prediction:
                        pred_x, pred_y = prediction
                        
                        # Apply smoothing if enabled
                        if enable_smoothing:
                            pred_x, pred_y = self.smooth_prediction(pred_x, pred_y, smoothing_alpha)
                        
                        # Record movement if analysis is enabled
                        if self.analysis_enabled and self.movement_analyzer:
                            self.movement_analyzer.record_gaze_point(pred_x, pred_y)
                        
                        # Draw prediction on frame
                        cv2.circle(frame, (pred_x, pred_y), 10, (0, 255, 0), -1)
                        cv2.putText(frame, f"Gaze: ({pred_x}, {pred_y})", 
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    else:
                        cv2.putText(frame, "No prediction", 
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    cv2.putText(frame, "No face detected", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Add status information
                status_y = 60
                if self.analysis_enabled:
                    cv2.putText(frame, "Analysis: ON", 
                               (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    status_y += 25
                
                if enable_smoothing:
                    cv2.putText(frame, f"Smoothing: {smoothing_alpha:.1f}", 
                               (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                # Display frame
                cv2.imshow('Real-time Gaze Prediction', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    break
                elif key == ord(' '):  # Space - toggle analysis
                    if self.analysis_enabled:
                        self.disable_movement_analysis()
                    else:
                        self.enable_movement_analysis()
                elif key == ord('s'):  # S - start/stop session
                    if self.movement_analyzer:
                        if hasattr(self.movement_analyzer, 'session_active') and self.movement_analyzer.session_active:
                            csv_file = self.finish_analysis_session()
                            if csv_file:
                                print(f"📁 Session data exported to: {csv_file}")
                        else:
                            self.start_analysis_session()
                elif key == ord('r'):  # R - reset smoothing
                    self.smoothed_x = None
                    self.smoothed_y = None
                    print("Smoothing reset")
        
        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            
            # Finish any active analysis session
            if self.analysis_enabled and self.movement_analyzer:
                csv_file = self.finish_analysis_session()
                if csv_file:
                    print(f"Final session data exported to: {csv_file}")
            
            print("Real-time prediction ended")

def main():
    """Main function for standalone prediction."""
    # Initialize predictor and try to load default model
    predictor = GazePredictor()
    
    if predictor.load_default_model():
        print("Default model loaded successfully")
        predictor.run_real_time_prediction()
    else:
        print("Failed to load default model")
        print("Please train a model first using the training module")
        return
    
    if predictor.model is not None:
        # Run real-time prediction
        predictor.run_real_time_prediction(enable_smoothing=True, smoothing_alpha=0.3)
    else:
        print("Failed to load model")

if __name__ == "__main__":
    main()
