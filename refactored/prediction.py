import cv2
import mediapipe as mp
import numpy as np
import joblib
import os
from calibration import EyeTrackerCalibrator

class GazePredictor:
    def __init__(self, model_path=None):
        # Initialize the calibrator for feature extraction
        self.calibrator = EyeTrackerCalibrator()
        
        # Prediction-specific attributes
        self.model = None
        self.model_path = model_path
        self.smoothed_x = None
        self.smoothed_y = None
        
        # Load model if path provided
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path):
        """Load a trained model from file"""
        try:
            model_data = joblib.load(model_path)
            self.model = model_data['model']
            self.model_path = model_path
            print(f"Model loaded from: {model_path}")
            
            # Print model info if available
            training_history = model_data.get('training_history', {})
            if training_history:
                test_error = training_history.get('test_mean_error', 'N/A')
                print(f"Model test error: {test_error:.2f} pixels" if isinstance(test_error, float) else f"Model test error: {test_error}")
            
            return True
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
    
    def predict_gaze_point(self, features):
        """Predict gaze coordinates from extracted features"""
        if self.model is None:
            print("No model loaded. Cannot make predictions.")
            return None
        
        try:
            # Use only the first 7 features (normalized iris positions + head pose)
            prediction = self.model.predict([features[:7]])[0]
            return int(prediction[0]), int(prediction[1])
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            return None
    
    def apply_smoothing(self, pred_x, pred_y):
        """Apply exponential smoothing to predictions"""
        if self.smoothed_x is None:
            self.smoothed_x, self.smoothed_y = pred_x, pred_y
        else:
            # Exponential moving average
            self.smoothed_x = self.smoothed_x * (1 - self.calibrator.SMOOTHING_FACTOR) + pred_x * self.calibrator.SMOOTHING_FACTOR
            self.smoothed_y = self.smoothed_y * (1 - self.calibrator.SMOOTHING_FACTOR) + pred_y * self.calibrator.SMOOTHING_FACTOR
        
        # Convert to integers and ensure coordinates are within bounds
        smoothed_x_int = int(self.smoothed_x)
        smoothed_y_int = int(self.smoothed_y)
        smoothed_x_int = max(0, min(self.calibrator.WINDOW_WIDTH - 1, smoothed_x_int))
        smoothed_y_int = max(0, min(self.calibrator.WINDOW_HEIGHT - 1, smoothed_y_int))
        
        return smoothed_x_int, smoothed_y_int
    
    def draw_head_pose_axis(self, image, rvec, tvec, cam_matrix):
        """Draw 3D axis on the nose to visualize head pose"""
        if rvec is None or tvec is None:
            return
        
        # Define 3D axis points (length in mm)
        axis_length = 100
        axis_points = np.array([
            (0, 0, 0),                    # Origin (nose tip)
            (axis_length, 0, 0),          # X-axis (red) - right
            (0, axis_length, 0),          # Y-axis (green) - down
            (0, 0, -axis_length)          # Z-axis (blue) - forward
        ], dtype=np.float32)
        
        # Project 3D points to 2D image plane
        projected_points, _ = cv2.projectPoints(
            axis_points, rvec, tvec, cam_matrix, np.zeros((4, 1))
        )
        
        # Convert to integer coordinates
        projected_points = projected_points.reshape(-1, 2).astype(int)
        
        if len(projected_points) == 4:
            origin = tuple(projected_points[0])
            x_axis = tuple(projected_points[1])
            y_axis = tuple(projected_points[2])
            z_axis = tuple(projected_points[3])
            
            # Draw axis lines with different colors
            cv2.line(image, origin, x_axis, (0, 0, 255), 3)  # X-axis: Red
            cv2.line(image, origin, y_axis, (0, 255, 0), 3)  # Y-axis: Green
            cv2.line(image, origin, z_axis, (255, 0, 0), 3)  # Z-axis: Blue
            
            # Add labels
            cv2.putText(image, 'X', x_axis, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(image, 'Y', y_axis, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(image, 'Z', z_axis, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    def create_text_window(self):
        """Create window with reading text for text analysis mode"""
        # Load the generated text image
        if os.path.exists(self.calibrator.text_image_path):
            text_window = cv2.imread(self.calibrator.text_image_path)
            if text_window is not None:
                return text_window
        
        # Fallback: create text window manually
        window = np.ones((self.calibrator.WINDOW_HEIGHT, self.calibrator.WINDOW_WIDTH, 3), dtype=np.uint8) * 255
        
        text = """The benefits of pets

Most pet owners are clear about the immediate joys that come with sharing their lives with companion animals.
However, many of us remain unaware of the physical and mental health benefits that can also accompany the
pleasure of snuggling up to a furry friend. It's only recently that studies have begun to scientifically explore the
benefits of the human-animal bond.

Pets have evolved to become acutely attuned to humans and our behavior and emotions. Dogs, for example, are
able to understand many of the words we use, but they're even better at interpreting our tone of voice, body
language, and gestures. And like any good human friend, a loyal dog will look into your eyes to gauge your
emotional state and try to understand what you're thinking and feeling (and to work out when the next walk or
treat might be coming, of course).

Pets, especially dogs and cats, can reduce stress, anxiety, and depression, ease loneliness, encourage exercise
and playfulness, and even improve your cardiovascular health. Caring for an animal can help children grow up
more secure and active. Pets also provide valuable companionship for older adults. Perhaps most importantly,
though, a pet can add real joy and unconditional love to your life.

Any pet can improve your health

While it's true that people with pets often experience greater health benefits than those without, a pet doesn't
necessarily have to be a dog or a cat. A rabbit could be ideal if you're allergic to other animals or have limited
space but still want a furry friend to snuggle with. Birds can encourage social interaction and help keep your
mind sharp if you're an older adult. Snakes, lizards, and other reptiles can make for exotic companions. Even
watching fish in an aquarium can help reduce muscle tension and lower your pulse rate."""
        
        # Draw text on window
        y = 50
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        line_spacing = 30
        color = (0, 0, 0)  # Black color
        thickness = 1
        
        paragraphs = text.split('\n\n')
        for paragraph in paragraphs:
            lines = paragraph.split('\n')
            for line in lines:
                cv2.putText(window, line, (50, y), font, font_scale, color, thickness)
                y += line_spacing
            y += 20  # Extra space between paragraphs
        
        return window
    
    def real_time_prediction(self, mode="standard"):
        """Real-time gaze prediction with mode selection"""
        if self.model is None:
            print("No model loaded. Cannot run real-time prediction.")
            return
        
        print(f"--- Starting Real-Time Prediction (Mode: {mode}) ---")
        print("Press 'q' to quit")
        
        window_name = 'Gaze Prediction' if mode == "standard" else 'Text Reading Analysis'
        cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        # Setup camera
        try:
            cap = self.calibrator.setup_camera()
        except Exception as e:
            print(f"Error setting up camera: {str(e)}")
            return
        
        # Load text window for text analysis mode
        if mode == "text_analysis":
            text_window = self.create_text_window()
            if text_window is None:
                print("Failed to create text window. Reverting to standard mode.")
                mode = "standard"
                text_window = None
        else:
            text_window = None
        
        # Reset smoothing
        self.smoothed_x = None
        self.smoothed_y = None
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Flip frame based on global flag
                if not self.calibrator.FLIP_FRAME:
                    frame = cv2.flip(frame, 1)
                
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.calibrator.face_mesh.process(rgb_frame)
                
                # Monitor user compliance during prediction
                if not self.calibrator.monitor_user_compliance(cap, results):
                    print("Session aborted during real-time prediction")
                    break
                
                # Create or get the appropriate window
                if mode == "standard":
                    window = np.zeros((self.calibrator.WINDOW_HEIGHT, self.calibrator.WINDOW_WIDTH, 3), dtype=np.uint8)
                else:
                    window = text_window.copy()
                
                # Extract features and predict
                result = self.calibrator.extract_iris_features(frame, results)
                if result and self.model:
                    features, (rotation_vector, translation_vector) = result
                    prediction = self.predict_gaze_point(features)
                    
                    if prediction:
                        pred_x, pred_y = prediction
                        
                        # Apply smoothing
                        smoothed_x_int, smoothed_y_int = self.apply_smoothing(pred_x, pred_y)
                        
                        # Draw gaze point (smaller and semi-transparent in text mode)
                        if mode == "text_analysis":
                            # Create a separate layer for the semi-transparent dot
                            overlay = window.copy()
                            cv2.circle(overlay, (smoothed_x_int, smoothed_y_int), 8, (0, 0, 255), -1)
                            cv2.addWeighted(overlay, 0.6, window, 0.4, 0, window)
                        else:
                            # Standard mode - larger, solid dot
                            cv2.circle(window, (smoothed_x_int, smoothed_y_int), 15, (0, 0, 255), -1)
                            
                            # Draw head pose information (only in standard mode)
                            if rotation_vector is not None and translation_vector is not None:
                                h, w = frame.shape[:2]
                                focal_length = w
                                center = (w/2, h/2)
                                camera_matrix = np.array([
                                    [focal_length, 0, center[0]],
                                    [0, focal_length, center[1]],
                                    [0, 0, 1]
                                ], dtype=np.float32)
                                
                                self.draw_head_pose_axis(window, rotation_vector, translation_vector, camera_matrix)
                            
                            # Display head pose values (only in standard mode)
                            yaw, pitch, roll = features[4], features[5], features[6]
                            cv2.putText(window, f"Yaw: {yaw:.2f}", (50, 100), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                            cv2.putText(window, f"Pitch: {pitch:.2f}", (50, 130), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                            cv2.putText(window, f"Roll: {roll:.2f}", (50, 160), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Add instructions
                text_color = (0, 0, 0) if mode == "text_analysis" else (255, 255, 255)
                cv2.putText(window, f"{window_name} - Press 'q' to quit", 
                           (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
                
                cv2.imshow(window_name, window)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
    
    def run_prediction(self, model_path=None, mode="standard"):
        """Run real-time prediction with specified model and mode"""
        print("=== Gaze Prediction System ===")
        
        # Load model if provided
        if model_path and model_path != self.model_path:
            if not self.load_model(model_path):
                print("Failed to load model. Cannot run prediction.")
                return
        
        if self.model is None:
            print("No model available. Please provide a model path or train a model first.")
            return
        
        # Run real-time prediction
        self.real_time_prediction(mode)
        
        print("=== Prediction Session Complete ===")

if __name__ == "__main__":
    # Example usage
    predictor = GazePredictor()
    
    # Load a trained model
    model_path = "models/gaze_model_20250821_120000.joblib"  # Replace with actual path
    
    # Run prediction in standard mode
    predictor.run_prediction(model_path, mode="standard")
    
    # Or run in text analysis mode
    # predictor.run_prediction(model_path, mode="text_analysis")
