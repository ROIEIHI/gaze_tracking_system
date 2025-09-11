import cv2
import mediapipe as mp
import numpy as np
import joblib
import os
from calibration import EyeTrackerCalibrator
from eye_movement_analyzer import EyeMovementAnalyzer
from utils.features import assemble_features_from_row, FEATURE_COLUMNS

class GazePredictor:
    def __init__(self, model_path=None):
        # Initialize the calibrator for feature extraction
        self.calibrator = EyeTrackerCalibrator()
        
        # Prediction-specific attributes
        self.model = None
        self.scaler = None  # Add scaler attribute
        self.model_path = model_path
        self.smoothed_x = None
        self.smoothed_y = None
        
        # Initialize eye movement analyzer
        self.movement_analyzer = None
        self.analysis_enabled = False
        
        # Multi-page text system
        self.current_page = 0
        self.total_pages = 4
        
        # Load model if path provided
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path):
        """Load a trained model from file"""
        try:
            model_data = joblib.load(model_path)
            self.model = model_data['model']
            self.scaler = model_data.get('scaler', None)  # Load the scaler!
            self.model_path = model_path
            print(f"Model loaded from: {model_path}")
            
            # Strict schema validation - fail fast on drift
            self.saved_feature_columns = model_data.get('feature_columns')
            if not self.saved_feature_columns:
                print("❌ Missing feature_columns in model file. Retrain with model_training.py to embed schema.")
                return False
            
            if self.scaler is None:
                print("❌ Missing scaler in model file. Retrain so scaler is saved.")
                return False
                
            if hasattr(self.scaler, 'n_features_in_') and self.scaler.n_features_in_ != len(self.saved_feature_columns):
                print(f"❌ Scaler/input dimension mismatch: scaler expects {self.scaler.n_features_in_} vs saved schema {len(self.saved_feature_columns)}. Retrain.")
                return False
                
            if self.saved_feature_columns != FEATURE_COLUMNS:
                print("⚠️ Saved feature order differs from code. Using saved order exclusively at inference.")
            
            # Print model info if available
            training_history = model_data.get('training_history', {})
            if training_history:
                test_error = training_history.get('test_mean_error', 'N/A')
                print(f"Model test error: {test_error:.2f} pixels" if isinstance(test_error, float) else f"Model test error: {test_error}")
            
            # Confirm scaler loading
            if self.scaler is not None:
                print("✅ Feature scaler loaded successfully")
            else:
                print("⚠️ No scaler found in model file")
            
            return True
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
    
    def load_default_model(self):
        """Load the default model with fixed filename."""
        # Try multiple possible locations for the model
        possible_paths = [
            "models/gaze_prediction_model.joblib",  # Local models directory
            "../models/gaze_prediction_model.joblib",  # Parent directory models
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "gaze_prediction_model.joblib")  # Absolute path to parent models
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                print(f"Found model at: {path}")
                success = self.load_model(path)
                if success:
                    print(f"Model loading result: {success}")
                    return True
                
        print("No default model found. Please train a model first.")
        return False
    
    def enable_movement_analysis(self, window_width=None, window_height=None):
        """Enable eye movement analysis during prediction"""
        if window_width is None:
            window_width = self.calibrator.WINDOW_WIDTH
        if window_height is None:
            window_height = self.calibrator.WINDOW_HEIGHT
            
        self.movement_analyzer = EyeMovementAnalyzer(window_width, window_height)
        self.analysis_enabled = True
        print("✅ Eye movement analysis enabled")
    
    def disable_movement_analysis(self):
        """Disable eye movement analysis"""
        self.analysis_enabled = False
        self.movement_analyzer = None
        print("❌ Eye movement analysis disabled")
    
    def start_analysis_session(self):
        """Start a new movement analysis session"""
        if self.movement_analyzer:
            self.movement_analyzer.start_reading_session()
            print("📊 Started eye movement analysis session")
    
    def finish_analysis_session(self, auto_export=True, text_reading_mode=False):
        """Finish movement analysis session and export data"""
        if self.movement_analyzer:
            if text_reading_mode:
                # Use specialized text reading export
                csv_file = self.movement_analyzer.export_text_reading_csv()
            else:
                # Use standard export
                csv_file = self.movement_analyzer.finish_analysis_session(auto_export)
            return csv_file
        return None
    
    def predict_gaze_point(self, features):
        """Predict gaze coordinates from extracted features"""
        if self.model is None:
            print("No model loaded. Cannot make predictions.")
            return None
        
        try:
            # Extract raw features: [norm_x_L, norm_y_L, norm_x_R, norm_y_R, yaw, pitch, roll]
            if len(features) < 7:
                print(f"Insufficient features: got {len(features)}, need 7")
                return None
                
            norm_x_L, norm_y_L, norm_x_R, norm_y_R, yaw, pitch, roll = features[:7]
            
            # Build engineered DataFrame using shared function
            eng_df, _ = assemble_features_from_row(norm_x_L, norm_y_L, norm_x_R, norm_y_R, yaw, pitch, roll)
            
            # Use model's saved order exclusively
            cols = self.saved_feature_columns
            
            # Minimal diagnostics under DEBUG_FEATURES=1
            if os.environ.get('DEBUG_FEATURES', '0') == '1':
                print(f"Engineered columns: {list(eng_df.columns)}")
                print(f"Using saved order: {cols}")
                if hasattr(self.scaler, 'n_features_in_'):
                    print(f"Scaler expects: {self.scaler.n_features_in_} features")
            
            # Strict alignment - enforce DataFrame reindex before scaling
            try:
                X1 = eng_df.reindex(columns=cols)
            except Exception as e:
                print(f"Feature alignment error: {e}")
                return None
                
            if X1.isna().any().any():
                missing = [c for c in cols if c not in eng_df.columns]
                print(f"❌ Missing engineered columns: {missing}")
                return None
            
            # Scale and predict with strict dimension checks
            if self.scaler is None:
                print("⚠️ No scaler in model; using unscaled features")
                X1_scaled = X1.values
            else:
                if hasattr(self.scaler, 'n_features_in_') and self.scaler.n_features_in_ != len(cols):
                    print(f"❌ Scaler expects {self.scaler.n_features_in_} features; saved schema has {len(cols)}")
                    return None
                X1_scaled = self.scaler.transform(X1.values)
            
            pred = self.model.predict(X1_scaled)
            return int(pred[0][0]), int(pred[0][1])
            
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            return None
    
    def predict_gaze_point_with_kalman(self, features):
        """Enhanced gaze prediction with adaptive Kalman filtering for better accuracy in text reading"""
        # Get raw prediction first
        raw_prediction = self.predict_gaze_point(features)
        if not raw_prediction:
            return None
        
        raw_x, raw_y = raw_prediction
        
        # If movement analyzer is active and has Kalman filter, use it for enhancement
        if self.analysis_enabled and self.movement_analyzer:
            # Get Kalman prediction
            kalman_prediction = self.movement_analyzer.predict_next_gaze()
            
            if kalman_prediction:
                kalman_x, kalman_y = kalman_prediction
                
                # Adaptive blending based on reading context
                current_movement_type = self.movement_analyzer.classify_movement_type()
                
                if current_movement_type == "Fixation":
                    # During fixations, trust Kalman more for stability
                    blend_factor = 0.1  # 70% Kalman, 30% raw
                elif current_movement_type == "Saccade":
                    # During saccades, trust raw prediction more for responsiveness
                    blend_factor = 0.1  # 10% Kalman, 90% raw
                else:  # Smooth pursuit
                    # Balanced blending for smooth movements
                    blend_factor = 0.1  # 15% Kalman, 85% raw

                # Enhanced blending with bounds checking
                enhanced_x = int(raw_x * (1 - blend_factor) + kalman_x * blend_factor)
                enhanced_y = int(raw_y * (1 - blend_factor) + kalman_y * blend_factor)
                
                # Ensure predictions stay within screen bounds
                enhanced_x = max(0, min(self.calibrator.WINDOW_WIDTH - 1, enhanced_x))
                enhanced_y = max(0, min(self.calibrator.WINDOW_HEIGHT - 1, enhanced_y))
                
                return enhanced_x, enhanced_y
        
        # Fallback to raw prediction if Kalman not available
        return raw_prediction
    
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
        """Create multi-page window with reading text for text analysis mode"""
        # Define the 4 pages of text content
        pages = [
            # Page 1
            """Most pet owners are clear about the immediate joys that come with sharing their lives with companion animals.
However, many of us remain unaware of the physical and mental health benefits that can also accompany the
pleasure of snuggling up to a furry friend. It's only recently that studies have begun to scientifically explore the
benefits of the human-animal bond.""",
            
            # Page 2
            """Pets have evolved to become acutely attuned to humans and our behavior and emotions. Dogs, for example, are
able to understand many of the words we use, but they're even better at interpreting our tone of voice, body
language, and gestures. And like any good human friend, a loyal dog will look into your eyes to gauge your
emotional state and try to understand what you're thinking and feeling (and to work out when the next walk or
treat might be coming, of course).""",
            
            # Page 3
            """Pets, especially dogs and cats, can reduce stress, anxiety, and depression, ease loneliness, encourage exercise
and playfulness, and even improve your cardiovascular health. Caring for an animal can help children grow up
more secure and active. Pets also provide valuable companionship for older adults. Perhaps most importantly,
though, a pet can add real joy and unconditional love to your life.""",
            
            # Page 4
            """Any pet can improve your health
While it's true that people with pets often experience greater health benefits than those without, a pet doesn't
necessarily have to be a dog or a cat. A rabbit could be ideal if you're allergic to other animals or have limited
space but still want a furry friend to snuggle with. Birds can encourage social interaction and help keep your
mind sharp if you're an older adult. Snakes, lizards, and other reptiles can make for exotic companions. Even
watching fish in an aquarium can help reduce muscle tension and lower your pulse rate."""
        ]
        
        # Create white background window
        window = np.ones((self.calibrator.WINDOW_HEIGHT, self.calibrator.WINDOW_WIDTH, 3), dtype=np.uint8) * 255
        
        # Calculate text area (70% of screen width, centered)
        text_width = int(self.calibrator.WINDOW_WIDTH * 0.7)
        text_height = int(self.calibrator.WINDOW_HEIGHT * 0.6)
        
        # Center the text area with slight downward offset for better visual balance
        text_start_x = (self.calibrator.WINDOW_WIDTH - text_width) // 2
        text_start_y = (self.calibrator.WINDOW_HEIGHT - text_height) // 2 + 50
        
        # Font settings
        text_font = cv2.FONT_HERSHEY_DUPLEX
        text_font_scale = 0.6
        text_thickness = 1
        color = (0, 0, 0)  # Black color
        line_spacing = 35
        
        # Get current page text
        current_text = pages[self.current_page]
        
        # Process and display text
        y = text_start_y
        words = current_text.split()
        current_line = ""
        
        for word in words:
            test_line = current_line + (" " if current_line else "") + word
            text_size = cv2.getTextSize(test_line, text_font, text_font_scale, text_thickness)[0]
            
            if text_size[0] <= text_width:
                current_line = test_line
            else:
                if current_line:
                    # Center the line within text area
                    line_size = cv2.getTextSize(current_line, text_font, text_font_scale, text_thickness)[0]
                    line_x = text_start_x + (text_width - line_size[0]) // 2
                    cv2.putText(window, current_line, (line_x, y), text_font, text_font_scale, color, text_thickness)
                    y += line_spacing
                current_line = word
        
        # Draw the remaining text
        if current_line:
            line_size = cv2.getTextSize(current_line, text_font, text_font_scale, text_thickness)[0]
            line_x = text_start_x + (text_width - line_size[0]) // 2
            cv2.putText(window, current_line, (line_x, y), text_font, text_font_scale, color, text_thickness)
        
        # Draw navigation arrows and page info
        self._draw_navigation(window)
        
        # Set text content for word detection if movement analyzer is active
        if self.analysis_enabled and self.movement_analyzer:
            self.movement_analyzer.set_text_content(
                current_text, text_start_x, text_start_y, text_width, line_spacing, text_font_scale
            )
        
        return window
    
    def _draw_navigation(self, window):
        """Draw navigation arrows and page information"""
        arrow_size = 30
        arrow_y = self.calibrator.WINDOW_HEIGHT // 2
        arrow_color = (100, 100, 100)  # Gray color
        arrow_thickness = 3
        
        # Left arrow (if not on first page)
        if self.current_page > 0:
            left_arrow_x = 50
            # Draw left arrow triangle
            pts = np.array([[left_arrow_x + arrow_size, arrow_y - arrow_size//2],
                           [left_arrow_x, arrow_y],
                           [left_arrow_x + arrow_size, arrow_y + arrow_size//2]], np.int32)
            cv2.fillPoly(window, [pts], arrow_color)
        
        # Right arrow (if not on last page)
        if self.current_page < self.total_pages - 1:
            right_arrow_x = self.calibrator.WINDOW_WIDTH - 50 - arrow_size
            # Draw right arrow triangle
            pts = np.array([[right_arrow_x, arrow_y - arrow_size//2],
                           [right_arrow_x + arrow_size, arrow_y],
                           [right_arrow_x, arrow_y + arrow_size//2]], np.int32)
            cv2.fillPoly(window, [pts], arrow_color)
        
        # Page information
        page_info = f"Page {self.current_page + 1} of {self.total_pages}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        text_size = cv2.getTextSize(page_info, font, font_scale, thickness)[0]
        page_x = (self.calibrator.WINDOW_WIDTH - text_size[0]) // 2
        page_y = self.calibrator.WINDOW_HEIGHT - 50
        cv2.putText(window, page_info, (page_x, page_y), font, font_scale, (50, 50, 50), thickness)
    
    def handle_page_navigation(self, key):
        """Handle page navigation based on key press"""
        key_code = key & 0xFF
        
        if key_code == ord('a') or key_code == ord('A'):  # A key - previous page
            if self.current_page > 0:
                self.current_page -= 1
                return True
        elif key_code == ord('d') or key_code == ord('D'):  # D key - next page
            if self.current_page < self.total_pages - 1:
                self.current_page += 1
                return True
        return False
    
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
                # ALWAYS enable movement analysis for text reading mode
                print("📊 Automatic eye movement analysis enabled for text reading")
                print("🔮 Enhanced Kalman filtering activated for improved accuracy")
                self.enable_movement_analysis()
                self.start_analysis_session()
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
                
                # Extract pupil size and blink data for analysis
                pupil_blink_data = self.calibrator.extract_pupil_and_blink_data(results)
                
                if result and self.model:
                    features, (rotation_vector, translation_vector) = result
                    
                    # Add pupil size and blink data to movement analyzer if available
                    if pupil_blink_data and self.analysis_enabled and self.movement_analyzer:
                        pupil_size, is_blink = pupil_blink_data
                        self.movement_analyzer.add_pupil_size(pupil_size)
                        if is_blink:
                            self.movement_analyzer.record_blink()
                if result and self.model:
                    features, (rotation_vector, translation_vector) = result
                    
                    # Use Kalman-enhanced prediction for text reading mode
                    if mode == "text_analysis" and self.analysis_enabled:
                        prediction = self.predict_gaze_point_with_kalman(features)
                    else:
                        prediction = self.predict_gaze_point(features)
                    
                    if prediction:
                        pred_x, pred_y = prediction
                        
                        # Flip x-coordinate if frame is flipped to maintain mirror effect
                        if not self.calibrator.FLIP_FRAME:
                            pred_x = self.calibrator.WINDOW_WIDTH - pred_x
                        
                        # Apply smoothing
                        smoothed_x_int, smoothed_y_int = self.apply_smoothing(pred_x, pred_y)
                        
                        # Add gaze point to movement analyzer if enabled
                        if self.analysis_enabled and self.movement_analyzer:
                            self.movement_analyzer.add_gaze_point(smoothed_x_int, smoothed_y_int)
                        
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
                
                # Add instructions in top-left corner
                text_color = (0, 0, 0) if mode == "text_analysis" else (255, 255, 255)
                if mode == "text_analysis":
                    instruction_text = "Press 'q' to quit | 'A' previous page | 'D' next page"
                else:
                    instruction_text = "Press 'q' to quit"
                cv2.putText(window, instruction_text, (20, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
                
                cv2.imshow(window_name, window)
                
                key = cv2.waitKey(1)
                key_code = key & 0xFF
                
                if key_code == ord('q'):
                    break
                elif mode == "text_analysis":
                    # Handle page navigation with A/D keys only
                    if key_code == ord('a') or key_code == ord('A'):  # A key - previous page
                        if hasattr(self, 'current_page') and self.current_page > 0:
                            self.current_page -= 1
                            text_window = self.create_text_window()
                            print(f"Moved to page {self.current_page + 1}")
                    elif key_code == ord('d') or key_code == ord('D'):  # D key - next page
                        if hasattr(self, 'current_page') and self.current_page < self.total_pages - 1:
                            self.current_page += 1
                            text_window = self.create_text_window()
                            print(f"Moved to page {self.current_page + 1}")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            # Finish movement analysis session if it was active
            if self.analysis_enabled and self.movement_analyzer:
                print("\n🔬 Automatic eye movement analysis completed!")
                # Use text reading export for text analysis mode
                csv_file = self.finish_analysis_session(auto_export=True, text_reading_mode=(mode == "text_analysis"))
                if csv_file:
                    if mode == "text_analysis":
                        print(f"📊 Text reading data automatically saved to: {csv_file}")
                        print("📖 Data includes: Fixation Order, Fixated Words, Screen Coordinates, Duration, etc.")
                    else:
                        print(f"📊 Eye movement data automatically saved to: {csv_file}")
                        print("💡 This data is ready for model training!")
                else:
                    print("⚠️  No movement data was collected during this session")
                self.disable_movement_analysis()
    
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
    
    def run_prediction_with_analysis(self, model_path=None, mode="text_analysis"):
        """Run prediction with movement analysis enabled (defaults to text analysis mode)"""
        print("=== Gaze Prediction System with Movement Analysis ===")
        
        # Load model if provided
        if model_path and model_path != self.model_path:
            if not self.load_model(model_path):
                print("Failed to load model. Cannot run prediction.")
                return
        
        if self.model is None:
            print("No model available. Please provide a model path or train a model first.")
            return
        
        # Force text analysis mode for data collection
        if mode != "text_analysis":
            print("⚠️  Switching to text_analysis mode for movement data collection")
            mode = "text_analysis"
        
        # Run real-time prediction with analysis
        self.real_time_prediction(mode)
        
        print("=== Prediction Session with Analysis Complete ===")

if __name__ == "__main__":
    # Example usage
    predictor = GazePredictor()
    
    # Load a trained model
    model_path = "models/gaze_model_20250821_120000.joblib"  # Replace with actual path
    
    # DEFAULT: Text reading with automatic eye movement analysis
    # This automatically collects movement data every time you read text
    predictor.run_prediction(model_path, mode="text_analysis")
    
    # Alternative options:
    # predictor.run_prediction(model_path, mode="standard")        # Standard mode (no text, no analysis)
    # predictor.run_prediction_with_analysis(model_path)          # Explicit analysis mode (same as text_analysis)
