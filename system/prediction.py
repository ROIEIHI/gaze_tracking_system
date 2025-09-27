"""
Enhanced Gaze Prediction with Text Reading Analysis
Integrates with EyeMovementAnalyzer for comprehensive reading behavior analysis
"""

import cv2
import mediapipe as mp
import numpy as np
import joblib
import os
import time
import pandas as pd
from datetime import datetime
from typing import Tuple, Optional, List, Dict
from config import *
from eye_movement_analyzer import EyeMovementAnalyzer, GazePoint, MovementMetrics
import pandas as pd

class TextReadingGazePredictor:
    """Enhanced gaze predictor with text reading analysis capabilities"""
    
    def __init__(self, model_path: str = None, output_dir: str = None):
        """Initialize the enhanced gaze predictor"""
        print("Initializing Text Reading Gaze Predictor...")
        
        # Set output directory
        self.output_dir = output_dir or OUTPUT_DIR
        
        # MediaPipe setup
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=FACE_DETECTION_CONFIDENCE,
            min_tracking_confidence=FACE_TRACKING_CONFIDENCE
        )
        
        # Camera setup
        self.camera = cv2.VideoCapture(CAMERA_INDEX)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        
        # Model components
        self.model = None
        self.scaler = None
        self.baseline_pitch = 0.0
        self.smoothed_x = None
        self.smoothed_y = None
        self.smoothing_factor = 0.3
        
        # Text reading system
        self.current_page = 0
        
        # Blink detection
        self.blink_count = 0
        self.last_blink_time = 0
        self.eye_closed_frames = 0
        self.blink_threshold = 3  # frames needed to register a blink
        self.total_pages = 0
        self.text_pages = []
        self.word_positions = []
        
        # Eye movement analysis
        self.movement_analyzer = EyeMovementAnalyzer(
            window_width=SCREEN_WIDTH,
            window_height=SCREEN_HEIGHT,
            text_reading_mode=True
        )
        self.analysis_enabled = True
        self.session_start_time = None
        self.fixation_data = []
        
        # Load model if provided
        if model_path:
            self.load_model(model_path)
        
        # Initialize text pages
        self._create_text_pages()
    
    def load_model(self, model_path: str) -> bool:
        """Load trained model and scaler"""
        try:
            print(f"Loading model from: {model_path}")
            
            # Load model components
            self.model = joblib.load(model_path)
            scalar_path = model_path.replace('gaze_model.pkl', 'scaler.pkl')
            self.scaler = joblib.load(scalar_path)
            
            if self.model and self.scaler:
                print("Model and scaler loaded successfully")
                return True
            else:
                print("Invalid model file format")
                return False
                
        except Exception as e:
            print(f"Failed to load model: {e}")
            return False
    
    def _create_text_pages(self):
        """Create adaptive text pages with proper word wrapping"""
        print("Creating adaptive text pages...")
        
        # Clean and split text into words
        words = READING_TEXT.strip().split()
        
        # Calculate approximate character width for better estimation
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = TEXT_FONT_SIZE / 32.0  # Match the rendering scale
        thickness = 2
        
        # Test character to estimate average character width
        char_width = cv2.getTextSize("A", font, font_scale, thickness)[0][0]
        margin_x = int(SCREEN_WIDTH * TEXT_MARGIN_X)
        available_width = SCREEN_WIDTH - 2 * margin_x
        
        # More conservative estimate of words per row based on actual width
        estimated_chars_per_row = available_width // (char_width + 2)  # +2 for spacing
        avg_word_length = 6  # Average English word length
        conservative_words_per_row = max(4, min(TEXT_WORDS_PER_ROW, estimated_chars_per_row // avg_word_length))
        
        print(f"Using {conservative_words_per_row} words per row (estimated from screen width)")
        
        # Calculate pages based on conservative word count
        words_per_page = TEXT_ROWS_PER_PAGE * conservative_words_per_row
        
        self.text_pages = []
        current_page_words = []
        
        for i, word in enumerate(words):
            current_page_words.append(word)
            
            # Create new page when reaching word limit
            if len(current_page_words) >= words_per_page or i == len(words) - 1:
                # Organize words into rows
                page_rows = []
                current_row = []
                
                for j, page_word in enumerate(current_page_words):
                    current_row.append(page_word)
                    
                    # Create new row when reaching conservative word limit or end of page
                    if len(current_row) >= conservative_words_per_row or j == len(current_page_words) - 1:
                        page_rows.append(current_row.copy())
                        current_row = []
                
                self.text_pages.append(page_rows)
                current_page_words = []
        
        self.total_pages = len(self.text_pages)
        print(f"Created {self.total_pages} pages with {TEXT_ROWS_PER_PAGE} rows and ~{conservative_words_per_row} words per row")
    
    def _render_text_page(self) -> np.ndarray:
        """Render current text page with precise word positioning and width checking"""
        # Create background
        background = np.full((SCREEN_HEIGHT, SCREEN_WIDTH, 3), TEXT_BACKGROUND, dtype=np.uint8)
        
        # Calculate text area
        margin_x = int(SCREEN_WIDTH * TEXT_MARGIN_X)
        margin_y = int(SCREEN_HEIGHT * TEXT_MARGIN_Y)
        text_width = SCREEN_WIDTH - 2 * margin_x
        
        # Font settings (OpenCV approximation of config settings)
        font = cv2.FONT_HERSHEY_SIMPLEX if not TEXT_FONT_BOLD else cv2.FONT_HERSHEY_DUPLEX
        font_scale = TEXT_FONT_SIZE / 32.0  # REDUCED scaling for better fit
        thickness = 3 if TEXT_FONT_BOLD else 2
        
        # Clear word positions for current page
        self.word_positions = []
        
        if self.current_page < len(self.text_pages):
            page_rows = self.text_pages[self.current_page]
            
            # Render each row with width checking
            for row_idx, row_words in enumerate(page_rows):
                if row_idx >= TEXT_ROWS_PER_PAGE:
                    break
                
                # Calculate row position
                y = margin_y + row_idx * TEXT_LINE_SPACING + int(TEXT_FONT_SIZE)
                
                # Render row with width constraints
                self._render_row_with_width_check(background, row_words, margin_x, y, font, font_scale, thickness, row_idx, text_width)
        
        # Add navigation info
        self._draw_navigation_info(background)
        
        return background
    
    def _render_row_with_width_check(self, image, words, start_x, start_y, font, font_scale, thickness, row_idx, max_width):
        """Render a row of text with width constraints and proper word wrapping"""
        
        # First, check if all words fit in the available width
        total_text = " ".join(words)
        total_size = cv2.getTextSize(total_text, font, font_scale, thickness)[0]
        
        if total_size[0] <= max_width:
            # All words fit - render normally with centering
            x = start_x + (max_width - total_size[0]) // 2
            self._render_row_with_positions(image, words, x, start_y, font, font_scale, thickness, row_idx)
        else:
            # Words don't fit - need to wrap or truncate
            fitted_words = []
            current_width = 0
            
            for word in words:
                word_size = cv2.getTextSize(word + " ", font, font_scale, thickness)[0]
                
                if current_width + word_size[0] <= max_width:
                    fitted_words.append(word)
                    current_width += word_size[0]
                else:
                    break  # Stop adding words when they don't fit
            
            if fitted_words:
                # Render fitted words with even spacing
                self._render_fitted_row(image, fitted_words, start_x, start_y, font, font_scale, thickness, row_idx, max_width)
    
    def _render_fitted_row(self, image, words, start_x, start_y, font, font_scale, thickness, row_idx, max_width):
        """Render words with even distribution across available width"""
        if not words:
            return
        
        if len(words) == 1:
            # Single word - center it
            word_size = cv2.getTextSize(words[0], font, font_scale, thickness)[0]
            x = start_x + (max_width - word_size[0]) // 2
            cv2.putText(image, words[0], (x, start_y), font, font_scale, TEXT_COLOR, thickness)
            
            # Store word position
            self.word_positions.append({
                'word': words[0],
                'x': x + word_size[0] // 2,
                'y': start_y - word_size[1] // 2,
                'row': row_idx,
                'col': 0,
                'width': word_size[0],
                'height': word_size[1],
                'page': self.current_page
            })
            return
        
        # Calculate total word width
        total_word_width = sum(cv2.getTextSize(word, font, font_scale, thickness)[0][0] for word in words)
        
        # Calculate spacing between words
        available_space = max_width - total_word_width
        space_between_words = available_space // (len(words) - 1) if len(words) > 1 else 0
        
        # Render words with calculated spacing
        current_x = start_x
        
        for word_idx, word in enumerate(words):
            word_size = cv2.getTextSize(word, font, font_scale, thickness)[0]
            
            # Render word
            cv2.putText(image, word, (current_x, start_y), font, font_scale, TEXT_COLOR, thickness)
            
            # Store word position
            word_center_x = current_x + word_size[0] // 2
            word_center_y = start_y - word_size[1] // 2
            
            self.word_positions.append({
                'word': word,
                'x': word_center_x,
                'y': word_center_y,
                'row': row_idx,
                'col': word_idx,
                'width': word_size[0],
                'height': word_size[1],
                'page': self.current_page
            })
            
            # Move to next word position
            current_x += word_size[0] + space_between_words
    
    def _render_row_with_positions(self, image, words, start_x, start_y, font, font_scale, thickness, row_idx):
        """Render a row of text and capture precise word positions"""
        current_x = start_x
        
        for word_idx, word in enumerate(words):
            # Get word dimensions
            word_size = cv2.getTextSize(word, font, font_scale, thickness)[0]
            
            # Render word
            cv2.putText(image, word, (current_x, start_y), font, font_scale, TEXT_COLOR, thickness)
            
            # Store word position for analysis
            word_center_x = current_x + word_size[0] // 2
            word_center_y = start_y - word_size[1] // 2
            
            self.word_positions.append({
                'word': word,
                'x': word_center_x,
                'y': word_center_y,
                'row': row_idx,
                'col': word_idx,
                'width': word_size[0],
                'height': word_size[1],
                'page': self.current_page
            })
            
            # Move to next word position
            current_x += word_size[0]
            if word_idx < len(words) - 1:  # Add space except for last word
                space_size = cv2.getTextSize(" ", font, font_scale, thickness)[0]
                current_x += space_size[0]
    
    def _draw_navigation_info(self, image):
        """Draw page navigation and controls"""
        # Page info
        page_info = f"Page {self.current_page + 1} of {self.total_pages}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        
        text_size = cv2.getTextSize(page_info, font, font_scale, thickness)[0]
        x = (SCREEN_WIDTH - text_size[0]) // 2
        y = int(SCREEN_HEIGHT * 0.95)  # Adaptive positioning - 95% down the screen
        
        cv2.putText(image, page_info, (x, y), font, font_scale, (100, 100, 100), thickness)
        
        # Controls
        controls = "ESC: Exit | A: Previous | D: Next | E: Export Data"
        text_size = cv2.getTextSize(controls, font, 0.5, 1)[0]
        x = (SCREEN_WIDTH - text_size[0]) // 2
        y = int(SCREEN_HEIGHT * 0.98)  # Adaptive positioning - 98% down the screen
        
        cv2.putText(image, controls, (x, y), font, 0.5, (150, 150, 150), 1)
    
    def _detect_face_and_predict(self, frame):
        """Detect face and predict gaze point"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return None
        
        face_landmarks = results.multi_face_landmarks[0]
        
        # Detect blinks
        self._detect_blink(face_landmarks)
        
        # Extract features (simplified - adapt to your feature extraction logic)
        features = self._extract_features(face_landmarks, frame.shape)
        
        if features and self.model:
            # Predict gaze point
            prediction = self._predict_gaze_point(features)
            return prediction
        
        return None
    
    def _calculate_head_pose(self, landmarks, frame_shape):
        """Calculate head pose using PnP algorithm"""
        h, w = frame_shape[:2]
        
        # Extract 2D image points
        image_points = []
        for idx in PNP_LANDMARK_INDICES:
            if idx < len(landmarks.landmark):
                landmark = landmarks.landmark[idx]
                image_points.append([landmark.x * w, landmark.y * h])
        
        if len(image_points) != len(PNP_3D_MODEL_POINTS):
            return {'pitch': 0.0, 'yaw': 0.0, 'roll': 0.0, 'tvect_x': 0.0, 'tvect_y': 0.0, 'tvect_z': 0.0}
        
        image_points = np.array(image_points, dtype=np.float64)
        model_points = np.array(PNP_3D_MODEL_POINTS, dtype=np.float64)
        
        # Camera matrix
        focal_length = w
        center = (w/2, h/2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)
        
        dist_coeffs = np.zeros((4, 1))
        
        try:
            success, rotation_vector, translation_vector = cv2.solvePnP(
                model_points, image_points, camera_matrix, dist_coeffs
            )
            
            if not success:
                return {'pitch': 0.0, 'yaw': 0.0, 'roll': 0.0, 'tvect_x': 0.0, 'tvect_y': 0.0, 'tvect_z': 0.0}
            
            # Convert to Euler angles
            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
            sy = np.sqrt(rotation_matrix[0,0]**2 + rotation_matrix[1,0]**2)
            
            singular = sy < 1e-6
            if not singular:
                yaw = np.arctan2(rotation_matrix[1,0], rotation_matrix[0,0])
                pitch = np.arctan2(-rotation_matrix[2,0], sy)
                roll = np.arctan2(rotation_matrix[2,1], rotation_matrix[2,2])
            else:
                yaw = np.arctan2(-rotation_matrix[1,2], rotation_matrix[1,1])
                pitch = np.arctan2(-rotation_matrix[2,0], sy)
                roll = 0
            
            return {
                'pitch': np.degrees(pitch),
                'yaw': np.degrees(yaw),
                'roll': np.degrees(roll),
                'tvect_x': translation_vector[0][0],
                'tvect_y': translation_vector[1][0],
                'tvect_z': translation_vector[2][0]
            }
            
        except Exception as e:
            return {'pitch': 0.0, 'yaw': 0.0, 'roll': 0.0, 'tvect_x': 0.0, 'tvect_y': 0.0, 'tvect_z': 0.0}

    
    def _extract_features(self, landmarks, frame_shape):
        """Extract features from face landmarks using your calibration logic"""
        h, w = frame_shape[:2]
        
        # Get face bounding box  
        x_coords = [landmarks.landmark[i].x for i in range(len(landmarks.landmark))]
        y_coords = [landmarks.landmark[i].y for i in range(len(landmarks.landmark))]
        face_x_min, face_x_max = min(x_coords), max(x_coords)
        face_y_min, face_y_max = min(y_coords), max(y_coords)
        face_width = face_x_max - face_x_min
        face_height = face_y_max - face_y_min

        # Extract iris positions
        left_eye_x = np.mean([landmarks.landmark[i].x for i in LEFT_IRIS_LANDMARKS])
        left_eye_y = np.mean([landmarks.landmark[i].y for i in LEFT_IRIS_LANDMARKS])
        right_eye_x = np.mean([landmarks.landmark[i].x for i in RIGHT_IRIS_LANDMARKS])
        right_eye_y = np.mean([landmarks.landmark[i].y for i in RIGHT_IRIS_LANDMARKS])
        face_x_min, face_x_max = min(x_coords), max(x_coords)
        face_y_min, face_y_max = min(y_coords), max(y_coords)
        face_width = face_x_max - face_x_min
        face_height = face_y_max - face_y_min
        
        # Extract iris positions
        left_eye_x = np.mean([landmarks.landmark[i].x for i in LEFT_IRIS_LANDMARKS])
        left_eye_y = np.mean([landmarks.landmark[i].y for i in LEFT_IRIS_LANDMARKS])
        right_eye_x = np.mean([landmarks.landmark[i].x for i in RIGHT_IRIS_LANDMARKS])
        right_eye_y = np.mean([landmarks.landmark[i].y for i in RIGHT_IRIS_LANDMARKS])
        
        # Normalize to face coordinates
        norm_L_x = (left_eye_x - face_x_min) / face_width if face_width > 0 else 0.5
        norm_L_y = (left_eye_y - face_y_min) / face_height if face_height > 0 else 0.5
        norm_R_x = (right_eye_x - face_x_min) / face_width if face_width > 0 else 0.5
        norm_R_y = (right_eye_y - face_y_min) / face_height if face_height > 0 else 0.5
        
        # Calculate head pose using PnP
        pose = self._calculate_head_pose(landmarks, frame_shape)
        
        # Return features in correct order
        return [norm_L_x, norm_L_y, norm_R_x, norm_R_y, 
                pose['pitch'], pose['yaw'], pose['roll'],
                pose['tvect_x'], pose['tvect_y'], pose['tvect_z']]
    
    def _predict_gaze_point(self, features):
        """Predict gaze point from features using trained model"""
        try:
            # Convert to pandas DataFrame for feature engineering (same as training)
            feature_names = ['norm_L_x', 'norm_L_y', 'norm_R_x', 'norm_R_y', 
                            'pitch', 'yaw', 'roll', 'tvect_x', 'tvect_y', 'tvect_z']
            df_features = pd.DataFrame([features], columns=feature_names)
            
            # Apply SAME feature engineering as training
            df_features['avg_iris_x'] = (df_features['norm_L_x'] + df_features['norm_R_x']) / 2
            df_features['avg_iris_y'] = (df_features['norm_L_y'] + df_features['norm_R_y']) / 2
            df_features['yaw_avg_x_inter'] = df_features['yaw'] * df_features['avg_iris_x']
            df_features['pitch_avg_y_inter'] = df_features['pitch'] * df_features['avg_iris_y']
            
            # Use same feature order as training
            engineered_features = FEATURE_COLUMNS + ['yaw_avg_x_inter', 'pitch_avg_y_inter']
            X = df_features[engineered_features].values
            
            # Scale features if scaler exists
            if self.scaler:
                X = self.scaler.transform(X)
            
            # Make prediction
            prediction = self.model.predict(X)[0]
            
            # Clamp to screen bounds
            pred_x = max(0, min(SCREEN_WIDTH - 1, int(prediction[0])))
            pred_y = max(0, min(SCREEN_HEIGHT - 1, int(prediction[1])))
            
            return (pred_x, pred_y)
                
        except Exception as e:
            print(f"Prediction error: {e}")
            return (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
    
    def _add_gaze_point_to_analysis(self, x, y):
        """Add gaze point to movement analysis"""
        if self.analysis_enabled:
            # Add to Kalman filter
            self.movement_analyzer.add_gaze_point(x, y)
            
            # Check for fixations and word associations
            current_time = time.time()
            if self.session_start_time is None:
                self.session_start_time = current_time
            
            time_from_onset = (current_time - self.session_start_time) * 1000  # Convert to ms
            
            # Find nearest word
            nearest_word = self._find_nearest_word(x, y)
            
            # Get movement metrics
            try:
                metrics = self.movement_analyzer.get_analysis_summary()
                current_velocity = getattr(metrics, 'current_velocity', 0) if hasattr(metrics, 'current_velocity') else 0
            except:
                current_velocity = 0
            
            # Store fixation data
            if current_velocity < SACCADE_VELOCITY_THRESHOLD:  # THIS IS THE LINE YOU ASKED ABOUT
                self.fixation_data.append({
                    'timestamp': current_time,
                    'x': x,
                    'y': y,
                    'word': nearest_word,
                    'time_from_onset': time_from_onset,
                    'page': self.current_page
                })
    
    def _find_nearest_word(self, gaze_x, gaze_y):
        """Find the nearest word to gaze coordinates"""
        if not self.word_positions:
            return "Unknown"
        
        min_distance = float('inf')
        nearest_word = "Unknown"
        
        for word_info in self.word_positions:
            word_x = word_info['x']
            word_y = word_info['y']
            
            distance = np.sqrt((gaze_x - word_x)**2 + (gaze_y - word_y)**2)
            
            if distance < min_distance and distance < WORD_PROXIMITY_THRESHOLD:
                min_distance = distance
                nearest_word = word_info['word']
        
        return nearest_word
    
    def _detect_blink(self, face_landmarks):
        """Detect blinks using eye aspect ratio"""
        # Eye landmarks (MediaPipe indices)
        left_eye = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        right_eye = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        
        def eye_aspect_ratio(eye_landmarks):
            # Compute distances between vertical eye landmarks
            A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
            B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
            # Compute distance between horizontal eye landmarks
            C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
            # Compute eye aspect ratio
            ear = (A + B) / (2.0 * C)
            return ear
        
        # Extract eye coordinates
        h, w = 480, 640  # Camera resolution
        left_coords = np.array([(face_landmarks.landmark[i].x * w, face_landmarks.landmark[i].y * h) for i in left_eye[:6]])
        right_coords = np.array([(face_landmarks.landmark[i].x * w, face_landmarks.landmark[i].y * h) for i in right_eye[:6]])
        
        # Calculate eye aspect ratios
        left_ear = eye_aspect_ratio(left_coords)
        right_ear = eye_aspect_ratio(right_coords)
        
        # Average the eye aspect ratios
        avg_ear = (left_ear + right_ear) / 2.0
        
        # Blink detection threshold (typical value is around 0.25)
        EAR_THRESHOLD = 0.25
        
        if avg_ear < EAR_THRESHOLD:
            self.eye_closed_frames += 1
        else:
            if self.eye_closed_frames >= self.blink_threshold:
                self.blink_count += 1
                self.last_blink_time = time.time()
            self.eye_closed_frames = 0
        
        return avg_ear < EAR_THRESHOLD
    
    def _calculate_blink_frequency(self, duration_seconds):
        """Calculate blinks per minute"""
        if duration_seconds > 0:
            return (self.blink_count / duration_seconds) * 60
        return 0.0
    
    def _export_fixation_data(self) -> str:
        """Export fixation data in the same format as your CSV example"""
        if not self.fixation_data:
            print("No fixation data to export")
            return None
        
        # Calculate session duration for blink frequency
        session_start = self.fixation_data[0]['timestamp'] if self.fixation_data else 0
        session_end = self.fixation_data[-1]['timestamp'] if self.fixation_data else 0
        session_duration = session_end - session_start
        blink_frequency = self._calculate_blink_frequency(session_duration)
        
        # Process fixation data to match your CSV format
        processed_fixations = []
        fixation_order = 1
        
        # Group consecutive gaze points into fixations
        current_fixation = None
        
        for data in self.fixation_data:
            if current_fixation is None:
                current_fixation = {
                    'start_time': data['timestamp'],
                    'end_time': data['timestamp'],
                    'x_positions': [data['x']],
                    'y_positions': [data['y']],
                    'word': data['word'],
                    'time_from_onset': data['time_from_onset']
                }
            else:
                # Check if this continues the current fixation
                time_gap = data['timestamp'] - current_fixation['end_time']
                
                if time_gap < 0.1 and data['word'] == current_fixation['word']:  # Same fixation
                    current_fixation['end_time'] = data['timestamp']
                    current_fixation['x_positions'].append(data['x'])
                    current_fixation['y_positions'].append(data['y'])
                else:  # New fixation
                    # Process completed fixation
                    duration = (current_fixation['end_time'] - current_fixation['start_time']) * 1000
                    
                    if duration >= FIXATION_THRESHOLD:  # Only include significant fixations
                        avg_x = np.mean(current_fixation['x_positions'])
                        avg_y = np.mean(current_fixation['y_positions'])
                        
                        processed_fixations.append({
                            'Fixation_Order': fixation_order,
                            'Fixated_Word': current_fixation['word'],
                            'Fixation_X_Screen': round(avg_x, 2),
                            'Fixation_Y_Screen': round(avg_y, 2),
                            'Fixation_Duration': round(duration, 2),
                            'Time_from_Stimulus_Onset': round(current_fixation['time_from_onset'], 2),
                            'Pupil_Size': 3.87,  # Placeholder - implement pupil size detection
                            'Blink_Frequency': round(blink_frequency, 2)
                        })
                        fixation_order += 1
                    
                    # Start new fixation
                    current_fixation = {
                        'start_time': data['timestamp'],
                        'end_time': data['timestamp'],
                        'x_positions': [data['x']],
                        'y_positions': [data['y']],
                        'word': data['word'],
                        'time_from_onset': data['time_from_onset']
                    }
        
        # Process final fixation
        if current_fixation:
            duration = (current_fixation['end_time'] - current_fixation['start_time']) * 1000
            if duration >= FIXATION_THRESHOLD:
                avg_x = np.mean(current_fixation['x_positions'])
                avg_y = np.mean(current_fixation['y_positions'])
                
                processed_fixations.append({
                    'Fixation_Order': fixation_order,
                    'Fixated_Word': current_fixation['word'],
                    'Fixation_X_Screen': round(avg_x, 2),
                    'Fixation_Y_Screen': round(avg_y, 2),
                    'Fixation_Duration': round(duration, 2),
                    'Time_from_Stimulus_Onset': round(current_fixation['time_from_onset'], 2),
                    'Pupil_Size': 3.87,
                    'Blink_Frequency': round(blink_frequency, 2)
                })
        
        # Create DataFrame and export
        if processed_fixations:
            df = pd.DataFrame(processed_fixations)
            
            # Filter out "Unknown" words - only keep detected words
            df = df[df['Fixated_Word'] != 'Unknown']
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"eye_movement_data_{timestamp}.csv"
            filepath = os.path.join(self.output_dir, filename)
            
            df.to_csv(filepath, index=False)
            print(f"✅ Exported {len(processed_fixations)} fixations to: {filepath}")
            return filepath
        
        return None
    
    def run_text_reading_analysis(self):
        """Run the complete text reading analysis system"""
        if self.model is None:
            print("❌ No model loaded. Please load a model first.")
            return
        
        print("🎯 Starting Text Reading Analysis System...")
        print("=" * 60)
        print("Controls:")
        print("  ESC - Exit and export data")
        print("  A - Previous page")
        print("  D - Next page")
        print("  E - Export current data")
        print("=" * 60)
        
        # Create fullscreen window
        cv2.namedWindow('Text Reading Analysis', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('Text Reading Analysis', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        self.session_start_time = time.time()
        frame_count = 0
        fps_start = time.time()
        
        try:
            while True:
                ret, frame = self.camera.read()
                if not ret:
                    break
                
                frame = cv2.flip(frame, 1)  # Mirror image
                
                # Predict gaze point
                prediction = self._detect_face_and_predict(frame)
                
                # Create text display
                text_display = self._render_text_page()
                
                if prediction:
                    pred_x, pred_y = prediction
                    
                    # Apply smoothing
                    if self.smoothed_x is None:
                        self.smoothed_x, self.smoothed_y = pred_x, pred_y
                    else:
                        self.smoothed_x = self.smoothed_x * (1 - self.smoothing_factor) + pred_x * self.smoothing_factor
                        self.smoothed_y = self.smoothed_y * (1 - self.smoothing_factor) + pred_y * self.smoothing_factor
                    
                    smoothed_x = int(self.smoothed_x)
                    smoothed_y = int(self.smoothed_y)
                    
                    # Add to analysis
                    self._add_gaze_point_to_analysis(smoothed_x, smoothed_y)
                    
                    # Draw gaze point (semi-transparent)
                    overlay = text_display.copy()
                    cv2.circle(overlay, (smoothed_x, smoothed_y), 12, (255, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.6, text_display, 0.4, 0, text_display)
                
                # Calculate and display FPS
                frame_count += 1
                if frame_count % 30 == 0:
                    fps = 30 / (time.time() - fps_start)
                    fps_start = time.time()
                    cv2.putText(text_display, f"FPS: {fps:.1f}", (20, 40), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)
                
                cv2.imshow('Text Reading Analysis', text_display)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    break
                elif key == ord('a') or key == ord('A'):  # Previous page
                    if self.current_page > 0:
                        self.current_page -= 1
                        print(f"Moved to page {self.current_page + 1}")
                elif key == ord('d') or key == ord('D'):  # Next page
                    if self.current_page < self.total_pages - 1:
                        self.current_page += 1
                        print(f"Moved to page {self.current_page + 1}")
                elif key == ord('e') or key == ord('E'):  # Export data
                    self._export_fixation_data()
        
        finally:
            cv2.destroyAllWindows()
            
            # Final data export
            print("\n" + "=" * 60)
            print("SESSION COMPLETE - EXPORTING DATA")
            print("=" * 60)
            
            csv_file = self._export_fixation_data()
            if csv_file:
                print(f"📊 Eye movement data exported to: {csv_file}")
                print("Data format matches your example CSV structure")
            
            print("🎉 Text Reading Analysis Complete!")

def main():
    """Main function for testing"""
    print("🎯 Text Reading Gaze Analysis System")
    print("=" * 50)
    
    # Check for trained model
    model_path = os.path.join(MODELS_DIR, "gaze_model.pkl")
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("Please train a model first using the main system")
        return
    
    # Create predictor and run analysis
    predictor = TextReadingGazePredictor(model_path)
    predictor.run_text_reading_analysis()

if __name__ == "__main__":
    main()
