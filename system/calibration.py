"""
Calibration module for the Gaze Tracking System
Handles face detection, feature extraction, and calibration data collection
"""

import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
import time
import os
from datetime import datetime
from typing import Tuple, List, Dict, Optional
from config import *

class GazeCalibrator:
    """Main calibration class handling all calibration processes"""
    
    def __init__(self):
        """Initialize the calibrator with MediaPipe and OpenCV setup"""
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
        
        # Calibration data storage
        self.calibration_data = []
        self.baseline_pitch = 0.0
        self.baseline_yaw = 0.0
        self.baseline_roll = 0.0
        
        # Face stability tracking
        self.face_stable_start = None
        self.last_face_detected = False
        
        # Current calibration state
        self.current_point = 0
        self.calibration_points = self._generate_calibration_points()
        
    def _generate_calibration_points(self) -> List[Tuple[int, int]]:
        """Generate comprehensive calibration targets: strategic points + full grid"""
        points = []
        
        # ========================================================================
        # STRATEGIC POINTS (21 points)
        # ========================================================================
        
        # Screen boundaries
        edge_margin = 20
        w, h = SCREEN_WIDTH, SCREEN_HEIGHT
        
        # Corners (4 points)
        corners = [
            (edge_margin, edge_margin),                    # Top-left
            (w - edge_margin, edge_margin),                # Top-right
            (edge_margin, h - edge_margin),                # Bottom-left
            (w - edge_margin, h - edge_margin)             # Bottom-right
        ]
        
        # Edges (4 points)
        edges = [
            (w // 2, edge_margin),                         # Top center
            (w // 2, h - edge_margin),                     # Bottom center
            (edge_margin, h // 2),                         # Left center
            (w - edge_margin, h // 2)                      # Right center
        ]
        
        # Inner grid (13 points) - 20% margin from edges
        margin = 0.2
        grid_left = int(w * margin)
        grid_right = int(w * (1 - margin))
        grid_top = int(h * margin)
        grid_bottom = int(h * (1 - margin))
        
        # 3x3 base grid (9 points)
        inner_grid = []
        for row in range(3):
            for col in range(3):
                x = grid_left + col * (grid_right - grid_left) // 2
                y = grid_top + row * (grid_bottom - grid_top) // 2
                inner_grid.append((x, y))
        
        # 4 intermediate points between corners and center
        center = inner_grid[4]  # Middle point (index 4)
        intermediates = [
            ((inner_grid[0][0] + center[0]) // 2, (inner_grid[0][1] + center[1]) // 2),  # TL-Center
            ((inner_grid[2][0] + center[0]) // 2, (inner_grid[2][1] + center[1]) // 2),  # TR-Center
            ((inner_grid[6][0] + center[0]) // 2, (inner_grid[6][1] + center[1]) // 2),  # BL-Center
            ((inner_grid[8][0] + center[0]) // 2, (inner_grid[8][1] + center[1]) // 2),  # BR-Center
        ]
        
        # Combine strategic points (21 total)
        strategic_points = corners + edges + inner_grid + intermediates
        
        # ========================================================================
        # ADDITIONAL GRID POINTS
        # ========================================================================
        
        additional_points = []
        for row in range(CALIBRATION_GRID_SIZE):
            for col in range(CALIBRATION_GRID_SIZE):
                # Calculate grid position
                if CALIBRATION_GRID_SIZE > 1:
                    x = CALIBRATION_MARGIN_X + col * (w - 2 * CALIBRATION_MARGIN_X) // (CALIBRATION_GRID_SIZE - 1)
                    y = CALIBRATION_MARGIN_Y + row * (h - 2 * CALIBRATION_MARGIN_Y) // (CALIBRATION_GRID_SIZE - 1)
                    additional_points.append((x, y))
                else:
                    x, y = w // 2, h // 2

        
        # ========================================================================
        # COMBINE AND RETURN
        # ========================================================================
        
        points = strategic_points + additional_points
        
        print(f"Generated {len(points)} calibration points:")
        print(f"  Strategic: {len(strategic_points)}, Additional: {len(additional_points)}")
        
        return points
    
    def _detect_and_validate_face(self, frame: np.ndarray) -> Tuple[bool, Optional[any], str]:
        """
        Detect face and validate it's within acceptable bounds
        Returns: (is_valid, face_landmarks, error_message)
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return False, None, MESSAGES['face_not_detected']
        
        face_landmarks = results.multi_face_landmarks[0]
        
        # Calculate face bounding box
        h, w = frame.shape[:2]
        x_coords = [landmark.x for landmark in face_landmarks.landmark]
        y_coords = [landmark.y for landmark in face_landmarks.landmark]
        
        face_width = (max(x_coords) - min(x_coords)) * w
        face_height = (max(y_coords) - min(y_coords)) * h
        
        # Check face size
        if face_width < MIN_FACE_SIZE or face_height < MIN_FACE_SIZE:
            return False, face_landmarks, MESSAGES['face_too_small']
        
        # Check if face is within acceptable bounds
        face_center_x = (max(x_coords) + min(x_coords)) / 2
        face_center_y = (max(y_coords) + min(y_coords)) / 2

        boundary_margin_x = FACE_BOUNDARY_MARGIN_X
        boundary_margin_y = FACE_BOUNDARY_MARGIN_Y
        sensitivity = FACE_BOUNDARY_SENSITIVITY

        effective_margin_x = boundary_margin_x + sensitivity
        effective_margin_y = boundary_margin_y + sensitivity

        if (face_center_x < effective_margin_x or face_center_x > 1 - effective_margin_x or
            face_center_y < effective_margin_y or face_center_y > 1 - effective_margin_y):
            return False, face_landmarks, MESSAGES['face_out_of_bounds']
        
        return True, face_landmarks, ""
    
    def _extract_iris_positions(self, landmarks, frame_shape: Tuple[int, int]) -> Dict[str, float]:
        """Extract normalized iris positions relative to face"""
        h, w = frame_shape[:2]
        
        # Get face bounding box
        x_coords = [landmarks.landmark[i].x for i in range(len(landmarks.landmark))]
        y_coords = [landmarks.landmark[i].y for i in range(len(landmarks.landmark))]
        
        face_x_min, face_x_max = min(x_coords), max(x_coords)
        face_y_min, face_y_max = min(y_coords), max(y_coords)
        face_width = face_x_max - face_x_min
        face_height = face_y_max - face_y_min
        
        # Extract iris positions (approximate using eye region)
        # Left iris
        left_eye_x = np.mean([landmarks.landmark[i].x for i in LEFT_IRIS_LANDMARKS])
        left_eye_y = np.mean([landmarks.landmark[i].y for i in LEFT_IRIS_LANDMARKS])
        
        # Right iris  
        right_eye_x = np.mean([landmarks.landmark[i].x for i in RIGHT_IRIS_LANDMARKS])
        right_eye_y = np.mean([landmarks.landmark[i].y for i in RIGHT_IRIS_LANDMARKS])
        
        # Normalize to face coordinates
        norm_L_x = (left_eye_x - face_x_min) / face_width if face_width > 0 else 0.5
        norm_L_y = (left_eye_y - face_y_min) / face_height if face_height > 0 else 0.5
        norm_R_x = (right_eye_x - face_x_min) / face_width if face_width > 0 else 0.5
        norm_R_y = (right_eye_y - face_y_min) / face_height if face_height > 0 else 0.5
        
        return {
            'norm_L_x': norm_L_x,
            'norm_L_y': norm_L_y,
            'norm_R_x': norm_R_x,
            'norm_R_y': norm_R_y
        }
    
    def _calculate_head_pose(self, landmarks, frame_shape: Tuple[int, int]) -> Dict[str, float]:
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
        
        # Camera matrix (simplified)
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
            
            # Convert rotation vector to Euler angles
            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
            
            # Extract Euler angles
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
            
            # Convert to degrees
            return {
                'pitch': np.degrees(pitch),
                'yaw': np.degrees(yaw),
                'roll': np.degrees(roll),
                'tvect_x': translation_vector[0][0],
                'tvect_y': translation_vector[1][0],
                'tvect_z': translation_vector[2][0]
            }
            
        except Exception as e:
            print(f"PnP calculation error: {e}")
            return {'pitch': 0.0, 'yaw': 0.0, 'roll': 0.0, 'tvect_x': 0.0, 'tvect_y': 0.0, 'tvect_z': 0.0}
    
    def _extract_features(self, landmarks, frame_shape: Tuple[int, int]) -> Dict[str, float]:
        """Extract all features for one frame"""
        iris_features = self._extract_iris_positions(landmarks, frame_shape)
        pose_features = self._calculate_head_pose(landmarks, frame_shape)
        
        # Combine features in the correct order
        features = {}
        features.update(iris_features)
        features.update(pose_features)
        
        return features
    
    def _draw_face_overlay(self, frame: np.ndarray, landmarks, is_valid: bool, error_msg: str = ""):
        """Draw face detection overlay"""
        h, w = frame.shape[:2]
        
        if landmarks:
            # Draw face boundary
            face_points = []
            for landmark_group in FACE_BOUNDARY_LANDMARKS.values():
                for idx in landmark_group:
                    if idx < len(landmarks.landmark):
                        landmark = landmarks.landmark[idx]
                        face_points.append([int(landmark.x * w), int(landmark.y * h)])
            
            # Left iris center
            left_iris_x = int(np.mean([landmarks.landmark[i].x for i in LEFT_IRIS_LANDMARKS if i < len(landmarks.landmark)]) * w)
            left_iris_y = int(np.mean([landmarks.landmark[i].y for i in LEFT_IRIS_LANDMARKS if i < len(landmarks.landmark)]) * h)
            cv2.circle(frame, (left_iris_x, left_iris_y), 5, GREEN, 1)

            # Right iris center  
            right_iris_x = int(np.mean([landmarks.landmark[i].x for i in RIGHT_IRIS_LANDMARKS if i < len(landmarks.landmark)]) * w)
            right_iris_y = int(np.mean([landmarks.landmark[i].y for i in RIGHT_IRIS_LANDMARKS if i < len(landmarks.landmark)]) * h)
            cv2.circle(frame, (right_iris_x, right_iris_y), 5, GREEN, 1)
            
            # Draw key facial points
            for idx in PNP_LANDMARK_INDICES:
                if idx < len(landmarks.landmark):
                    landmark = landmarks.landmark[idx]
                    cv2.circle(frame, (int(landmark.x * w), int(landmark.y * h)), 3, BLUE, -1)

            # GET FACE BOUNDING BOX FROM ALL LANDMARKS AND DRAW IT
            x_coords = [landmarks.landmark[i].x for i in range(len(landmarks.landmark))]
            y_coords = [landmarks.landmark[i].y for i in range(len(landmarks.landmark))]

            face_x_min = int(min(x_coords) * w)
            face_x_max = int(max(x_coords) * w)
            face_y_min = int(min(y_coords) * h)
            face_y_max = int(max(y_coords) * h)

            # Draw face bounding box from landmarks
            face_box_color = GREEN if is_valid else RED
            cv2.rectangle(frame, (face_x_min, face_y_min), (face_x_max, face_y_max), face_box_color, 2)

            # Add face box dimensions
            face_width = face_x_max - face_x_min
            face_height = face_y_max - face_y_min
            dimensions_text = f"Face: {face_width}x{face_height}px"
            cv2.putText(frame, dimensions_text, (face_x_min, face_y_max + 20), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, face_box_color, 1)
        
        # Draw status
        color = GREEN if is_valid else RED
        status_text = "FACE DETECTED" if is_valid else error_msg
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   FONT_SCALE, color, FONT_THICKNESS)
        
        # Draw face boundary box
        margin_x = int(FACE_BOUNDARY_MARGIN_X * w)
        margin_y = int(FACE_BOUNDARY_MARGIN_Y * h)
        cv2.rectangle(frame, (margin_x, margin_y), (w-margin_x, h-margin_y), YELLOW, 2)
    
    def run_video_preview(self) -> bool:
        """Video preview with face detection"""
        print("Video Preview - Press SPACE when ready")
        
        while True:
            ret, frame = self.camera.read()
            if not ret:
                print("Error: Could not read from camera")
                return False
            
            frame = cv2.flip(frame, 1)  # Mirror the image
            
            # Detect and validate face
            is_valid, landmarks, error_msg = self._detect_and_validate_face(frame)
            
            # Track face stability
            current_time = time.time()
            if is_valid:
                if not self.last_face_detected:
                    self.face_stable_start = current_time
                self.last_face_detected = True
                
                # Check stability duration
                stable_duration = current_time - (self.face_stable_start or current_time)
                if stable_duration >= FACE_STABILITY_THRESHOLD:
                    cv2.putText(frame, "FACE STABLE - Ready for calibration", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, GREEN, FONT_THICKNESS)
            else:
                self.last_face_detected = False
                self.face_stable_start = None
            
            # Draw overlay
            self._draw_face_overlay(frame, landmarks, is_valid, error_msg)
            
            # Instructions
            cv2.putText(frame, MESSAGES['press_space'], (10, frame.shape[0] - 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2)
            cv2.putText(frame, MESSAGES['press_esc'], (10, frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2)
            
            cv2.imshow('Gaze Tracker - Video Preview', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' ') and is_valid:  # Space bar
                cv2.destroyAllWindows()
                return True
            elif key == 27:  # Escape
                cv2.destroyAllWindows()
                return False
    
    def run_calibration_preview(self) -> bool:
        """Calibration preview screen"""
        print("Calibration Preview")
        
        # Create instruction screen
        instruction_frame = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)
        
        # Title
        title = "CALIBRATION"
        title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 2.0, 3)[0]
        title_x = (SCREEN_WIDTH - title_size[0]) // 2
        cv2.putText(instruction_frame, title, (title_x, 200), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2.0, WHITE, 3)
        
        # Instructions
        instructions = [
            "Please click on the red calibration points",
            "and keep your gaze steady on each point",
            "",
            "Tips:",
            "- Sit comfortably and avoid moving your head",
            "- Look directly at each point before clicking",
            "- The calibration will take about 2 minutes",
            "",
            "Press SPACE to start calibration"
        ]
        
        start_y = 300
        for i, instruction in enumerate(instructions):
            text_size = cv2.getTextSize(instruction, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            text_x = (SCREEN_WIDTH - text_size[0]) // 2
            cv2.putText(instruction_frame, instruction, (text_x, start_y + i * 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2)
        
        cv2.namedWindow('Calibration Instructions', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('Calibration Instructions', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow('Calibration Instructions', instruction_frame)
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):  # Space bar
                cv2.destroyAllWindows()
                return True
            elif key == 27:  # Escape
                cv2.destroyAllWindows()
                return False
    
    def run_baseline_calibration(self) -> bool:
        """Baseline pitch/yaw calibration"""
        print("Baseline Calibration")
        
        baseline_data = []
        frames_collected = 0
        collecting = False
        
        # Create fullscreen window
        cv2.namedWindow('Baseline Calibration', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('Baseline Calibration', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        def mouse_callback(event, x, y, flags, param):
            nonlocal collecting, frames_collected
            if event == cv2.EVENT_LBUTTONDOWN and not collecting:
                collecting = True
                frames_collected = 0
        
        cv2.setMouseCallback('Baseline Calibration', mouse_callback)
        
        while True:
            ret, frame = self.camera.read()
            if not ret:
                return False
            
            frame = cv2.flip(frame, 1)
            
            # Validate face
            is_valid, landmarks, error_msg = self._detect_and_validate_face(frame)
            
            if not is_valid:
                # Show error on calibration screen
                calib_frame = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)
                cv2.putText(calib_frame, error_msg, (SCREEN_WIDTH//2 - 200, SCREEN_HEIGHT//2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, RED, 2)
                cv2.imshow('Baseline Calibration', calib_frame)
                cv2.waitKey(1)
                continue
            
            # Create calibration screen
            calib_frame = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)
            
            # Draw center point
            center_x, center_y = SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2
            cv2.circle(calib_frame, (center_x, center_y), CALIBRATION_POINT_SIZE, RED, -1)
            
            # Instructions
            if not collecting:
                cv2.putText(calib_frame, MESSAGES['baseline_instruction'], 
                           (SCREEN_WIDTH//2 - 300, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, WHITE, 2)
            else:
                progress_text = f"Collecting baseline data... {frames_collected}/{BASELINE_FRAMES}"
                cv2.putText(calib_frame, progress_text, 
                           (SCREEN_WIDTH//2 - 200, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, WHITE, 2)
            
            cv2.imshow('Baseline Calibration', calib_frame)
            
            # Collect baseline data
            if collecting and is_valid:
                features = self._extract_features(landmarks, frame.shape[:2])
                baseline_data.append(features)
                frames_collected += 1
                
                if frames_collected >= BASELINE_FRAMES:
                    # Calculate baseline averages
                    self.baseline_pitch = np.mean([f['pitch'] for f in baseline_data])
                    self.baseline_yaw = np.mean([f['yaw'] for f in baseline_data])
                    self.baseline_roll = np.mean([f['roll'] for f in baseline_data])
                    
                    print(f"Baseline calculated: pitch={self.baseline_pitch:.2f}, yaw={self.baseline_yaw:.2f}, roll={self.baseline_roll:.2f}")
                    cv2.destroyAllWindows()
                    return True
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # Escape
                cv2.destroyAllWindows()
                return False
    
    def run_calibration_loop(self) -> bool:
        """Main calibration loop"""
        print("Calibration Loop")

        # Create fullscreen window
        cv2.namedWindow('Calibration', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('Calibration', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        current_point_idx = 0
        collecting = False
        frames_collected = 0
        point_data = []
        animation_frame = 0
        clicked = False
        
        def mouse_callback(event, x, y, flags, param):
            nonlocal collecting, frames_collected, clicked, animation_frame
            if event == cv2.EVENT_LBUTTONDOWN and not collecting:
                # Check if click is near the calibration point
                target_x, target_y = self.calibration_points[current_point_idx]
                distance = np.sqrt((x - target_x)**2 + (y - target_y)**2)
                if distance <= CALIBRATION_POINT_SIZE:
                    collecting = True
                    frames_collected = 0
                    animation_frame = 0
                    clicked = True
        
        cv2.setMouseCallback('Calibration', mouse_callback)
        
        while current_point_idx < len(self.calibration_points):
            ret, frame = self.camera.read()
            if not ret:
                return False
            
            frame = cv2.flip(frame, 1)
            
            # Validate face
            is_valid, landmarks, error_msg = self._detect_and_validate_face(frame)
            
            if not is_valid:
                # Show error
                calib_frame = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)
                cv2.putText(calib_frame, error_msg, (SCREEN_WIDTH//2 - 200, SCREEN_HEIGHT//2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, RED, 2)
                cv2.imshow('Calibration', calib_frame)
                cv2.waitKey(1)
                continue
            
            # Create calibration screen
            calib_frame = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)
            
            # Get current target point
            target_x, target_y = self.calibration_points[current_point_idx]
            
            # Draw calibration point with animation
            if collecting:
                # Shrinking animation during collection
                shrink_progress = min(animation_frame / ANIMATION_SHRINK_FRAMES, 1.0)
                current_radius = int(CALIBRATION_POINT_SIZE * (1 - shrink_progress))
                animation_frame += 1
            else:
                current_radius = CALIBRATION_POINT_SIZE
            
            if current_radius > 0:
                cv2.circle(calib_frame, (target_x, target_y), current_radius, RED, -1)
            
            # Progress information
            progress_text = f"Point {current_point_idx + 1}/{len(self.calibration_points)}"
            cv2.putText(calib_frame, progress_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, WHITE, 2)
            
            if collecting:
                frames_text = f"Collecting: {frames_collected}/{CALIBRATION_FRAMES}"
                cv2.putText(calib_frame, frames_text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, WHITE, 2)
            
            cv2.imshow('Calibration', calib_frame)
            
            # Collect calibration data
            if collecting and is_valid:
                features = self._extract_features(landmarks, frame.shape[:2])
                
                # Add target coordinates
                
                features['point_id'] = current_point_idx
                #features['burst_id'] =
                #features['timestamp'] =
                features['target_x'] = target_x
                features['target_y'] = target_y

                point_data.append(features)
                frames_collected += 1
                
                if frames_collected >= CALIBRATION_FRAMES:
                    # Move to next point
                    self.calibration_data.extend(point_data)
                    point_data = []
                    current_point_idx += 1
                    collecting = False
                    clicked = False
                    print(f"Completed point {current_point_idx}/{len(self.calibration_points)}")
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # Escape
                cv2.destroyAllWindows()
                return False
        
        cv2.destroyAllWindows()
        return True
    
    def save_calibration_data(self) -> str:
        """Save calibration data to CSV"""
        print("Saving calibration data")

        if not self.calibration_data:
            print("No calibration data to save!")
            return ""
        
        # Convert to DataFrame
        df = pd.DataFrame(self.calibration_data)
        
        # Ensure correct column order
        target_columns = FEATURE_COLUMNS + ['target_x', 'target_y', 'point_id']
        
        # Reorder columns
        df = df[target_columns]
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"calibration_data_{timestamp}.csv"
        filepath = os.path.join(CALIBRATION_DATA_DIR, filename)
        
        # Save to CSV
        df.to_csv(filepath, index=False)
        
        print(f"Calibration data saved: {filepath}")
        print(f"Total samples: {len(df)}")
        print(f"Features: {FEATURE_COLUMNS}")
        
        return filepath
    
    def run_full_calibration(self) -> str:
        """Run the complete calibration process"""
        print("Starting Gaze Tracking Calibration System")
        print("="*50)
        
        # Phase 1: Video Preview
        if not self.run_video_preview():
            print("Calibration cancelled at video preview")
            return ""
        
        # Phase 2.1: Calibration Preview
        if not self.run_calibration_preview():
            print("Calibration cancelled at preview")
            return ""
        
        # Phase 2.2: Baseline Calibration
        if not self.run_baseline_calibration():
            print("Calibration cancelled at baseline")
            return ""
        
        # Phase 2.3: Calibration Loop
        if not self.run_calibration_loop():
            print("Calibration cancelled at main loop")
            return ""
        
        # Phase 2.4: Save Data
        filepath = self.save_calibration_data()
        
        # Show completion message
        completion_frame = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)
        cv2.putText(completion_frame, MESSAGES['calibration_complete'], 
                   (SCREEN_WIDTH//2 - 300, SCREEN_HEIGHT//2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, GREEN, 3)
        cv2.putText(completion_frame, f"Data saved: {os.path.basename(filepath)}", 
                   (SCREEN_WIDTH//2 - 200, SCREEN_HEIGHT//2 + 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, WHITE, 2)
        cv2.putText(completion_frame, "Press any key to continue", 
                   (SCREEN_WIDTH//2 - 150, SCREEN_HEIGHT//2 + 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, WHITE, 2)
        
        cv2.imshow('Calibration Complete', completion_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return filepath
    
    def __del__(self):
        """Cleanup camera resources"""
        if hasattr(self, 'camera'):
            self.camera.release()
        cv2.destroyAllWindows()
