import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import time
import datetime
import math
import os
from collections import deque

class EyeTrackerCalibrator:
    def __init__(self):
        # Constants
        self.WINDOW_WIDTH = 1080    
        self.WINDOW_HEIGHT = 720  
        self.CAPTURE_FRAMES = 15
        self.SMOOTHING_FACTOR = 0.2
        self.FLIP_FRAME = False  # Global flag to control frame flipping
        
        # Create assets directory if it doesn't exist
        self.assets_dir = os.path.join(os.path.dirname(__file__), 'assets')
        os.makedirs(self.assets_dir, exist_ok=True)
        
        # Boundary box for user positioning (normalized coordinates)
        self.BOUNDARY_LEFT = 0.35
        self.BOUNDARY_RIGHT = 0.65
        self.BOUNDARY_TOP = 0.2
        self.BOUNDARY_BOTTOM = 0.8
        
        # Initialize MediaPipe
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Data storage
        self.calibration_data = []
        
        # Session pitch baseline for consistent adjustment
        self.session_pitch_baseline = None
        
        # Monitoring thresholds
        self.FACE_DETECTION_THRESHOLD = 0.85
        self.session_paused = False
        
        # Setup calibration targets
        self._setup_calibration_targets()
        
        # Boundary hysteresis state tracking
        self.is_currently_in_boundary = False
        self.boundary_exit_timestamp = None
        self.boundary_entry_timestamp = None
        
        # Add variables for blink detection
        self.previous_eye_openness = None
        self.blink_threshold = 0.2  # Lower threshold for more sensitive blink detection
        self.eye_openness_history = deque(maxlen=5)  # Track recent eye openness values
        
    def measure_pitch_baseline(self, cap, min_samples=60):
        """
        Measure pitch baseline by showing a center dot and waiting for user click.
        Collects exactly 60 frames after click and averages them.
        Returns the median pitch value to be used as session baseline.
        """
        print(f"📏 Measuring pitch baseline for vertical accuracy...")
        
        # Create fullscreen window for baseline measurement
        cv2.namedWindow('Calibration', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('Calibration', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        # Set up mouse callback for click detection
        mouse_clicked = [False]
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                mouse_clicked[0] = True
        
        cv2.setMouseCallback('Calibration', mouse_callback)
        
        # Show instruction screen first
        instruction_window = np.zeros((self.WINDOW_HEIGHT, self.WINDOW_WIDTH, 3), dtype=np.uint8)
        center_x = self.WINDOW_WIDTH // 2
        center_y = self.WINDOW_HEIGHT // 2
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5
        thickness = 3
        line_height = 60
        
        # Draw center dot
        cv2.circle(instruction_window, (center_x, center_y), 20, (0, 255, 0), -1)
        
        # Wait for user click
        print("   Waiting for user to click to start baseline measurement...")
        while not mouse_clicked[0]:
            cv2.imshow('Calibration', instruction_window)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key
                return None
        
        print(f"   Collecting {min_samples} frames for baseline measurement...")
        
        # Collect exactly min_samples frames after click
        pitch_samples = []
        frames_collected = 0
        
        # Create measurement window with progress
        while frames_collected < min_samples:
            ret, frame = cap.read()
            if not ret:
                continue
                
            # Flip frame based on global flag
            if not self.FLIP_FRAME:
                frame = cv2.flip(frame, 1)
                
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            
            # Extract features if face is detected
            result = self.extract_iris_features(frame, results)
            if result:
                features, _ = result
                if len(features) >= 7:
                    pitch = features[5]  # Pitch is at index 5
                    pitch_samples.append(pitch)
                    frames_collected += 1
                    
                    cv2.waitKey(1)
        
        if pitch_samples:
            baseline = np.median(pitch_samples)  # Use median instead of mean for robustness
            print(f"✅ Pitch baseline established: {baseline:.2f}° ({len(pitch_samples)} samples)")
            return baseline
        else:
            print("   ⚠️ No valid pitch values collected, using 0.0")
            return 0.0
        
    def extract_pupil_and_blink_data(self, landmarks):
        """Extract pupil size and detect blinks from facial landmarks"""
        if not landmarks.multi_face_landmarks:
            return None, False
            
        face_landmarks = landmarks.multi_face_landmarks[0]
        
        # Left eye landmarks (eye contour points)
        left_eye_top = face_landmarks.landmark[159]     # Top eyelid
        left_eye_bottom = face_landmarks.landmark[145]  # Bottom eyelid
        left_eye_left = face_landmarks.landmark[33]     # Left corner
        left_eye_right = face_landmarks.landmark[133]   # Right corner
        
        # Right eye landmarks  
        right_eye_top = face_landmarks.landmark[386]    # Top eyelid
        right_eye_bottom = face_landmarks.landmark[374] # Bottom eyelid
        right_eye_left = face_landmarks.landmark[362]   # Left corner
        right_eye_right = face_landmarks.landmark[263]  # Right corner
        
        # Calculate eye openness (vertical distance / horizontal distance)
        left_eye_height = abs(left_eye_top.y - left_eye_bottom.y)
        left_eye_width = abs(left_eye_left.x - left_eye_right.x)
        left_eye_openness = left_eye_height / max(left_eye_width, 0.001)
        
        right_eye_height = abs(right_eye_top.y - right_eye_bottom.y)
        right_eye_width = abs(right_eye_left.x - right_eye_right.x)
        right_eye_openness = right_eye_height / max(right_eye_width, 0.001)
        
        # Average eye openness
        avg_eye_openness = (left_eye_openness + right_eye_openness) / 2
        
        # Add to history for better blink detection
        self.eye_openness_history.append(avg_eye_openness)
        
        # Detect blink using improved algorithm
        is_blink = False
        if len(self.eye_openness_history) >= 3:
            # Look for a dip in eye openness (blink pattern)
            recent_values = list(self.eye_openness_history)
            current = recent_values[-1]
            previous = recent_values[-2]
            
            # Detect significant drop in eye openness
            if current < self.blink_threshold and previous > self.blink_threshold:
                is_blink = True
        
        # Estimate pupil size based on eye dimensions (normalized)
        # This is an approximation since we don't have direct pupil measurements
        estimated_pupil_size = (left_eye_width + right_eye_width) / 2 * 100  # Scale for better readability
        
        return estimated_pupil_size, is_blink
        
    def _setup_calibration_targets(self):
        """Setup calibration targets - Comprehensive 21-point calibration"""
        # Screen dimensions for calculations
        screen_w = self.WINDOW_WIDTH
        screen_h = self.WINDOW_HEIGHT
        
        # Define margins
        edge_margin = 20  # Small margin from absolute edges
        grid_margin_percent = 0.2  # 20% margin for middle grid

        # Calculate grid boundaries (20% from edges)
        grid_left = int(screen_w * grid_margin_percent)
        grid_right = int(screen_w * (1 - grid_margin_percent))
        grid_top = int(screen_h * grid_margin_percent)
        grid_bottom = int(screen_h * (1 - grid_margin_percent))
        
        self.calibration_targets = []
        
        # 1. Four corners of the screen
        corners = [
            (edge_margin, edge_margin),                           # Top-left corner
            (screen_w - edge_margin, edge_margin),                # Top-right corner
            (edge_margin, screen_h - edge_margin),                # Bottom-left corner
            (screen_w - edge_margin, screen_h - edge_margin)      # Bottom-right corner
        ]
        
        # 2. Four edges (middle of each edge)
        edges = [
            (screen_w // 2, edge_margin),                         # Top edge center
            (screen_w // 2, screen_h - edge_margin),              # Bottom edge center
            (edge_margin, screen_h // 2),                         # Left edge center
            (screen_w - edge_margin, screen_h // 2)               # Right edge center
        ]
        
        # 3. Enhanced 13-point grid in the middle (20% from edges)
        grid_points = []
        
        # Original 3x3 grid points with better spread
        original_grid = []
        for row in range(3):
            for col in range(3):
                # Calculate evenly spaced points across the grid
                x = int(grid_left + (col / 2.0) * (grid_right - grid_left))
                y = int(grid_top + (row / 2.0) * (grid_bottom - grid_top))
                original_grid.append((x, y))
                grid_points.append((x, y))
        
        # Add intermediate points between specified targets
        # Between target 11 and 13 (Grid TR and Grid MC) - indices 2 and 4
        tr_x, tr_y = original_grid[2]  # Top-Right
        mc_x, mc_y = original_grid[4]  # Middle-Center
        mid_11_13 = ((tr_x + mc_x) // 2, (tr_y + mc_y) // 2)
        grid_points.append(mid_11_13)
        
        # Between target 9 and 13 (Grid TL and Grid MC) - indices 0 and 4
        tl_x, tl_y = original_grid[0]  # Top-Left
        mc_x, mc_y = original_grid[4]  # Middle-Center
        mid_9_13 = ((tl_x + mc_x) // 2, (tl_y + mc_y) // 2)
        grid_points.append(mid_9_13)
        
        # Between target 15 and 13 (Grid BL and Grid MC) - indices 6 and 4
        bl_x, bl_y = original_grid[6]  # Bottom-Left
        mc_x, mc_y = original_grid[4]  # Middle-Center
        mid_15_13 = ((bl_x + mc_x) // 2, (bl_y + mc_y) // 2)
        grid_points.append(mid_15_13)
        
        # Between target 17 and 13 (Grid BR and Grid MC) - indices 8 and 4
        br_x, br_y = original_grid[8]  # Bottom-Right
        mc_x, mc_y = original_grid[4]  # Middle-Center
        mid_17_13 = ((br_x + mc_x) // 2, (br_y + mc_y) // 2)
        grid_points.append(mid_17_13)
        
        # Combine all calibration targets
        self.calibration_targets = corners + edges + grid_points
        
        # Label targets for better tracking during calibration (now 21 total targets)
        self.target_labels = (
            ["Corner TL", "Corner TR", "Corner BL", "Corner BR"] +
            ["Edge Top", "Edge Bottom", "Edge Left", "Edge Right"] +
            ["Grid TL", "Grid TC", "Grid TR", "Grid ML", "Grid MC", "Grid MR", "Grid BL", "Grid BC", "Grid BR"] +
            ["Mid TR-MC", "Mid TL-MC", "Mid BL-MC", "Mid BR-MC"]
        )

    def setup_camera(self):
        """Initialize camera with simple, reliable settings"""
        cap = cv2.VideoCapture(0)  # Use default API (no DirectShow)
        if not cap.isOpened():
            raise Exception("Could not open camera")
        
        # Set only essential camera properties (like working script)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.WINDOW_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.WINDOW_HEIGHT)
        
        return cap

    def extract_iris_features(self, image, landmarks):
        """Extract normalized iris position features relative to face bounding box and head pose"""
        if not landmarks.multi_face_landmarks:
            return None
            
        face_landmarks = landmarks.multi_face_landmarks[0]
        h, w = image.shape[:2]
        
        # Get face bounding box using key facial landmarks
        face_boundary_landmarks = [
            10, 151, 9, 175,    # Top of forehead/face
            234, 454, 132, 361,  # Left and right face boundaries  
            172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323  # Bottom face boundary
        ]
        
        # Extract all face boundary coordinates
        face_x_coords = []
        face_y_coords = []
        
        for idx in face_boundary_landmarks:
            if idx < len(face_landmarks.landmark):
                landmark = face_landmarks.landmark[idx]
                face_x_coords.append(landmark.x)
                face_y_coords.append(landmark.y)
        
        # Calculate face bounding box
        face_left = min(face_x_coords)
        face_right = max(face_x_coords)
        face_top = min(face_y_coords)
        face_bottom = max(face_y_coords)
        
        # Add margin to create a more stable bounding box (10% margin)
        face_width = face_right - face_left
        face_height = face_bottom - face_top
        margin_x = face_width * 0.1
        margin_y = face_height * 0.1
        
        face_left_bounded = max(0, face_left - margin_x)
        face_right_bounded = min(1, face_right + margin_x)
        face_top_bounded = max(0, face_top - margin_y)
        face_bottom_bounded = min(1, face_bottom + margin_y)
        
        # Update face dimensions with margin
        face_width_bounded = face_right_bounded - face_left_bounded
        face_height_bounded = face_bottom_bounded - face_top_bounded
        
        # Get iris positions
        left_iris = face_landmarks.landmark[468]   # Left iris center
        right_iris = face_landmarks.landmark[473]  # Right iris center
        
        # Normalize iris positions relative to face bounding box
        norm_x_L = (left_iris.x - face_left_bounded) / max(face_width_bounded, 0.01)
        norm_y_L = (left_iris.y - face_top_bounded) / max(face_height_bounded, 0.01)
        
        norm_x_R = (right_iris.x - face_left_bounded) / max(face_width_bounded, 0.01)
        norm_y_R = (right_iris.y - face_top_bounded) / max(face_height_bounded, 0.01)
        
        # Clamp values to [0, 1] range to handle edge cases
        norm_x_L = max(0, min(1, norm_x_L))
        norm_y_L = max(0, min(1, norm_y_L))
        norm_x_R = max(0, min(1, norm_x_R))
        norm_y_R = max(0, min(1, norm_y_R))
        
        # Calculate head pose
        yaw, pitch, roll, rotation_vector, translation_vector = self.calculate_head_pose(image, landmarks)
        
        # If head pose calculation failed, use default values
        if yaw is None:
            yaw, pitch, roll = 0.0, 0.0, 0.0
            rotation_vector, translation_vector = None, None
        
        # Return features for model and visualization data
        features = [norm_x_L, norm_y_L, norm_x_R, norm_y_R, yaw, pitch, roll]
        visualization_data = (rotation_vector, translation_vector)
        
        return features, visualization_data

    def calculate_head_pose(self, image, landmarks):
        """Calculate head pose (yaw, pitch, roll) using PnP algorithm"""
        if not landmarks.multi_face_landmarks:
            return None, None, None, None, None
            
        face_landmarks = landmarks.multi_face_landmarks[0]
        h, w = image.shape[:2]
        
        # Define 3D model points for stable facial landmarks
        # Using realistic anthropometric measurements for adult head (in mm)
        model_points = np.array([
            (0.0, 0.0, 0.0),        # Nose tip (1) - reference point
            (0.0, -70.0, -65.0),    # Chin (152) - 70mm below nose, 65mm back
            (-35.0, 15.0, -15.0),   # Left eye inner corner (33) - 35mm left, 15mm up, 15mm back
            (35.0, 15.0, -15.0),    # Right eye inner corner (362) - 35mm right, 15mm up, 15mm back
            (-25.0, -25.0, -10.0),  # Left mouth corner (61) - 25mm left, 25mm down, 10mm back
            (25.0, -25.0, -10.0)    # Right mouth corner (291) - 25mm right, 25mm down, 10mm back
        ], dtype=np.float64)
        
        # Get corresponding 2D image points - CORRECTED nose tip index
        landmark_indices = [1, 152, 33, 362, 61, 291]  # nose tip, chin, eye corners, mouth corners
        image_points = []
        
        for idx in landmark_indices:
            if idx < len(face_landmarks.landmark):
                landmark = face_landmarks.landmark[idx]
                image_points.append([landmark.x * w, landmark.y * h])
        
        if len(image_points) != 6:
            return None, None, None, None, None
            
        image_points = np.array(image_points, dtype=np.float64)
        
        # Camera matrix estimation with proper focal length
        focal_length = w * 0.7  # More accurate focal length approximation
        center = (w/2, h/2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)
        
        # Distortion coefficients
        dist_coeffs = np.zeros((4, 1))
        
        # Solve PnP
        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs
        )
        
        if not success:
            return None, None, None, None, None
        
        # Convert rotation vector to rotation matrix
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        
        # Extract Euler angles
        sy = np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
        
        singular = sy < 1e-6
        
        if not singular:
            x = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
            y = np.arctan2(-rotation_matrix[2, 0], sy)
            z = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        else:
            x = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
            y = np.arctan2(-rotation_matrix[2, 0], sy)
            z = 0
        
        # Convert to degrees and normalize 
        # Note: x=roll, y=pitch, z=yaw based on Tait-Bryan X-Y-Z convention
        roll_deg = x * 180.0 / np.pi
        pitch_deg = y * 180.0 / np.pi  
        yaw_deg = z * 180.0 / np.pi
        
        # Normalize to [-1, 1] range with optimized ranges for each angle type
        # Based on empirical analysis of typical head movements in front of camera
        pitch = np.clip(pitch_deg, -45, 45) / 45.0        # Pitch: ±45° for head up/down
        yaw = np.clip(yaw_deg, -180, 180) / 180.0         # Yaw: ±180° to handle discontinuity
        
        # For roll, center around typical camera position (~-160°) with ±30° range  
        # This gives better sensitivity for natural head tilting movements
        roll_centered = roll_deg + 160  # Center around -160°
        roll = np.clip(roll_centered, -30, 30) / 30.0     # Roll: ±30° around typical position
        
        return yaw, pitch, roll, rotation_vector, translation_vector

    def is_face_in_boundary(self, landmarks, image_width, image_height):
        """Check if face bounding box is completely within the positioning boundary"""
        face_box = self.calculate_face_bounding_box(landmarks)
        
        if face_box is None:
            return False
        
        return (face_box['left'] >= self.BOUNDARY_LEFT and 
                face_box['right'] <= self.BOUNDARY_RIGHT and
                face_box['top'] >= self.BOUNDARY_TOP and 
                face_box['bottom'] <= self.BOUNDARY_BOTTOM)

    def calculate_face_bounding_box(self, landmarks):
        """Calculate the actual bounding box of the user's face"""
        if not landmarks.multi_face_landmarks:
            return None
            
        face_landmarks = landmarks.multi_face_landmarks[0]
        
        face_boundary_landmarks = [
            10, 151, 9, 175,    # Top of forehead/face
            234, 454, 132, 361,  # Left and right face boundaries  
            172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323
        ]
        
        face_x_coords = []
        face_y_coords = []
        
        for idx in face_boundary_landmarks:
            if idx < len(face_landmarks.landmark):
                landmark = face_landmarks.landmark[idx]
                face_x_coords.append(landmark.x)
                face_y_coords.append(landmark.y)
        
        if not face_x_coords or not face_y_coords:
            return None
        
        return {
            'left': min(face_x_coords),
            'right': max(face_x_coords),
            'top': min(face_y_coords),
            'bottom': max(face_y_coords)
        }

    def check_face_detection_quality(self, landmarks):
        """Check if face detection meets quality threshold"""
        if not landmarks.multi_face_landmarks:
            return False
        
        face_landmarks = landmarks.multi_face_landmarks[0]
        required_landmarks = [33, 133, 468, 362, 263, 473, 145, 159, 374, 386]
        
        try:
            for idx in required_landmarks:
                landmark = face_landmarks.landmark[idx]
                if not (0 <= landmark.x <= 1 and 0 <= landmark.y <= 1):
                    return False
            return True
        except (IndexError, AttributeError):
            return False

    def monitor_user_compliance(self, cap, results):
        """Monitor face detection quality and boundary compliance"""
        face_quality_ok = self.check_face_detection_quality(results)
        face_in_boundary = self.is_face_in_boundary(results, self.WINDOW_WIDTH, self.WINDOW_HEIGHT) if results.multi_face_landmarks else False
        
        if not face_quality_ok:
            print("WARNING: Face detection quality insufficient")
            self.session_paused = True
            return self.show_warning_window(cap, 'face_detection')
        
        if not face_in_boundary:
            print("WARNING: User moved outside boundary area")
            self.session_paused = True
            return self.show_warning_window(cap, 'boundary_exit')
        
        return True

    def show_warning_window(self, cap, warning_type):
        """Display warning window when session is paused"""
        warning_messages = {
            'face_detection': {
                'title': 'FACE DETECTION ERROR',
                'message': 'Face not detected with sufficient quality!',
                'instruction': 'Please ensure your face is clearly visible'
            },
            'boundary_exit': {
                'title': 'BOUNDARY VIOLATION',
                'message': 'You have moved outside the required area!',
                'instruction': 'Please position your face within the green box'
            }
        }
        
        msg = warning_messages.get(warning_type, warning_messages['face_detection'])
        cv2.namedWindow('Session Paused - Warning', cv2.WND_PROP_AUTOSIZE)
        
        while self.session_paused:
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Flip frame based on global flag
            if not self.FLIP_FRAME:
                frame = cv2.flip(frame, 1)
                
            frame_resized = cv2.resize(frame, (640, 480))
            rgb_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            
            results = self.face_mesh.process(rgb_frame)
            
            face_quality_ok = self.check_face_detection_quality(results)
            face_in_boundary = self.is_face_in_boundary(results, 640, 480)
            
            if warning_type == 'face_detection' and face_quality_ok:
                if face_in_boundary:
                    self.session_paused = False
                    break
            elif warning_type == 'boundary_exit' and face_in_boundary and face_quality_ok:
                self.session_paused = False
                break
            
            # Draw visual feedback
            self.draw_boundary_box_warning(frame_resized, face_in_boundary and face_quality_ok)
            self.draw_face_bounding_box(frame_resized, results, color=(255, 255, 0))
            
            # Add warning overlay
            overlay = frame_resized.copy()
            cv2.rectangle(overlay, (10, 10), (630, 150), (0, 0, 255), -1)
            cv2.addWeighted(frame_resized, 0.7, overlay, 0.3, 0, frame_resized)
            
            cv2.putText(frame_resized, msg['title'], (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame_resized, msg['message'], (20, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame_resized, msg['instruction'], (20, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame_resized, "Press ESC to abort session", (20, 130), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('Session Paused - Warning', frame_resized)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key
                cv2.destroyWindow('Session Paused - Warning')
                return False
        
        cv2.destroyWindow('Session Paused - Warning')
        return True

    def draw_boundary_box_warning(self, image, is_positioned_correctly):
        """Draw boundary box for warning window"""
        h, w = image.shape[:2]
        
        left = int(self.BOUNDARY_LEFT * w)
        right = int(self.BOUNDARY_RIGHT * w)
        top = int(self.BOUNDARY_TOP * h)
        bottom = int(self.BOUNDARY_BOTTOM * h)
        
        color = (0, 255, 0) if is_positioned_correctly else (0, 0, 255)
        cv2.rectangle(image, (left, top), (right, bottom), color, 2)

    def draw_boundary_box(self, image, is_positioned_correctly, can_proceed=True):
        """Draw the positioning boundary box"""
        h, w = image.shape[:2]
        
        left = int(self.BOUNDARY_LEFT * w)
        right = int(self.BOUNDARY_RIGHT * w)
        top = int(self.BOUNDARY_TOP * h)
        bottom = int(self.BOUNDARY_BOTTOM * h)
        
        if is_positioned_correctly and can_proceed:
            color = (0, 255, 0)  # Green
            text = "Perfect! Press ENTER to continue"
            label = "Ready!"
        elif is_positioned_correctly and not can_proceed:
            color = (0, 255, 255)  # Yellow
            text = "Hold position steady for 1 second..."
            label = "Hold Position"
        else:
            color = (0, 0, 255)  # Red
            text = "Position your YELLOW face box inside the boundary"
            label = "Target Area"
        
        cv2.rectangle(image, (left, top), (right, bottom), color, 3)
        cv2.putText(image, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(image, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    def draw_face_bounding_box(self, image, landmarks, color=(0, 255, 255)):
        """Draw the actual face bounding box for visualization"""
        face_box = self.calculate_face_bounding_box(landmarks)
        
        if face_box is None:
            return
            
        h, w = image.shape[:2]
        
        left_px = int(face_box['left'] * w)
        right_px = int(face_box['right'] * w)
        top_px = int(face_box['top'] * h)
        bottom_px = int(face_box['bottom'] * h)
        
        cv2.rectangle(image, (left_px, top_px), (right_px, bottom_px), color, 2)
        cv2.putText(image, "Your Face", (left_px, top_px - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    def user_positioning_phase(self, cap):
        """Guide user to correct positioning with hysteresis"""
        cv2.namedWindow('User Positioning', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('User Positioning', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        print("--- User Positioning Phase ---")
        print("Position your yellow face box completely within the boundary box and press ENTER")
        
        # Reset hysteresis state
        self.is_currently_in_boundary = False
        self.boundary_exit_timestamp = None
        self.boundary_entry_timestamp = None
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Flip frame based on global flag
            if not self.FLIP_FRAME:
                frame = cv2.flip(frame, 1)
            frame_resized = cv2.resize(frame, (self.WINDOW_WIDTH, self.WINDOW_HEIGHT))
            
            rgb_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            
            # Apply boundary logic with hysteresis
            is_geometrically_in = self.is_face_in_boundary(results, self.WINDOW_WIDTH, self.WINDOW_HEIGHT)
            current_time = time.time()
            
            if is_geometrically_in:
                if self.boundary_exit_timestamp is not None:
                    self.boundary_entry_timestamp = current_time
                elif self.boundary_entry_timestamp is None:
                    self.boundary_entry_timestamp = current_time
                
                self.is_currently_in_boundary = True
                self.boundary_exit_timestamp = None
            else:
                self.boundary_entry_timestamp = None
                if self.boundary_exit_timestamp is None:
                    self.boundary_exit_timestamp = current_time
                elif current_time - self.boundary_exit_timestamp >= 0.25:
                    self.is_currently_in_boundary = False
            
            # Determine UI status
            if is_geometrically_in:
                is_positioned_for_ui = True
            elif self.boundary_exit_timestamp is not None and current_time - self.boundary_exit_timestamp < 0.25:
                is_positioned_for_ui = True
            else:
                is_positioned_for_ui = False
            
            # Check if can proceed
            can_proceed = False
            if is_positioned_for_ui and self.boundary_entry_timestamp is not None:
                time_in_boundary = current_time - self.boundary_entry_timestamp
                can_proceed = time_in_boundary >= 1.0
            
            # Draw UI elements
            self.draw_boundary_box(frame_resized, is_positioned_for_ui, can_proceed)
            self.draw_face_bounding_box(frame_resized, results, color=(0, 255, 255))
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame_resized, face_landmarks, self.mp_face_mesh.FACEMESH_IRISES,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)
                    )
            
            cv2.imshow('User Positioning', frame_resized)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 13 and can_proceed:  # Enter key
                break
            elif key == 27:  # Escape key
                cv2.destroyAllWindows()
                return False
                
        cv2.destroyAllWindows()
        return True

    def show_calibration_instructions(self):
        """Show calibration instructions window before starting calibration"""
        # Create black instruction window
        instruction_window = np.zeros((self.WINDOW_HEIGHT, self.WINDOW_WIDTH, 3), dtype=np.uint8)
        cv2.namedWindow('Calibration Instructions', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('Calibration Instructions', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        print("--- Showing Calibration Instructions ---")
        
        while True:
            # Clear window (black background)
            instruction_window.fill(0)
            
            # Title
            title_text = "Calibration"
            title_size = cv2.getTextSize(title_text, cv2.FONT_HERSHEY_SIMPLEX, 2.0, 3)[0]
            title_x = (self.WINDOW_WIDTH - title_size[0]) // 2
            title_y = self.WINDOW_HEIGHT // 3
            cv2.putText(instruction_window, title_text, (title_x, title_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 3)
            
            # Instructions
            instruction_text1 = "Click on the red circles and keep your gaze stable"
            instruction_size1 = cv2.getTextSize(instruction_text1, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
            instruction_x1 = (self.WINDOW_WIDTH - instruction_size1[0]) // 2
            instruction_y1 = title_y + 100
            cv2.putText(instruction_window, instruction_text1, (instruction_x1, instruction_y1), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            
            # Continue instruction
            continue_text = "Press Enter to continue"
            continue_size = cv2.getTextSize(continue_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            continue_x = (self.WINDOW_WIDTH - continue_size[0]) // 2
            continue_y = instruction_y1 + 80
            cv2.putText(instruction_window, continue_text, (continue_x, continue_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            cv2.imshow('Calibration Instructions', instruction_window)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 13:  # Enter key
                break
            elif key == 27:  # Escape key
                cv2.destroyWindow('Calibration Instructions')
                return False
                
        cv2.destroyWindow('Calibration Instructions')
        return True

    def mouse_callback(self, event, x, y, flags, param):
        """Mouse callback for calibration target clicks"""
        if event == cv2.EVENT_LBUTTONDOWN:
            target_x, target_y = param['target']
            distance = math.sqrt((x - target_x)**2 + (y - target_y)**2)
            if distance <= 30:
                param['clicked'] = True

    def capture_calibration_data(self, cap, target_x, target_y):
        """Capture burst data for a calibration target"""
        window = np.zeros((self.WINDOW_HEIGHT, self.WINDOW_WIDTH, 3), dtype=np.uint8)
        cv2.namedWindow('Calibration', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('Calibration', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        captured_features = []
        
        for frame_idx in range(self.CAPTURE_FRAMES):
            ret, frame = cap.read()
            if not ret:
                continue
                
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            
            if not self.monitor_user_compliance(cap, results):
                print("Session aborted during calibration capture")
                return []
            
            result = self.extract_iris_features(frame, results)
            if result:
                features, _ = result
                # features: [norm_x_L, norm_y_L, norm_x_R, norm_y_R, yaw, pitch_raw, roll]
                norm_x_L, norm_y_L, norm_x_R, norm_y_R, yaw, pitch_raw, roll = features
                
                # Compute adjusted pitch (will be saved as 'pitch')
                pitch = pitch_raw - self.session_pitch_baseline if self.session_pitch_baseline is not None else pitch_raw
                
                # Store: [norm_x_L, norm_y_L, norm_x_R, norm_y_R, yaw, pitch, roll, target_x, target_y]
                row_data = [norm_x_L, norm_y_L, norm_x_R, norm_y_R, yaw, pitch, roll,
                           target_x, target_y]
                captured_features.append(row_data)
            
            # Show capture animation
            window.fill(0)
            radius = int(20 - (frame_idx / self.CAPTURE_FRAMES) * 20)
            cv2.circle(window, (target_x, target_y), radius, (0, 0, 255), -1)
            #cv2.putText(window, f"Capturing... {frame_idx + 1}/{self.CAPTURE_FRAMES}", 
                       #(50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            cv2.imshow('Calibration', window)
            cv2.waitKey(100)
        
        return captured_features

    def calibration_process(self, cap):
        """Main calibration process"""
        total_targets = len(self.calibration_targets)
        print(f"--- Starting Calibration ({total_targets} targets) ---")
        
        for i, (target_x, target_y) in enumerate(self.calibration_targets):
            target_label = self.target_labels[i] if i < len(self.target_labels) else f"Target {i+1}"
            print(f"Calibration target {i + 1}/{total_targets}: {target_label} at ({target_x}, {target_y})")
            
            # Create calibration window
            window = np.zeros((self.WINDOW_HEIGHT, self.WINDOW_WIDTH, 3), dtype=np.uint8)
            cv2.namedWindow('Calibration', cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty('Calibration', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            
            # Draw target
            cv2.circle(window, (target_x, target_y), 30, (0, 0, 255), -1)
            cv2.circle(window, (target_x, target_y), 35, (255, 255, 255), 2)
            
            # Mouse callback
            mouse_data = {'target': (target_x, target_y), 'clicked': False}
            cv2.setMouseCallback('Calibration', self.mouse_callback, mouse_data)
            
            # Wait for click
            while not mouse_data['clicked']:
                ret, frame = cap.read()
                if ret:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = self.face_mesh.process(rgb_frame)
                    
                    if not self.monitor_user_compliance(cap, results):
                        print("Session aborted during calibration")
                        cv2.destroyAllWindows()
                        return False
                
                cv2.imshow('Calibration', window)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # Escape
                    cv2.destroyAllWindows()
                    return False
            
            # Capture data
            captured_data = self.capture_calibration_data(cap, target_x, target_y)
            if not captured_data:
                print("Calibration data capture failed")
                cv2.destroyAllWindows()
                return False
                
            self.calibration_data.extend(captured_data)
            time.sleep(0.5)
        
        cv2.destroyAllWindows()
        return True

    def export_calibration_data(self, filename=None):
        """Export calibration data to CSV"""
        if not self.calibration_data:
            print("No calibration data to export!")
            return None
        
        df = pd.DataFrame(self.calibration_data, 
                         columns=['norm_x_L', 'norm_y_L', 'norm_x_R', 'norm_y_R', 'yaw', 'pitch', 'roll', 'target_x', 'target_y'])
        
        if filename is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            # Create calibration_data directory if it doesn't exist
            calibration_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'calibration_data')
            os.makedirs(calibration_dir, exist_ok=True)
            filename = os.path.join(calibration_dir, f"calibration_data_{timestamp}.csv")
        
        df.to_csv(filename, index=False)
        print(f"Calibration data exported to {filename}")
        return filename

    def run_calibration(self):
        """Run the complete calibration process with proper flow"""
        print("=== Eye Tracking Calibration ===")
        
        try:
            # Step 1: Setup camera and video preview
            cap = self.setup_camera()
            
            # Step 2: User positioning phase (video preview)
            if not self.user_positioning_phase(cap):
                print("Calibration cancelled during positioning phase")
                return None
            
            # Step 3: Show start calibration message
            if not self.show_calibration_instructions():
                print("Calibration cancelled during instruction phase")
                return None
            
            # Step 4: Measure pitch baseline with mouse click + 60 frames
            self.session_pitch_baseline = self.measure_pitch_baseline(cap, min_samples=60)
            if self.session_pitch_baseline is None:
                print("Baseline measurement cancelled")
                return None
            
            # Step 5: Run calibration loop with 21 targets
            if not self.calibration_process(cap):
                print("Calibration cancelled during calibration phase")
                return None
            
            # Step 6: Export data
            filename = self.export_calibration_data()
            
            cap.release()
            cv2.destroyAllWindows()
            
            print("=== Calibration Complete ===")
            return filename
            
        except Exception as e:
            print(f"Error during calibration: {str(e)}")
            return None

    def run_calibration_with_camera(self, cap):
        """Run the complete calibration process with an existing camera"""
        print("=== Eye Tracking Calibration ===")
        
        try:
            # User positioning phase
            if not self.user_positioning_phase(cap):
                print("Calibration cancelled during positioning phase")
                return None
            
            # Show calibration instructions
            if not self.show_calibration_instructions():
                print("Calibration cancelled during instruction phase")
                return None
            
            # Calibration process
            if not self.calibration_process(cap):
                print("Calibration cancelled during calibration phase")
                return None
            
            # Export data
            filename = self.export_calibration_data()
            
            # Note: Don't release camera here as it's managed by main
            cv2.destroyAllWindows()
            
            print("=== Calibration Complete ===")
            return filename
            
        except Exception as e:
            print(f"Error during calibration: {str(e)}")
            return None
