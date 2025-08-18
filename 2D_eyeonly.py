import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
import time
import datetime
import math

class EyeTracker:
    def __init__(self):
        # Constants
        self.WINDOW_WIDTH = 1080
        self.WINDOW_HEIGHT = 720
        self.CAPTURE_FRAMES = 10
        self.SMOOTHING_FACTOR = 0.2
        
        # Boundary box for user positioning (normalized coordinates)
        self.BOUNDARY_LEFT = 0.375
        self.BOUNDARY_RIGHT = 0.625
        self.BOUNDARY_TOP = 0.3
        self.BOUNDARY_BOTTOM = 0.7
        
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
        self.model = None
        self.smoothed_x = None
        self.smoothed_y = None
        
        # Monitoring thresholds
        self.FACE_DETECTION_THRESHOLD = 0.85
        self.session_paused = False
        
        # Calibration targets - Comprehensive 17-point calibration
        # 4 corners + 4 edges + 9 middle grid (10% from edges)
        
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
        
        # Original 3x3 grid points
        original_grid = []
        for row in range(3):
            for col in range(3):
                x = grid_left + col * (grid_right - grid_left) // 2
                y = grid_top + row * (grid_bottom - grid_top) // 2
                original_grid.append((x, y))
                grid_points.append((x, y))
        
        # Add intermediate points between specified targets
        # Grid layout indices: 0=TL, 1=TC, 2=TR, 3=ML, 4=MC, 5=MR, 6=BL, 7=BC, 8=BR
        # Targets 9-17 correspond to indices 0-8
        
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
            ["Mid TR-MC", "Mid TL-MC", "Mid BL-MC", "Mid BR-MC"]  # New intermediate targets
        )
        
        # Boundary hysteresis state tracking
        self.is_currently_in_boundary = False
        self.boundary_exit_timestamp = None
        self.boundary_entry_timestamp = None  # Track when user enters boundary
        
    def calculate_face_bounding_box(self, landmarks):
        """Calculate the actual bounding box of the user's face"""
        if not landmarks.multi_face_landmarks:
            return None
            
        face_landmarks = landmarks.multi_face_landmarks[0]
        
        # Use the same comprehensive set of face boundary landmarks as in extract_iris_features
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
        
        if not face_x_coords or not face_y_coords:
            return None
        
        # Calculate actual face bounding box
        face_left = min(face_x_coords)
        face_right = max(face_x_coords)
        face_top = min(face_y_coords)
        face_bottom = max(face_y_coords)
        
        return {
            'left': face_left,
            'right': face_right,
            'top': face_top,
            'bottom': face_bottom
        }
        
    def extract_iris_features(self, image, landmarks):
        """Extract normalized iris position features relative to face bounding box and head pose"""
        if not landmarks.multi_face_landmarks:
            return None
            
        face_landmarks = landmarks.multi_face_landmarks[0]
        h, w = image.shape[:2]
        
        # Get face bounding box using key facial landmarks
        # Use a comprehensive set of face boundary landmarks
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
        # Values will be between 0 and 1, where:
        # (0,0) = top-left of face bounding box
        # (1,1) = bottom-right of face bounding box
        
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
        # Using standard face model coordinates (in mm)
        model_points = np.array([
            (0.0, 0.0, 0.0),         # Nose tip (30)
            (0.0, -330.0, -65.0),    # Chin (152)
            (-225.0, 170.0, -135.0), # Left eye left corner (33)
            (225.0, 170.0, -135.0),  # Right eye right corner (362)
            (-150.0, -150.0, -125.0),# Left mouth corner (61)
            (150.0, -150.0, -125.0)  # Right mouth corner (291)
        ], dtype=np.float32)
        
        # Get corresponding 2D image points
        landmark_indices = [30, 152, 33, 362, 61, 291]  # nose tip, chin, eye corners, mouth corners
        image_points = []
        
        for idx in landmark_indices:
            if idx < len(face_landmarks.landmark):
                landmark = face_landmarks.landmark[idx]
                image_points.append([landmark.x * w, landmark.y * h])
        
        if len(image_points) != 6:
            return None, None, None, None, None
            
        image_points = np.array(image_points, dtype=np.float32)
        
        # Camera matrix estimation (assuming no lens distortion)
        focal_length = w
        center = (w/2, h/2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Distortion coefficients (assuming no distortion)
        dist_coeffs = np.zeros((4, 1))
        
        # Solve PnP
        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs
        )
        
        if not success:
            return None, None, None, None, None
        
        # Convert rotation vector to rotation matrix
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        
        # Extract Euler angles (yaw, pitch, roll) from rotation matrix
        # Using a more direct approach to extract Euler angles
        # Alternative method: extract angles directly from rotation matrix
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
        
        # Convert to degrees and assign to intuitive names
        pitch = x * 180.0 / np.pi  # Rotation around X-axis (up/down)
        yaw = y * 180.0 / np.pi    # Rotation around Y-axis (left/right)
        roll = z * 180.0 / np.pi   # Rotation around Z-axis (tilt)
        
        # Normalize angles to reasonable ranges for feature consistency
        yaw = np.clip(yaw, -90, 90) / 90.0       # Normalize to [-1, 1]
        pitch = np.clip(pitch, -90, 90) / 90.0   # Normalize to [-1, 1]
        roll = np.clip(roll, -45, 45) / 45.0     # Normalize to [-1, 1]
        
        return yaw, pitch, roll, rotation_vector, translation_vector
    
    def is_face_in_boundary(self, landmarks, image_width, image_height):
        """Check if face bounding box is completely within the positioning boundary"""
        face_box = self.calculate_face_bounding_box(landmarks)
        
        if face_box is None:
            return False
        
        # Check if the entire face bounding box is within the absolute boundary
        # The entire face must be within the boundary for stable tracking
        return (face_box['left'] >= self.BOUNDARY_LEFT and 
                face_box['right'] <= self.BOUNDARY_RIGHT and
                face_box['top'] >= self.BOUNDARY_TOP and 
                face_box['bottom'] <= self.BOUNDARY_BOTTOM)
    
    def draw_boundary_box(self, image, is_positioned_correctly, can_proceed=True):
        """Draw the positioning boundary box (absolute boundary)"""
        h, w = image.shape[:2]
        
        left = int(self.BOUNDARY_LEFT * w)
        right = int(self.BOUNDARY_RIGHT * w)
        top = int(self.BOUNDARY_TOP * h)
        bottom = int(self.BOUNDARY_BOTTOM * h)
        
        # Color logic: Green only if positioned AND can proceed
        if is_positioned_correctly and can_proceed:
            color = (0, 255, 0)  # Green - ready to proceed
        elif is_positioned_correctly and not can_proceed:
            color = (0, 255, 255)  # Yellow - positioned but waiting
        else:
            color = (0, 0, 255)  # Red - not positioned correctly
        
        thickness = 3
        
        cv2.rectangle(image, (left, top), (right, bottom), color, thickness)
        
        # Add text instructions
        if not is_positioned_correctly:
            text = "Position your YELLOW face box inside the boundary"
        elif is_positioned_correctly and not can_proceed:
            text = "Hold position steady for 1 second..."
        else:
            text = "Perfect! Press ENTER to continue"
        cv2.putText(image, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add boundary label with status
        if is_positioned_correctly and can_proceed:
            label = "Ready!"
        elif is_positioned_correctly:
            label = "Hold Position"
        else:
            label = "Target Area"
        cv2.putText(image, label, (left, top - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    def draw_face_bounding_box(self, image, landmarks, color=(0, 255, 255)):
        """Draw the actual face bounding box for visualization (yellow by default)"""
        face_box = self.calculate_face_bounding_box(landmarks)
        
        if face_box is None:
            return
            
        h, w = image.shape[:2]
        
        # Convert normalized coordinates to pixel coordinates
        left_px = int(face_box['left'] * w)
        right_px = int(face_box['right'] * w)
        top_px = int(face_box['top'] * h)
        bottom_px = int(face_box['bottom'] * h)
        
        # Draw face bounding box
        cv2.rectangle(image, (left_px, top_px), (right_px, bottom_px), color, 2)
        
        # Add label
        cv2.putText(image, "Your Face", (left_px, top_px - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
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
    
    def visualize_all_calibration_targets(self):
        """Display all calibration targets for verification"""
        print("\n--- Calibration Target Visualization ---")
        window = np.zeros((self.WINDOW_HEIGHT, self.WINDOW_WIDTH, 3), dtype=np.uint8)
        
        # Draw all targets with different colors for different groups
        colors = {
            'corners': (255, 0, 0),    # Red for corners
            'edges': (0, 255, 0),      # Green for edges  
            'grid': (0, 0, 255)        # Blue for grid
        }
        
        for i, (target_x, target_y) in enumerate(self.calibration_targets):
            # Determine target group and color
            if i < 4:  # Corners
                color = colors['corners']
                group = "Corner"
            elif i < 8:  # Edges
                color = colors['edges']
                group = "Edge"
            else:  # Grid
                color = colors['grid']
                group = "Grid"
            
            # Draw target
            cv2.circle(window, (target_x, target_y), 15, color, -1)
            cv2.circle(window, (target_x, target_y), 18, (255, 255, 255), 1)
            
            # Add target number
            cv2.putText(window, str(i + 1), (target_x - 8, target_y + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add legend
        cv2.putText(window, "Calibration Targets Overview", (50, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        cv2.putText(window, f"Total: {len(self.calibration_targets)} targets", (50, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Legend
        legend_y = 120
        cv2.circle(window, (70, legend_y), 10, colors['corners'], -1)
        cv2.putText(window, "Corners (1-4)", (90, legend_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.circle(window, (70, legend_y + 30), 10, colors['edges'], -1)
        cv2.putText(window, "Edges (5-8)", (90, legend_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.circle(window, (70, legend_y + 60), 10, colors['grid'], -1)
        cv2.putText(window, "Grid (9-21)", (90, legend_y + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.putText(window, "Press any key to continue...", (50, self.WINDOW_HEIGHT - 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        cv2.namedWindow('Calibration Target Overview', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('Calibration Target Overview', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow('Calibration Target Overview', window)
        cv2.waitKey(0)
        cv2.destroyWindow('Calibration Target Overview')
    
    def check_face_detection_quality(self, landmarks):
        """Check if face detection meets quality threshold"""
        if not landmarks.multi_face_landmarks:
            return False
        
        # Check detection confidence (this is a simplified check)
        # In practice, MediaPipe doesn't directly provide confidence scores
        # We'll use the presence of key landmarks as a proxy for quality
        face_landmarks = landmarks.multi_face_landmarks[0]
        
        # Check if we have the required landmarks for eye tracking
        required_landmarks = [33, 133, 468, 362, 263, 473, 145, 159, 374, 386]
        
        try:
            for idx in required_landmarks:
                landmark = face_landmarks.landmark[idx]
                # Basic sanity check - landmarks should be within valid range
                if not (0 <= landmark.x <= 1 and 0 <= landmark.y <= 1):
                    return False
            return True
        except (IndexError, AttributeError):
            return False
    
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
                
            frame = cv2.flip(frame, 1)
            frame_resized = cv2.resize(frame, (640, 480))  # Smaller window for warning
            rgb_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            
            results = self.face_mesh.process(rgb_frame)
            
            # Check if issue is resolved
            face_quality_ok = self.check_face_detection_quality(results)
            face_in_boundary = self.is_face_in_boundary(results, 640, 480)
            
            if warning_type == 'face_detection' and face_quality_ok:
                if face_in_boundary:
                    self.session_paused = False
                    break
            elif warning_type == 'boundary_exit' and face_in_boundary and face_quality_ok:
                self.session_paused = False
                break
            
            # Draw boundary box with adjusted coordinates for smaller window
            self.draw_boundary_box_warning(frame_resized, face_in_boundary and face_quality_ok)
            
            # Draw face bounding box for visualization
            self.draw_face_bounding_box(frame_resized, results, color=(255, 255, 0))
            
            # Add warning overlay
            overlay = frame_resized.copy()
            cv2.rectangle(overlay, (10, 10), (630, 150), (0, 0, 255), -1)
            cv2.addWeighted(frame_resized, 0.7, overlay, 0.3, 0, frame_resized)
            
            # Add warning text
            cv2.putText(frame_resized, msg['title'], (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame_resized, msg['message'], (20, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame_resized, msg['instruction'], (20, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame_resized, "Press ESC to abort session", (20, 130), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Show face mesh if detected
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame_resized, face_landmarks, self.mp_face_mesh.FACEMESH_IRISES,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)
                    )
            
            cv2.imshow('Session Paused - Warning', frame_resized)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key to abort
                cv2.destroyWindow('Session Paused - Warning')
                return False
        
        cv2.destroyWindow('Session Paused - Warning')
        return True
    
    def draw_boundary_box_warning(self, image, is_positioned_correctly):
        """Draw boundary box for warning window (smaller size)"""
        h, w = image.shape[:2]
        
        left = int(self.BOUNDARY_LEFT * w)
        right = int(self.BOUNDARY_RIGHT * w)
        top = int(self.BOUNDARY_TOP * h)
        bottom = int(self.BOUNDARY_BOTTOM * h)
        
        color = (0, 255, 0) if is_positioned_correctly else (0, 0, 255)
        thickness = 2
        
        cv2.rectangle(image, (left, top), (right, bottom), color, thickness)
    
    def monitor_user_compliance(self, cap, results):
        """Monitor face detection quality and boundary compliance"""
        face_quality_ok = self.check_face_detection_quality(results)
        face_in_boundary = self.is_face_in_boundary(results, self.WINDOW_WIDTH, self.WINDOW_HEIGHT) if results.multi_face_landmarks else False
        
        if not face_quality_ok:
            print("⚠️ WARNING: Face detection quality insufficient")
            self.session_paused = True
            return self.show_warning_window(cap, 'face_detection')
        
        if not face_in_boundary:
            print("⚠️ WARNING: User moved outside boundary area")
            self.session_paused = True
            return self.show_warning_window(cap, 'boundary_exit')
        
        return True
    
    def user_positioning_phase(self, cap):
        """Guide user to correct positioning with hysteresis"""
        cv2.namedWindow('User Positioning', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('User Positioning', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        print("--- User Positioning Phase ---")
        print("Position your yellow face box completely within the boundary box and press ENTER")
        
        # Reset hysteresis state for positioning phase
        self.is_currently_in_boundary = False
        self.boundary_exit_timestamp = None
        self.boundary_entry_timestamp = None
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame = cv2.flip(frame, 1)  # Mirror the image
            frame_resized = cv2.resize(frame, (self.WINDOW_WIDTH, self.WINDOW_HEIGHT))
            rgb_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            
            results = self.face_mesh.process(rgb_frame)
            
            # Perform geometric boundary check
            is_geometrically_in = self.is_face_in_boundary(results, self.WINDOW_WIDTH, self.WINDOW_HEIGHT)
            
            # Apply hysteresis logic
            current_time = time.time()
            
            if is_geometrically_in:
                # User is back in boundary - reset exit timestamp and start entry tracking
                if self.boundary_exit_timestamp is not None:
                    # User just re-entered - start entry grace period
                    self.boundary_entry_timestamp = current_time
                elif self.boundary_entry_timestamp is None:
                    # User has been in boundary but we haven't started tracking entry time
                    self.boundary_entry_timestamp = current_time
                
                self.is_currently_in_boundary = True
                self.boundary_exit_timestamp = None
            else:
                # User is out of boundary - reset entry timestamp
                self.boundary_entry_timestamp = None
                if self.boundary_exit_timestamp is None:
                    # First frame out of boundary - start grace period
                    self.boundary_exit_timestamp = current_time
                elif current_time - self.boundary_exit_timestamp >= 0.25:
                    # Grace period expired - user is officially out
                    self.is_currently_in_boundary = False
                # If still within grace period, maintain previous state
            
            # Determine final positioning status for UI (exit hysteresis)
            if is_geometrically_in:
                is_positioned_for_ui = True
            elif self.boundary_exit_timestamp is not None and current_time - self.boundary_exit_timestamp < 0.25:
                is_positioned_for_ui = True  # Still in grace period
            else:
                is_positioned_for_ui = False
            
            # Determine if user can proceed (entry grace period)
            can_proceed = False
            entry_time_remaining = 0
            if is_positioned_for_ui and self.boundary_entry_timestamp is not None:
                time_in_boundary = current_time - self.boundary_entry_timestamp
                if time_in_boundary >= 1.0:
                    can_proceed = True
                else:
                    entry_time_remaining = 1.0 - time_in_boundary
            
            # Draw boundary box (changes color based on positioning status)
            self.draw_boundary_box(frame_resized, is_positioned_for_ui, can_proceed)
            
            # Draw face bounding box for visualization (always yellow)
            self.draw_face_bounding_box(frame_resized, results, color=(0, 255, 255))
            
            # Add timing information if in exit grace period
            if not is_geometrically_in and self.boundary_exit_timestamp is not None:
                time_remaining = 0.25 - (current_time - self.boundary_exit_timestamp)
                if time_remaining > 0:
                    cv2.putText(frame_resized, f"Exit grace: {time_remaining:.2f}s", 
                               (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Add timing information for entry grace period
            if is_positioned_for_ui and not can_proceed and entry_time_remaining > 0:
                cv2.putText(frame_resized, f"Stay in position: {entry_time_remaining:.1f}s", 
                           (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame_resized, face_landmarks, self.mp_face_mesh.FACEMESH_IRISES,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)
                    )
            
            cv2.imshow('User Positioning', frame_resized)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 13 and can_proceed:  # Enter key - only allow if user has been in boundary for 1 second
                break
            elif key == 27:  # Escape key
                cv2.destroyAllWindows()
                return False
                
        cv2.destroyAllWindows()
        return True
    
    def mouse_callback(self, event, x, y, flags, param):
        """Mouse callback for calibration target clicks"""
        if event == cv2.EVENT_LBUTTONDOWN:
            target_x, target_y = param['target']
            distance = math.sqrt((x - target_x)**2 + (y - target_y)**2)
            if distance <= 30:  # Click within target circle
                param['clicked'] = True
    
    def capture_calibration_data(self, cap, target_x, target_y):
        """Capture burst data for a calibration target"""
        # Create window for data capture animation
        window = np.zeros((self.WINDOW_HEIGHT, self.WINDOW_WIDTH, 3), dtype=np.uint8)
        cv2.namedWindow('Calibration', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('Calibration', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        captured_features = []
        
        for frame_idx in range(self.CAPTURE_FRAMES):
            ret, frame = cap.read()
            if not ret:
                continue
                
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            
            # Monitor user compliance during capture
            if not self.monitor_user_compliance(cap, results):
                print("Session aborted during calibration capture")
                return []
            
            # Extract features
            result = self.extract_iris_features(frame, results)
            if result:
                features, _ = result  # Unpack features and visualization data
                captured_features.append(features + [target_x, target_y])
            
            # Show capture animation (shrinking circle)
            window.fill(0)
            radius = int(30 - (frame_idx / self.CAPTURE_FRAMES) * 20)
            cv2.circle(window, (target_x, target_y), radius, (0, 0, 255), -1)
            cv2.putText(window, f"Capturing... {frame_idx + 1}/{self.CAPTURE_FRAMES}", 
                       (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            cv2.imshow('Calibration', window)
            cv2.waitKey(100)  # Brief pause between frames
        
        return captured_features
    
    def calibration_process(self, cap):
        """Main calibration process"""
        total_targets = len(self.calibration_targets)
        print(f"--- Starting Calibration ({total_targets} targets) ---")
        print("Sequence: 4 Corners → 4 Edges → 13 Grid Points (9 main + 4 intermediate)")
        
        for i, (target_x, target_y) in enumerate(self.calibration_targets):
            target_label = self.target_labels[i] if i < len(self.target_labels) else f"Target {i+1}"
            print(f"Calibration target {i + 1}/{total_targets}: {target_label} at ({target_x}, {target_y})")
            
            # Create calibration window
            window = np.zeros((self.WINDOW_HEIGHT, self.WINDOW_WIDTH, 3), dtype=np.uint8)
            cv2.namedWindow('Calibration', cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty('Calibration', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            
            # Draw target with enhanced visualization
            cv2.circle(window, (target_x, target_y), 30, (0, 0, 255), -1)
            cv2.circle(window, (target_x, target_y), 35, (255, 255, 255), 2)  # White outline
            
            # Enhanced text display
            cv2.putText(window, f"Click the red circle ({i + 1}/{total_targets})", 
                       (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(window, f"Target: {target_label}", 
                       (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
            
            # Show progress bar
            progress_width = 400
            progress_height = 20
            progress_x = 50
            progress_y = 120
            progress_fill = int((i / total_targets) * progress_width)
            
            cv2.rectangle(window, (progress_x, progress_y), (progress_x + progress_width, progress_y + progress_height), (100, 100, 100), 2)
            cv2.rectangle(window, (progress_x, progress_y), (progress_x + progress_fill, progress_y + progress_height), (0, 255, 0), -1)
            cv2.putText(window, f"Progress: {i}/{total_targets}", 
                       (progress_x + progress_width + 20, progress_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Set up mouse callback
            mouse_data = {'target': (target_x, target_y), 'clicked': False}
            cv2.setMouseCallback('Calibration', self.mouse_callback, mouse_data)
            
            # Wait for click with monitoring
            while not mouse_data['clicked']:
                # Check user compliance periodically
                ret, frame = cap.read()
                if ret:
                    frame = cv2.flip(frame, 1)
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = self.face_mesh.process(rgb_frame)
                    
                    if not self.monitor_user_compliance(cap, results):
                        print("Session aborted during calibration")
                        cv2.destroyAllWindows()
                        return False
                
                cv2.imshow('Calibration', window)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # Escape key
                    cv2.destroyAllWindows()
                    return False
            
            # Capture data
            captured_data = self.capture_calibration_data(cap, target_x, target_y)
            if not captured_data:  # If capture was interrupted
                print("Calibration data capture failed")
                cv2.destroyAllWindows()
                return False
                
            self.calibration_data.extend(captured_data)
            
            time.sleep(0.5)  # Brief pause between targets
        
        cv2.destroyAllWindows()
        return True
    
    def export_and_visualize_data(self):
        """Export data to CSV and create visualizations"""
        if not self.calibration_data:
            print("No calibration data to export!")
            return False
        
        # Create DataFrame
        df = pd.DataFrame(self.calibration_data, 
                         columns=['norm_x_L', 'norm_y_L', 'norm_x_R', 'norm_y_R', 'yaw', 'pitch', 'roll', 'target_x', 'target_y'])
        
        # Export to CSV
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"calibration_data_{timestamp}.csv"
        df.to_csv(filename, index=False)
        print(f"Data exported to {filename}")
        
        # Create separate visualizations
        
        # Figure 1: Histograms of features
        plt.figure(figsize=(12, 8))
        df[['norm_x_L', 'norm_y_L', 'norm_x_R', 'norm_y_R', 'yaw', 'pitch', 'roll']].hist(bins=20, alpha=0.7, figsize=(12, 8))
        plt.suptitle('Feature Histograms', fontsize=16)
        plt.tight_layout()
        plt.show()
        
        # Figure 2: Correlation heatmap
        plt.figure(figsize=(10, 8))
        correlation_matrix = df.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Matrix', fontsize=16)
        plt.tight_layout()
        plt.show()
        
        # Figure 3: Scatter plots
        features = ['norm_x_L', 'norm_y_L', 'norm_x_R', 'norm_y_R', 'yaw', 'pitch', 'roll']
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        fig.suptitle('Features vs Target Coordinates', fontsize=16)
        
        for i, feature in enumerate(features):
            row = i // 3
            col = i % 3
            ax = axes[row, col]
            
            ax.scatter(df[feature], df['target_x'], alpha=0.6, label='target_x', color='blue')
            ax.scatter(df[feature], df['target_y'], alpha=0.6, label='target_y', color='red')
            ax.set_xlabel(feature)
            ax.set_ylabel('Target Coordinates')
            ax.legend()
            ax.set_title(f'{feature} vs Targets')
        
        # Hide unused subplots
        for i in range(len(features), 9):
            row = i // 3
            col = i % 3
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        plt.show()
        
        return df
    
    def train_model(self, df):
        """Train XGBoost model"""
        print("--- Training Model ---")
        
        # Prepare data
        X = df[['norm_x_L', 'norm_y_L', 'norm_x_R', 'norm_y_R', 'yaw', 'pitch', 'roll']].values
        y = df[['target_x', 'target_y']].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        xgb_regressor = xgb.XGBRegressor(n_estimators=100, random_state=42)
        self.model = MultiOutputRegressor(xgb_regressor)
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        
        rmse_x = np.sqrt(mean_squared_error(y_test[:, 0], y_pred[:, 0]))
        rmse_y = np.sqrt(mean_squared_error(y_test[:, 1], y_pred[:, 1]))
        
        # Calculate mean prediction error in pixels
        distances = np.sqrt((y_test[:, 0] - y_pred[:, 0])**2 + (y_test[:, 1] - y_pred[:, 1])**2)
        mean_error = np.mean(distances)
        
        print(f"Model Evaluation:")
        print(f"RMSE X: {rmse_x:.2f} pixels")
        print(f"RMSE Y: {rmse_y:.2f} pixels")
        print(f"Mean Prediction Error: {mean_error:.2f} pixels")
        
        return True
    
    def real_time_prediction(self, cap):
        """Real-time gaze prediction"""
        print("--- Starting Real-Time Prediction ---")
        print("Press 'q' to quit")
        
        cv2.namedWindow('Gaze Prediction', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('Gaze Prediction', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            
            # Monitor user compliance during prediction
            if not self.monitor_user_compliance(cap, results):
                print("Session aborted during real-time prediction")
                break
            
            # Create prediction window
            window = np.zeros((self.WINDOW_HEIGHT, self.WINDOW_WIDTH, 3), dtype=np.uint8)
            
            # Extract features and predict
            result = self.extract_iris_features(frame, results)
            if result and self.model:
                features, (rotation_vector, translation_vector) = result
                prediction = self.model.predict([features[:7]])[0]  # Use all 7 features
                pred_x, pred_y = int(prediction[0]), int(prediction[1])
                
                # Apply smoothing with exponential moving average
                if self.smoothed_x is None:
                    self.smoothed_x, self.smoothed_y = pred_x, pred_y
                else:
                    # new_coord = old_coord * (1 - smoothing_factor) + predicted_coord * smoothing_factor
                    self.smoothed_x = self.smoothed_x * (1 - self.SMOOTHING_FACTOR) + pred_x * self.SMOOTHING_FACTOR
                    self.smoothed_y = self.smoothed_y * (1 - self.SMOOTHING_FACTOR) + pred_y * self.SMOOTHING_FACTOR
                
                # Convert to integers and ensure coordinates are within bounds
                smoothed_x_int = int(self.smoothed_x)
                smoothed_y_int = int(self.smoothed_y)
                smoothed_x_int = max(0, min(self.WINDOW_WIDTH - 1, smoothed_x_int))
                smoothed_y_int = max(0, min(self.WINDOW_HEIGHT - 1, smoothed_y_int))
                
                # Draw gaze point
                cv2.circle(window, (smoothed_x_int, smoothed_y_int), 15, (0, 0, 255), -1)
                
                # Draw head pose axis if available
                if rotation_vector is not None and translation_vector is not None:
                    # Create camera matrix for visualization
                    h, w = frame.shape[:2]
                    focal_length = w
                    center = (w/2, h/2)
                    camera_matrix = np.array([
                        [focal_length, 0, center[0]],
                        [0, focal_length, center[1]],
                        [0, 0, 1]
                    ], dtype=np.float32)
                    
                    self.draw_head_pose_axis(window, rotation_vector, translation_vector, camera_matrix)
                
                # Display head pose values
                yaw, pitch, roll = features[4], features[5], features[6]
                cv2.putText(window, f"Yaw: {yaw:.2f}", (50, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(window, f"Pitch: {pitch:.2f}", (50, 130), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(window, f"Roll: {roll:.2f}", (50, 160), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Add instructions
            cv2.putText(window, "Gaze Prediction - Press 'q' to quit", 
                       (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            cv2.imshow('Gaze Prediction', window)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        cv2.destroyAllWindows()
    
    def run_full_pipeline(self):
        """Execute the complete eye tracking pipeline"""
        print("=== Eye Tracking System ===")
        
        # Initialize camera once at the beginning
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.WINDOW_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.WINDOW_HEIGHT)
        
        try:
            # Step 1: User positioning
            if not self.user_positioning_phase(cap):
                print("User positioning cancelled.")
                return
            
            # Step 1.5: Show calibration target overview
            print("Showing calibration target overview...")
            self.visualize_all_calibration_targets()
            
            # Step 2: Calibration
            if not self.calibration_process(cap):
                print("Calibration cancelled.")
                return
            
            # Step 3: Data export and visualization
            df = self.export_and_visualize_data()
            if df is None or df.empty:
                print("Failed to export data.")
                return
            
            # Step 4: Model training
            if not self.train_model(df):
                print("Model training failed.")
                return
            
            # Step 5: Real-time prediction
            self.real_time_prediction(cap)
            
        finally:
            # Always release camera resources
            cap.release()
            cv2.destroyAllWindows()
            
        print("=== Eye Tracking Session Complete ===")

# Main execution
if __name__ == "__main__":
    tracker = EyeTracker()
    tracker.run_full_pipeline()
