# feature_extraction.py
# This file contains functions dedicated to extracting features from the raw camera feed
# and MediaPipe landmarks. These features are then used to train the machine learning model.
# The functions here are designed to be self-contained and focus purely on data processing.

import numpy as np
import cv2

def calculate_head_pose(image, landmarks):
    """
    Calculates the head pose (yaw, pitch, roll) using the PnP algorithm.

    Args:
        image: The camera frame.
        landmarks: The MediaPipe face landmarks result.

    Returns:
        A tuple containing:
        - yaw (float): Normalized head rotation around the Y-axis.
        - pitch (float): Normalized head rotation around the X-axis.
        - roll (float): Normalized head rotation around the Z-axis.
        - rotation_vector (np.array): The rotation vector from solvePnP.
        - translation_vector (np.array): The translation vector from solvePnP.
    """
    if not landmarks.multi_face_landmarks:
        return None, None, None, None, None

    face_landmarks = landmarks.multi_face_landmarks[0]
    h, w = image.shape[:2]

    # Define a standard 3D model of a face
    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -330.0, -65.0),        # Chin
        (-225.0, 170.0, -135.0),     # Left eye left corner
        (225.0, 170.0, -135.0),      # Right eye right corner
        (-150.0, -150.0, -125.0),    # Left mouth corner
        (150.0, -150.0, -125.0)      # Right mouth corner
    ], dtype=np.float64)

    # Get corresponding 2D points from the camera feed
    image_points = np.array([
        (face_landmarks.landmark[1].x * w, face_landmarks.landmark[1].y * h),      # Nose tip
        (face_landmarks.landmark[152].x * w, face_landmarks.landmark[152].y * h),   # Chin
        (face_landmarks.landmark[263].x * w, face_landmarks.landmark[263].y * h),   # Left eye corner
        (face_landmarks.landmark[33].x * w, face_landmarks.landmark[33].y * h),     # Right eye corner
        (face_landmarks.landmark[287].x * w, face_landmarks.landmark[287].y * h),   # Left mouth corner
        (face_landmarks.landmark[57].x * w, face_landmarks.landmark[57].y * h)      # Right mouth corner
    ], dtype=np.float64)

    # Estimate camera matrix
    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)

    # Solve for pose
    (success, rotation_vector, translation_vector) = cv2.solvePnP(
        model_points, image_points, camera_matrix, np.zeros((4, 1), dtype=np.float64),
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return None, None, None, None, None

    # Convert rotation vector to rotation matrix to get Euler angles
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
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

    # Convert to degrees and normalize for use as features
    pitch = np.clip(x * 180.0 / np.pi, -90, 90) / 90.0
    yaw = np.clip(y * 180.0 / np.pi, -90, 90) / 90.0
    roll = np.clip(z * 180.0 / np.pi, -45, 45) / 45.0
    
    return yaw, pitch, roll, rotation_vector, translation_vector

def calculate_face_bounding_box(landmarks):
    """Calculates the bounding box of the user's face from landmarks."""
    if not landmarks.multi_face_landmarks:
        return None
        
    face_landmarks = landmarks.multi_face_landmarks[0]
    
    face_boundary_indices = [
        10, 151, 9, 175, 234, 454, 132, 361, 172, 136, 150, 
        149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 323
    ]
    
    face_x_coords = [face_landmarks.landmark[i].x for i in face_boundary_indices]
    face_y_coords = [face_landmarks.landmark[i].y for i in face_boundary_indices]
    
    return {
        'left': min(face_x_coords), 'right': max(face_x_coords),
        'top': min(face_y_coords), 'bottom': max(face_y_coords)
    }

def extract_all_features(image, landmarks):
    """
    Main feature extraction function. It calculates iris positions and head pose.

    Args:
        image: The camera frame.
        landmarks: The MediaPipe face landmarks result.

    Returns:
        A tuple containing:
        - features (list or None): A list of 7 features [norm_x_L, norm_y_L, 
          norm_x_R, norm_y_R, yaw, pitch, roll] for the ML model.
        - visualization_data (tuple or None): Data needed for drawing the head 
          pose axis (rotation_vector, translation_vector).
    """
    if not landmarks.multi_face_landmarks:
        return None, None

    face_landmarks = landmarks.multi_face_landmarks[0]
    
    # 1. Calculate Face Bounding Box
    face_box = calculate_face_bounding_box(landmarks)
    if face_box is None:
        return None, None

    face_width = face_box['right'] - face_box['left']
    face_height = face_box['bottom'] - face_box['top']

    # 2. Extract and Normalize Iris Positions
    left_iris = face_landmarks.landmark[473]
    right_iris = face_landmarks.landmark[468]

    norm_x_L = (left_iris.x - face_box['left']) / max(face_width, 1e-6)
    norm_y_L = (left_iris.y - face_box['top']) / max(face_height, 1e-6)
    norm_x_R = (right_iris.x - face_box['left']) / max(face_width, 1e-6)
    norm_y_R = (right_iris.y - face_box['top']) / max(face_height, 1e-6)

    # 3. Calculate Head Pose
    yaw, pitch, roll, rvec, tvec = calculate_head_pose(image, landmarks)
    if yaw is None:
        yaw, pitch, roll = 0.0, 0.0, 0.0 # Default values if PnP fails

    # 4. Combine all features
    features = [
        np.clip(norm_x_L, 0, 1), np.clip(norm_y_L, 0, 1),
        np.clip(norm_x_R, 0, 1), np.clip(norm_y_R, 0, 1),
        yaw, pitch, roll
    ]
    
    visualization_data = (rvec, tvec)
    
    return features, visualization_data