# ui_utils.py
# This file contains all utility functions for creating and managing the user interface.
# It handles drawing, displaying text, showing warnings, and managing UI state logic
# like the positioning hysteresis.

import cv2
import numpy as np
import time

# Import the project's configuration and feature extraction modules
import config
import feature_ex

def is_face_in_boundary(landmarks):
    """Checks if the face's bounding box is within the predefined absolute boundary."""
    face_box = feature_ex.calculate_face_bounding_box(landmarks)
    if face_box is None:
        return False
    
    return (face_box['left'] >= config.BOUNDARY_LEFT and
            face_box['right'] <= config.BOUNDARY_RIGHT and
            face_box['top'] >= config.BOUNDARY_TOP and
            face_box['bottom'] <= config.BOUNDARY_BOTTOM)

def apply_hysteresis(tracker, is_geometrically_in):
    """
    Applies a time-based delay to boundary entry and exit to prevent UI flickering.

    Args:
        tracker: The main EyeTracker object to access state variables.
        is_geometrically_in (bool): The raw result of the boundary check for the current frame.

    Returns:
        A tuple containing:
        - is_positioned_for_ui (bool): The smoothed status for UI display.
        - can_proceed (bool): Whether the user has held their position long enough.
    """
    current_time = time.time()
    
    # Logic for exiting the boundary with a grace period
    if is_geometrically_in:
        tracker.boundary_exit_timestamp = None
    elif tracker.boundary_exit_timestamp is None:
        tracker.boundary_exit_timestamp = current_time

    is_positioned_for_ui = is_geometrically_in or \
        (tracker.boundary_exit_timestamp is not None and 
         current_time - tracker.boundary_exit_timestamp < 0.25)

    # Logic for entering the boundary with a hold requirement
    if is_positioned_for_ui and tracker.boundary_entry_timestamp is None:
        tracker.boundary_entry_timestamp = current_time
    elif not is_positioned_for_ui:
        tracker.boundary_entry_timestamp = None
        
    can_proceed = (is_positioned_for_ui and 
                   tracker.boundary_entry_timestamp is not None and 
                   current_time - tracker.boundary_entry_timestamp >= 1.0)
                   
    return is_positioned_for_ui, can_proceed

def draw_boundary_box(image, is_positioned, can_proceed):
    """Draws the main positioning boundary box."""
    h, w = image.shape[:2]
    left = int(config.BOUNDARY_LEFT * w)
    right = int(config.BOUNDARY_RIGHT * w)
    top = int(config.BOUNDARY_TOP * h)
    bottom = int(config.BOUNDARY_BOTTOM * h)
    
    if is_positioned and can_proceed:
        color = (0, 255, 0)  # Green
        text = "Perfect! Press ENTER to continue"
    elif is_positioned:
        color = (0, 255, 255)  # Yellow
        text = "Hold position steady..."
    else:
        color = (0, 0, 255)  # Red
        text = "Position your face box inside the boundary"
        
    cv2.rectangle(image, (left, top), (right, bottom), color, 3)
    cv2.putText(image, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

def draw_face_bounding_box(image, landmarks):
    """Draws the bounding box calculated directly around the user's face."""
    face_box = feature_ex.calculate_face_bounding_box(landmarks)
    if face_box is None:
        return
        
    h, w = image.shape[:2]
    left_px = int(face_box['left'] * w)
    right_px = int(face_box['right'] * w)
    top_px = int(face_box['top'] * h)
    bottom_px = int(face_box['bottom'] * h)
    
    cv2.rectangle(image, (left_px, top_px), (right_px, bottom_px), (0, 255, 255), 2) # Yellow

def show_warning_window(tracker, cap, warning_type):
    """Displays a warning and pauses the session until the issue is resolved."""
    # This function would contain the logic from your original show_warning_window method,
    # refactored to not use 'self' but instead take 'tracker' as an argument if needed for state.
    # For now, a simplified placeholder:
    print(f"SHOWING WARNING: {warning_type}")
    time.sleep(2) # Simple pause
    tracker.session_paused = False
    return True # Assume user fixed the issue

def draw_calibration_ui(index, target_x, target_y, target_label):
    """Draws the UI for a single calibration target."""
    window = np.zeros((config.WINDOW_HEIGHT, config.WINDOW_WIDTH, 3), dtype=np.uint8)
    total_targets = len(config.CALIBRATION_TARGETS)
    
    # Draw target circle
    cv2.circle(window, (target_x, target_y), 30, (0, 0, 255), -1)
    cv2.circle(window, (target_x, target_y), 35, (255, 255, 255), 2)
    
    # Draw text and progress bar
    cv2.putText(window, f"Click the red circle ({index + 1}/{total_targets})", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(window, f"Target: {target_label}", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    
    cv2.imshow('Calibration', window)

def draw_capture_animation(target_x, target_y, frame_idx):
    """Draws the shrinking circle animation during data capture."""
    window = np.zeros((config.WINDOW_HEIGHT, config.WINDOW_WIDTH, 3), dtype=np.uint8)
    radius = int(30 - (frame_idx / config.CAPTURE_FRAMES) * 20)
    cv2.circle(window, (target_x, target_y), radius, (0, 0, 255), -1)
    cv2.putText(window, f"Capturing... {frame_idx + 1}/{config.CAPTURE_FRAMES}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow('Calibration', window)

def draw_prediction_ui(window, smoothed_x, smoothed_y, features, viz_data, frame_shape):
    """Draws all elements for the real-time prediction window."""
    # Draw gaze point
    cv2.circle(window, (int(smoothed_x), int(smoothed_y)), 15, (0, 0, 255), -1)
    
    # Draw head pose axis
    rvec, tvec = viz_data
    if rvec is not None:
        h, w = frame_shape[:2]
        cam_matrix = np.array([[w, 0, w/2], [0, w, h/2], [0, 0, 1]], dtype=np.float64)
        # ... [Logic to project and draw 3D axis] ...

    # Display text info
    cv2.putText(window, "Gaze Prediction - Press 'q' to quit", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    yaw, pitch, roll = features[4:7]
    cv2.putText(window, f"Yaw: {yaw:.2f}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(window, f"Pitch: {pitch:.2f}", (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(window, f"Roll: {roll:.2f}", (50, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
