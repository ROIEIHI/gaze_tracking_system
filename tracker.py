# tracker_core.py
# This file contains the main EyeTracker class, which orchestrates the entire
# eye-tracking pipeline. It manages the application state and calls functions
# from the various utility modules (config, feature_ex, ui_utils, model_utils).

import cv2
import mediapipe as mp
import time
import datetime
import pandas as pd
import math
import numpy as np

# Import the custom utility modules
import config
import feature_ex
import model_utils
import ui_utils

class EyeTracker:
    """
    The main class that manages the eye-tracking application workflow.
    """
    def __init__(self):
        """Initializes the EyeTracker, setting up MediaPipe and application state."""
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils

        # Data storage and model
        self.calibration_data = []
        self.model = None
        self.smoothed_x = None
        self.smoothed_y = None

        # State tracking for UI and monitoring
        self.session_paused = False
        self.is_currently_in_boundary = False
        self.boundary_exit_timestamp = None
        self.boundary_entry_timestamp = None

    def _mouse_callback(self, event, x, y, flags, param):
        """Internal mouse callback for handling clicks on calibration targets."""
        if event == cv2.EVENT_LBUTTONDOWN:
            target_x, target_y = param['target']
            # Check if the click is within the target circle's radius
            if math.sqrt((x - target_x)**2 + (y - target_y)**2) <= 30:
                param['clicked'] = True

    def _check_face_detection_quality(self, landmarks):
        """A simple check to ensure key landmarks are present for quality tracking."""
        if not landmarks.multi_face_landmarks:
            return False
        # A simple proxy for quality is the presence of landmarks.
        # A more robust check could analyze landmark stability over frames.
        return True

    def _monitor_user_compliance(self, cap, results):
        """Monitors user position and face detection quality, pausing if non-compliant."""
        face_quality_ok = self._check_face_detection_quality(results)
        face_in_boundary = feature_ex.calculate_face_bounding_box(results) is not None

        if not face_quality_ok:
            print("⚠️ WARNING: Face detection quality insufficient.")
            self.session_paused = True
            return ui_utils.show_warning_window(self, cap, 'face_detection')
        
        if not face_in_boundary:
            print("⚠️ WARNING: User moved outside boundary area.")
            self.session_paused = True
            return ui_utils.show_warning_window(self, cap, 'boundary_exit')
            
        return True

    def user_positioning_phase(self, cap):
        """Guides the user to the correct position with visual feedback and hysteresis."""
        print("--- User Positioning Phase ---")
        # Reset hysteresis state for this phase
        self.is_currently_in_boundary = False
        self.boundary_exit_timestamp = None
        self.boundary_entry_timestamp = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            
            is_geometrically_in = ui_utils.is_face_in_boundary(results)

            # Apply hysteresis logic for smooth UI feedback
            is_positioned_for_ui, can_proceed = ui_utils.apply_hysteresis(
                self, is_geometrically_in
            )

            # Draw UI elements
            ui_utils.draw_boundary_box(frame, is_positioned_for_ui, can_proceed)
            ui_utils.draw_face_bounding_box(frame, results)
            
            cv2.imshow('User Positioning', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 13 and can_proceed:  # Enter key
                break
            elif key == 27:  # Escape key
                cv2.destroyAllWindows()
                return False
                
        cv2.destroyAllWindows()
        return True

    def _capture_calibration_data(self, cap, target_x, target_y):
        """Captures a burst of frames for a single calibration target."""
        captured_features = []
        for frame_idx in range(config.CAPTURE_FRAMES):
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            
            if not self._monitor_user_compliance(cap, results):
                return [] # Abort capture if user becomes non-compliant

            result = feature_ex.extract_all_features(frame, results)
            if result[0]:
                features, _ = result
                captured_features.append(features + [target_x, target_y])
            
            ui_utils.draw_capture_animation(target_x, target_y, frame_idx)
            cv2.waitKey(100)
        
        return captured_features

    def calibration_process(self, cap):
        """Manages the entire calibration sequence, showing targets and capturing data."""
        print(f"--- Starting Calibration ({len(config.CALIBRATION_TARGETS)} targets) ---")
        for i, (target_x, target_y) in enumerate(config.CALIBRATION_TARGETS):
            target_label = config.TARGET_LABELS[i]
            print(f"Target {i+1}/{len(config.CALIBRATION_TARGETS)}: {target_label}")

            mouse_data = {'target': (target_x, target_y), 'clicked': False}
            
            # Create the window first, then set mouse callback
            ui_utils.draw_calibration_ui(i, target_x, target_y, target_label)
            cv2.setMouseCallback('Calibration', self._mouse_callback, mouse_data)
            
            # Wait for user to click the target
            while not mouse_data['clicked']:
                ui_utils.draw_calibration_ui(i, target_x, target_y, target_label)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    cv2.destroyAllWindows()
                    return False

            captured_data = self._capture_calibration_data(cap, target_x, target_y)
            if not captured_data:
                return False # Abort if capture failed
            self.calibration_data.extend(captured_data)
            time.sleep(0.5)
            
        cv2.destroyAllWindows()
        return True

    def _export_data(self):
        """Exports the collected calibration data to a timestamped CSV file."""
        if not self.calibration_data:
            print("No calibration data to export.")
            return None
        
        columns = ['norm_x_L', 'norm_y_L', 'norm_x_R', 'norm_y_R', 'yaw', 'pitch', 'roll', 'target_x', 'target_y']
        df = pd.DataFrame(self.calibration_data, columns=columns)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"calibration_data_{timestamp}.csv"
        df.to_csv(filename, index=False)
        print(f"Calibration data exported to {filename}")
        return df

    def real_time_prediction(self, cap):
        """Runs the main real-time gaze prediction loop."""
        print("--- Starting Real-Time Prediction ---")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)

            if not self._monitor_user_compliance(cap, results):
                break

            prediction_window = np.zeros((config.WINDOW_HEIGHT, config.WINDOW_WIDTH, 3), dtype=np.uint8)
            
            result = feature_ex.extract_all_features(frame, results)
            if result[0] and self.model:
                features, viz_data = result
                prediction = self.model.predict([features])[0]
                
                # Apply smoothing
                if self.smoothed_x is None:
                    self.smoothed_x, self.smoothed_y = prediction[0], prediction[1]
                else:
                    self.smoothed_x = self.smoothed_x * (1 - config.SMOOTHING_FACTOR) + prediction[0] * config.SMOOTHING_FACTOR
                    self.smoothed_y = self.smoothed_y * (1 - config.SMOOTHING_FACTOR) + prediction[1] * config.SMOOTHING_FACTOR
                
                # Draw all prediction UI elements
                ui_utils.draw_prediction_ui(prediction_window, self.smoothed_x, self.smoothed_y, features, viz_data, frame.shape)
            
            cv2.imshow('Gaze Prediction', prediction_window)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cv2.destroyAllWindows()

    def run_full_pipeline(self):
        """Executes the complete eye-tracking pipeline from start to finish."""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera.")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.WINDOW_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.WINDOW_HEIGHT)
        
        try:
            if not self.user_positioning_phase(cap):
                return
            
            
            if not self.calibration_process(cap):
                return

            df = self._export_data()
            if df is None:
                return
            
            model_utils.visualize_data(df)
            
            self.model = model_utils.train_model(df)
            if self.model is None:
                return
            
            self.real_time_prediction(cap)

        finally:
            print("=== Eye Tracking Session Complete ===")
            cap.release()
            cv2.destroyAllWindows()
