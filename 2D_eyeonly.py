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
        self.BOUNDARY_LEFT = 0.3
        self.BOUNDARY_RIGHT = 0.7
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
        
        # Calibration targets (3x3 grid)
        self.calibration_targets = [
            (135, 90),   # Top-left
            (540, 90),   # Top-center
            (945, 90),   # Top-right
            (135, 360),  # Middle-left
            (540, 360),  # Middle-center
            (945, 360),  # Middle-right
            (135, 630),  # Bottom-left
            (540, 630),  # Bottom-center
            (945, 630)   # Bottom-right
        ]
        
    def extract_iris_features(self, image, landmarks):
        """Extract normalized iris position features"""
        if not landmarks.multi_face_landmarks:
            return None
            
        face_landmarks = landmarks.multi_face_landmarks[0]
        h, w = image.shape[:2]
        
        # Get key facial landmarks
        # Left eye corners and iris
        left_eye_left = face_landmarks.landmark[33]   # Left eye left corner
        left_eye_right = face_landmarks.landmark[133] # Left eye right corner
        left_iris = face_landmarks.landmark[468]      # Left iris center
        
        # Right eye corners and iris
        right_eye_left = face_landmarks.landmark[362]  # Right eye left corner
        right_eye_right = face_landmarks.landmark[263] # Right eye right corner
        right_iris = face_landmarks.landmark[473]      # Right iris center
        
        # Normalize iris positions relative to eye corners
        # Left eye normalization
        left_eye_width = abs(left_eye_right.x - left_eye_left.x)
        left_eye_height = max(abs(face_landmarks.landmark[145].y - face_landmarks.landmark[159].y), 0.01)
        
        norm_x_L = (left_iris.x - left_eye_left.x) / max(left_eye_width, 0.01)
        norm_y_L = (left_iris.y - left_eye_left.y) / left_eye_height
        
        # Right eye normalization
        right_eye_width = abs(right_eye_right.x - right_eye_left.x)
        right_eye_height = max(abs(face_landmarks.landmark[374].y - face_landmarks.landmark[386].y), 0.01)
        
        norm_x_R = (right_iris.x - right_eye_left.x) / max(right_eye_width, 0.01)
        norm_y_R = (right_iris.y - right_eye_left.y) / right_eye_height
        
        return [norm_x_L, norm_y_L, norm_x_R, norm_y_R]
    
    def is_face_in_boundary(self, landmarks, image_width, image_height):
        """Check if face is within the positioning boundary"""
        if not landmarks.multi_face_landmarks:
            return False
            
        face_landmarks = landmarks.multi_face_landmarks[0]
        
        # Get face bounding box
        x_coords = [landmark.x for landmark in face_landmarks.landmark]
        y_coords = [landmark.y for landmark in face_landmarks.landmark]
        
        face_left = min(x_coords)
        face_right = max(x_coords)
        face_top = min(y_coords)
        face_bottom = max(y_coords)
        
        # Check if entire face is within boundary
        return (face_left >= self.BOUNDARY_LEFT and 
                face_right <= self.BOUNDARY_RIGHT and
                face_top >= self.BOUNDARY_TOP and 
                face_bottom <= self.BOUNDARY_BOTTOM)
    
    def draw_boundary_box(self, image, is_positioned_correctly):
        """Draw the positioning boundary box"""
        h, w = image.shape[:2]
        
        left = int(self.BOUNDARY_LEFT * w)
        right = int(self.BOUNDARY_RIGHT * w)
        top = int(self.BOUNDARY_TOP * h)
        bottom = int(self.BOUNDARY_BOTTOM * h)
        
        color = (0, 255, 0) if is_positioned_correctly else (0, 0, 255)  # Green if positioned, red if not
        thickness = 3
        
        cv2.rectangle(image, (left, top), (right, bottom), color, thickness)
        
        # Add text instructions
        text = "Position your face in the box and press ENTER" if not is_positioned_correctly else "Good! Press ENTER to start calibration"
        cv2.putText(image, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
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
        """Guide user to correct positioning"""
        cv2.namedWindow('User Positioning', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('User Positioning', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        print("--- User Positioning Phase ---")
        print("Position your face within the boundary box and press ENTER")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame = cv2.flip(frame, 1)  # Mirror the image
            frame_resized = cv2.resize(frame, (self.WINDOW_WIDTH, self.WINDOW_HEIGHT))
            rgb_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            
            results = self.face_mesh.process(rgb_frame)
            
            # Check positioning
            is_positioned = self.is_face_in_boundary(results, self.WINDOW_WIDTH, self.WINDOW_HEIGHT)
            
            # Draw boundary box and face mesh
            self.draw_boundary_box(frame_resized, is_positioned)
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame_resized, face_landmarks, self.mp_face_mesh.FACEMESH_IRISES,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)
                    )
            
            cv2.imshow('User Positioning', frame_resized)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 13 and is_positioned:  # Enter key
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
            features = self.extract_iris_features(frame, results)
            if features:
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
        print("--- Starting Calibration ---")
        
        for i, (target_x, target_y) in enumerate(self.calibration_targets):
            print(f"Calibration target {i + 1}/9 at ({target_x}, {target_y})")
            
            # Create calibration window
            window = np.zeros((self.WINDOW_HEIGHT, self.WINDOW_WIDTH, 3), dtype=np.uint8)
            cv2.namedWindow('Calibration', cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty('Calibration', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            
            # Draw target
            cv2.circle(window, (target_x, target_y), 30, (0, 0, 255), -1)
            cv2.putText(window, f"Click the red circle ({i + 1}/9)", 
                       (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
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
                         columns=['norm_x_L', 'norm_y_L', 'norm_x_R', 'norm_y_R', 'target_x', 'target_y'])
        
        # Export to CSV
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"calibration_data_{timestamp}.csv"
        df.to_csv(filename, index=False)
        print(f"Data exported to {filename}")
        
        # Create separate visualizations
        
        # Figure 1: Histograms of features
        plt.figure(figsize=(12, 8))
        df[['norm_x_L', 'norm_y_L', 'norm_x_R', 'norm_y_R']].hist(bins=20, alpha=0.7, figsize=(12, 8))
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
        features = ['norm_x_L', 'norm_y_L', 'norm_x_R', 'norm_y_R']
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Features vs Target Coordinates', fontsize=16)
        
        for i, feature in enumerate(features):
            row = i // 2
            col = i % 2
            ax = axes[row, col]
            
            ax.scatter(df[feature], df['target_x'], alpha=0.6, label='target_x', color='blue')
            ax.scatter(df[feature], df['target_y'], alpha=0.6, label='target_y', color='red')
            ax.set_xlabel(feature)
            ax.set_ylabel('Target Coordinates')
            ax.legend()
            ax.set_title(f'{feature} vs Targets')
        
        plt.tight_layout()
        plt.show()
        
        return df
    
    def train_model(self, df):
        """Train XGBoost model"""
        print("--- Training Model ---")
        
        # Prepare data
        X = df[['norm_x_L', 'norm_y_L', 'norm_x_R', 'norm_y_R']].values
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
            features = self.extract_iris_features(frame, results)
            if features and self.model:
                prediction = self.model.predict([features[:4]])[0]
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
