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
from PIL import ImageDraw, Image
from text_layout import build_reading_sequence, draw_instructions, load_text_font, render_page
from calibration import draw_face_box, face_position_problem
import os
import re
import threading
from queue import Queue, Full
import json

class VideoRecorder:
    """Video recording helper class for gaze tracking sessions"""
    
    def __init__(self, output_dir: str, session_id: str = "session"):
        """Initialize video recorder"""
        self.output_dir = output_dir
        self.session_id = session_id
        self.camera_writer = None
        self.overlay_writer = None
        self.is_recording = False
        self.camera_path = None
        self.overlay_path = None
        self.fps = VIDEO_FPS
        self.camera_resolution = f"{CAMERA_WIDTH}x{CAMERA_HEIGHT}"
        
        # Encoding runs on its own thread so it never holds up gaze tracking
        self.frame_queue = Queue(maxsize=VIDEO_BUFFER_SIZE)
        self.write_thread = None
        self.frames_dropped = 0
        
    def start_recording(self, fps: int = VIDEO_FPS) -> tuple:
        """Start video recording and return file paths"""
        if not VIDEO_RECORDING_ENABLED:
            return None, None
            
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Define output paths
            if RECORD_CAMERA_FEED:
                self.camera_path = os.path.join(self.output_dir, f"camera_feed_{self.session_id}_{timestamp}.mp4")
            if RECORD_OVERLAY_DISPLAY:
                self.overlay_path = os.path.join(self.output_dir, f"overlay_display_{self.session_id}_{timestamp}.mp4")
            
            # Initialize video writers
            fourcc = cv2.VideoWriter_fourcc(*VIDEO_CODEC)
            
            # The camera writer opens on the first frame, see record_frame
            self.fps = fps
                
            if RECORD_OVERLAY_DISPLAY and self.overlay_path:
                self.overlay_writer = cv2.VideoWriter(
                    self.overlay_path, fourcc, fps, (SCREEN_WIDTH, SCREEN_HEIGHT)
                )
            
            self.is_recording = True
            self.write_thread = threading.Thread(target=self._write_frames, daemon=True)
            self.write_thread.start()
            print(f"Video recording started:")
            if self.camera_path:
                print(f"  Camera feed: {os.path.basename(self.camera_path)}")
            if self.overlay_path:
                print(f"  Overlay display: {os.path.basename(self.overlay_path)}")
            
            return self.camera_path, self.overlay_path
            
        except Exception as e:
            print(f"Error starting video recording: {e}")
            self.disable_recording()
            return None, None
    
    def record_frame(self, camera_frame=None, overlay_frame=None):
        """Queue frames for the encoder thread. This never waits: if encoding falls behind,
        frames are lost from the video rather than from gaze tracking"""
        if not self.is_recording:
            return

        try:
            self.frame_queue.put_nowait((camera_frame, overlay_frame))
        except Full:
            self.frames_dropped += 1

    def _write_frames(self):
        """Encoder thread: writes queued frames until stop_recording queues None"""
        while True:
            frames = self.frame_queue.get()
            if frames is None:
                break

            try:
                self._write_pair(*frames)
            except Exception as e:
                print(f"Error recording frame: {e}")
                # Continue without recording rather than crash

    def _write_pair(self, camera_frame, overlay_frame):
        """Encode one queued pair of frames"""
        if camera_frame is not None and self.camera_path and RECORD_CAMERA_FEED:
            if self.camera_writer is None:
                # Size it from a real frame: cameras can ignore CAMERA_WIDTH/HEIGHT, and
                # VideoWriter silently drops every frame that doesn't match its size
                height, width = camera_frame.shape[:2]
                self.camera_resolution = f"{width}x{height}"
                self.camera_writer = cv2.VideoWriter(
                    self.camera_path, cv2.VideoWriter_fourcc(*VIDEO_CODEC), self.fps, (width, height)
                )
            self.camera_writer.write(camera_frame)

        if overlay_frame is not None and self.overlay_writer and RECORD_OVERLAY_DISPLAY:
            self.overlay_writer.write(overlay_frame)

    def stop_recording(self) -> dict:
        """Stop recording and return metadata"""
        if not self.is_recording:
            return {}
            
        try:
            # Stop recording flag
            self.is_recording = False

            # Finish what is queued, but a stuck encoder must not block the data export
            self.frame_queue.put(None)
            self.write_thread.join(timeout=30)

            # Release video writers
            if self.camera_writer:
                self.camera_writer.release()
                self.camera_writer = None
                
            if self.overlay_writer:
                self.overlay_writer.release()
                self.overlay_writer = None
            
            # Create metadata
            metadata = {
                'recording_ended': datetime.now().isoformat(),
                'camera_feed_path': self.camera_path,
                'overlay_display_path': self.overlay_path,
                'fps': VIDEO_FPS,
                'codec': VIDEO_CODEC,
                'camera_resolution': self.camera_resolution,
                'overlay_resolution': f"{SCREEN_WIDTH}x{SCREEN_HEIGHT}",
                'frames_dropped': self.frames_dropped
            }
            
            # Save metadata file
            if self.camera_path or self.overlay_path:
                metadata_path = os.path.join(self.output_dir, f"video_metadata_{self.session_id}.json")
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                print(f"Video metadata saved: {os.path.basename(metadata_path)}")
            
            if self.frames_dropped:
                print(f"Warning: {self.frames_dropped} video frames dropped, encoding fell behind")
            print("Video recording stopped successfully")
            return metadata
            
        except Exception as e:
            print(f"Error stopping video recording: {e}")
            return {}
    
    def disable_recording(self):
        """Disable recording due to errors"""
        self.is_recording = False
        print("Video recording disabled due to errors")

class TextReadingGazePredictor:
    """Enhanced gaze predictor with text reading analysis capabilities"""
    
    def __init__(self, model_path: str = None, output_dir: str = None):
        """Initialize the enhanced gaze predictor"""
        print("Initializing Text Reading Gaze Predictor...")
        
        # Set output directory
        # Set output directory structure
        base_output_dir = output_dir if output_dir is not None else USER_DATA_DIR
        
        # Generate session timestamp
        self.session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir_name = f"session_{self.session_timestamp}"
        
        # Create session directories
        self.session_dir = os.path.join(base_output_dir, session_dir_name)
        self.analysis_dir = os.path.join(self.session_dir, "analysis")
        self.recordings_dir = os.path.join(self.session_dir, "recordings")
        
        # Ensure directories exist
        os.makedirs(self.analysis_dir, exist_ok=True)
        os.makedirs(self.recordings_dir, exist_ok=True)
        
        # Set main output dir to analysis for CSV exports
        self.output_dir = self.analysis_dir
        print(f"Session directory created: {self.session_dir}")
        
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
        self.total_pages = 0
        self.text_pages = []
        self.word_positions = []
        self.page_image = None
        self.cached_page = None
        
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
        
        # Video recording initialization
        if VIDEO_RECORDING_ENABLED:
            # Use the recordings directory and existing timestamp
            self.video_recorder = VideoRecorder(self.recordings_dir, self.session_timestamp)
            self.recording_paths = None
            self.video_metadata = None
        else:
            self.video_recorder = None
        
        # Initialize text pages
        self.reading_sequence = []
        self.probe_answers = {}
        self._create_reading_sequence()
        
        # Mouse state for probes
        self.mouse_x = 0
        self.mouse_y = 0
        self.mouse_clicked = False

        # Position check during reading
        self.face_problem = ""              # Why the face is out of position on this frame, "" when it's fine
        self.out_of_position_since = None
        self.in_position_since = None
        self.paused = False
        self.position_event = None          # The stretch out of position happening now
        self.position_events = []
        
    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for probe interaction"""
        if event == cv2.EVENT_MOUSEMOVE:
            self.mouse_x = x
            self.mouse_y = y
        elif event == cv2.EVENT_LBUTTONDOWN:
            self.mouse_x = x
            self.mouse_y = y
            self.mouse_clicked = True
    
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
    
    def _create_reading_sequence(self):
        """Create the sequence of text pages, questions and the closing message from config"""
        print("Creating reading sequence from config...")

        self.reading_sequence = build_reading_sequence()
        self.total_pages = sum(1 for step in self.reading_sequence if step['type'] == 'page')

        print(f"Created sequence with {len(self.reading_sequence)} steps ({self.total_pages} visual pages)")

    def _render_text_page(self) -> np.ndarray:
        """Render current text page. A page is drawn once and reused, callers get a copy to draw on"""
        if self.current_page >= len(self.reading_sequence):
            return np.full((SCREEN_HEIGHT, SCREEN_WIDTH, 3), TEXT_BACKGROUND, dtype=np.uint8)

        step = self.reading_sequence[self.current_page]

        if step['type'] == 'probe':
            background = np.full((SCREEN_HEIGHT, SCREEN_WIDTH, 3), TEXT_BACKGROUND, dtype=np.uint8)
            return self._render_probe(step, background)

        if self.cached_page != self.current_page:
            if step['type'] == 'page':
                self.page_image, self.word_positions = render_page(step['content'], self.current_page, step['first_line'], step['first_word'])
            else:
                background = np.full((SCREEN_HEIGHT, SCREEN_WIDTH, 3), TEXT_BACKGROUND, dtype=np.uint8)
                self.page_image = self._render_question(step, background)
                self.word_positions = []
            self._draw_navigation_info(self.page_image)
            self.cached_page = self.current_page

        return self.page_image.copy()

    def _update_position(self, shown_step):
        """Track the face against the box. Reading pauses after OOB_PAUSE_AFTER_SECONDS out of it, on text pages only,
        and resumes after OOB_RESUME_AFTER_SECONDS steady inside. Stretches out of the box are logged once reading starts"""
        now = time.time()

        if self.face_problem:
            self.in_position_since = None
            if self.out_of_position_since is None:
                self.out_of_position_since = now
                self._open_position_event(shown_step, now)
            if not self.paused and shown_step['type'] == 'page' and now - self.out_of_position_since >= OOB_PAUSE_AFTER_SECONDS:
                self.paused = True
                if self.position_event:
                    self.position_event['Paused'] = True
                print("Reading paused, the face is out of position")
        else:
            if self.out_of_position_since is not None:
                self.out_of_position_since = None
                self._close_position_event(now)
            if self.in_position_since is None:
                self.in_position_since = now
            if self.paused and now - self.in_position_since >= OOB_RESUME_AFTER_SECONDS:
                self.paused = False
                self.smoothed_x = self.smoothed_y = None  # Start the gaze smoothing fresh
                print("Reading resumed")

    def _open_position_event(self, shown_step, now):
        """Start logging a stretch out of position, once the reading clock is running"""
        if self.session_start_time is None:
            return
        self.position_event = {
            'Start_ms': round((now - self.session_start_time) * 1000, 2),
            'Duration_ms': None,
            'Screen': 'text' if shown_step['type'] == 'page' else shown_step['type'],
            'Text_Number': shown_step.get('text'),
            'Part_Number': shown_step.get('part'),
            'Page_Number': shown_step.get('page_number'),
            'Reason': self.face_problem,
            'Paused': self.paused,
            'started': now
        }

    def _close_position_event(self, now=None):
        """Finish the current stretch out of position and keep it for the events file"""
        if self.position_event is None:
            return
        event = self.position_event
        event['Duration_ms'] = round(((now or time.time()) - event.pop('started')) * 1000, 2)
        self.position_events.append(event)
        self.position_event = None

    def _render_reposition_screen(self, frame):
        """Live camera view with the face box, shown while reading is paused"""
        h, w = frame.shape[:2]
        scale = min(SCREEN_WIDTH / w, (SCREEN_HEIGHT - 110) / h)
        preview = cv2.resize(frame, (int(w * scale), int(h * scale)))
        draw_face_box(preview, RED if self.face_problem else GREEN)

        screen = np.full((SCREEN_HEIGHT, SCREEN_WIDTH, 3), 40, dtype=np.uint8)
        top = SCREEN_HEIGHT - preview.shape[0] - 10
        left = (SCREEN_WIDTH - preview.shape[1]) // 2
        screen[top:top + preview.shape[0], left:left + preview.shape[1]] = preview
        return self._render_hebrew_text(REPOSITION_MESSAGE, (SCREEN_WIDTH // 2, 30), font_size=36,
                                        color=(255, 255, 255), background_image=screen, centered=True)

    def _draw_position_status(self, image, frame):
        """Small camera view and the position status, on screens that wait for the face to be in the box"""
        h, w = frame.shape[:2]
        thumbnail = cv2.resize(frame, (240, int(240 * h / w)))
        draw_face_box(thumbnail, RED if self.face_problem else GREEN)
        image[20:20 + thumbnail.shape[0], 20:20 + thumbnail.shape[1]] = thumbnail
        status = self.face_problem or "Position OK - press D to start"
        cv2.putText(image, status, (275, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.55, RED if self.face_problem else (0, 150, 0), 2)

    def _render_question(self, step, image):
        """Draw a step's text centred on the question screen"""
        # Draw semi-transparent overlay
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (SCREEN_WIDTH, SCREEN_HEIGHT), (240, 240, 240), -1)
        cv2.addWeighted(overlay, 0.9, image, 0.1, 0, image)

        # Draw Question. A screen with instructions puts its title higher, with the instructions below
        question_text = step['content']
        details = step.get('details')
        image = self._render_hebrew_text(
            question_text, 
            (SCREEN_WIDTH // 2, int(SCREEN_HEIGHT * (0.12 if details else 0.3))),
            font_size=40, 
            color=(0, 0, 0), 
            background_image=image,
            centered=True # Center the text
        )
        return draw_instructions(image, details) if details else image

    def _render_probe(self, step, image):
        """Render probe dialog"""
        image = self._render_question(step, image)

        # Draw Yes/No Buttons
        button_y = int(SCREEN_HEIGHT * 0.6)
        button_width = 150
        button_height = 60
        spacing = 100
        
        # Yes Button (Right side for Hebrew?) - Let's put Yes on Right, No on Left
        yes_x = SCREEN_WIDTH // 2 + spacing // 2
        no_x = SCREEN_WIDTH // 2 - button_width - spacing // 2
        
        # Check hover
        mx, my = self.mouse_x, self.mouse_y
        
        yes_hover = yes_x <= mx <= yes_x + button_width and button_y <= my <= button_y + button_height
        no_hover = no_x <= mx <= no_x + button_width and button_y <= my <= button_y + button_height
        
        # Draw Yes
        color = (100, 200, 100) if yes_hover else (150, 255, 150)
        cv2.rectangle(image, (yes_x, button_y), (yes_x + button_width, button_y + button_height), color, -1)
        cv2.rectangle(image, (yes_x, button_y), (yes_x + button_width, button_y + button_height), (0, 100, 0), 2)
        cv2.putText(image, "YES", (yes_x + 40, button_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 50, 0), 2)
        
        # Draw No
        color = (100, 100, 255) if no_hover else (150, 150, 255)
        cv2.rectangle(image, (no_x, button_y), (no_x + button_width, button_y + button_height), color, -1)
        cv2.rectangle(image, (no_x, button_y), (no_x + button_width, button_y + button_height), (0, 0, 100), 2)
        cv2.putText(image, "NO", (no_x + 50, button_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 50), 2)
        
        # Handle Click
        if self.mouse_clicked:
            if yes_hover:
                self.probe_answers[self.current_page] = "Yes"
                self.current_page += 1
                self.mouse_clicked = False
                print(f"Text {step['text']} probe answered: Yes")
            elif no_hover:
                self.probe_answers[self.current_page] = "No"
                self.current_page += 1
                self.mouse_clicked = False
                print(f"Text {step['text']} probe answered: No")
            else:
                self.mouse_clicked = False # Reset if clicked elsewhere
                
        return image

    def _render_hebrew_text(self, text, position, font_size = TEXT_FONT_SIZE, color = TEXT_COLOR, background_image = None, image_width =SCREEN_WIDTH, image_height = SCREEN_HEIGHT, centered=False):
        """Render Hebrew text using PIL and convert to OpenCV format"""
        try:
            # Detect if text contains Hebrew characters
            hebrew_pattern = re.compile(r'[\u0590-\u05FF]')
            contains_hebrew = bool(hebrew_pattern.search(text))

            #print(f"DEBUG: Rendering text: '{text[:50]}...' (Hebrew: {contains_hebrew})")

            # Create PIL image
            if background_image is not None:
                # Convert OpenCV(BGR) image to PIL(RGB)
                pil_image = Image.fromarray(cv2.cvtColor(background_image, cv2.COLOR_BGR2RGB))
            else:
                pil_image = Image.new("RGB", (image_width, image_height), (255, 255, 255))
            
            draw = ImageDraw.Draw(pil_image)

            hebrew_font = load_text_font(font_size)

            # Handle RTL text properly
            display_text = text
            if contains_hebrew:
                try:
                    # Try to use bidi library for proper RTL handling
                    from bidi.algorithm import get_display
                    import arabic_reshaper
                    
                    # Reshape and reorder Hebrew text for proper display
                    reshaped_text = arabic_reshaper.reshape(text)
                    display_text = get_display(reshaped_text)
                    #print(f"DEBUG: RTL processed successfully")
                    
                except ImportError:
                    # Fallback: Simple word reversal for Hebrew
                    words = text.split()
                    display_text = ' '.join(reversed(words))
                    #print(f"DEBUG: Using simple RTL fallback")
                except Exception as e:
                    #print(f"DEBUG: RTL processing error: {e}, using original text")
                    display_text = text

            # Adjust position for alignment if needed
            render_position = position
            
            if centered:
                # Calculate text dimensions
                bbox = draw.textbbox((0, 0), display_text, font=hebrew_font)
                text_width = bbox[2] - bbox[0]

                # Center text around position[0]
                text_start_x = position[0] - text_width // 2

                # Ensure proper left margin constraint
                final_x = max(text_start_x, 50)
                render_position = (final_x, position[1])

            # Render text
            #print(f"DEBUG: Rendering at position {render_position} with font size {font_size}")
            draw.text(render_position, display_text, font=hebrew_font, fill=color)
            
            # Convert back to OpenCV(BGR) format
            result_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            #print(f"DEBUG: Successfully rendered Hebrew text, image shape: {result_image.shape}")
            
            # Verify the image is valid
            if result_image is None or result_image.size == 0:
                print("ERROR: Rendered image is empty!")
                return None
                
            return result_image
            
        except Exception as e:
            print(f"ERROR in Hebrew rendering: {e}")
            # Final fallback - return original image or create blank
            if background_image is not None:
                return background_image
            else:
                return np.zeros((image_height, image_width, 3), dtype=np.uint8)

    def _draw_navigation_info(self, image):
        """Draw the page number on text pages, and the controls on every screen"""
        current_step = self.reading_sequence[self.current_page]

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2

        if current_step['type'] == 'page':
            page_info = f"Page {current_step['page_number']} of {self.total_pages}"
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
            self.face_problem = MESSAGES['face_not_detected']
            return None, None, None # gaze point, blink, pupil size

        face_landmarks = results.multi_face_landmarks[0]
        self.face_problem = face_position_problem(face_landmarks, frame.shape)
        
        # Extract features (simplified - adapt to your feature extraction logic)
        features = self._extract_features(face_landmarks, frame.shape)
        
        
        # Calculate pupil size and blink detection
        prediction = self._predict_gaze_point(features)
        pupil_size = self._calculate_pupil_size(face_landmarks, frame.shape)
        current_blink = self._detect_blink(face_landmarks)
        blink_frequency = self._update_blink_frequency(current_blink)

        return prediction, pupil_size, blink_frequency

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
            return {'pitch': 0.0, 'yaw': 0.0, 'tvect_x': 0.0, 'tvect_y': 0.0, 'tvect_z': 0.0}
        
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
                return {'pitch': 0.0, 'yaw': 0.0, 'tvect_x': 0.0, 'tvect_y': 0.0, 'tvect_z': 0.0}
            
            # Convert to Euler angles
            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
            sy = np.sqrt(rotation_matrix[0,0]**2 + rotation_matrix[1,0]**2)
            
            singular = sy < 1e-6
            if not singular:
                yaw = np.arctan2(rotation_matrix[1,0], rotation_matrix[0,0])
                pitch = np.arctan2(-rotation_matrix[2,0], sy)
                #roll = np.arctan2(rotation_matrix[2,1], rotation_matrix[2,2])
            else:
                yaw = np.arctan2(-rotation_matrix[1,2], rotation_matrix[1,1])
                pitch = np.arctan2(-rotation_matrix[2,0], sy)
                #roll = 0
            
            return {
                'pitch': np.degrees(pitch),
                'yaw': np.degrees(yaw),
                'tvect_x': translation_vector[0][0],
                'tvect_y': translation_vector[1][0],
                'tvect_z': translation_vector[2][0]
            }
            
        except Exception as e:
            return {'pitch': 0.0, 'yaw': 0.0, 'tvect_x': 0.0, 'tvect_y': 0.0, 'tvect_z': 0.0}
    
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
                pose['pitch'], pose['yaw'],
                pose['tvect_x'], pose['tvect_y'], pose['tvect_z']]
    
    def _calculate_pupil_size(self, landmarks, frame_shape):
        """Calculate relative pupil size and print values"""
        try:
            h, w = frame_shape[:2]
            
            # Get left and right iris landmarks
            left_iris_points = np.array([[landmarks.landmark[i].x * w, landmarks.landmark[i].y * h] 
                                    for i in LEFT_IRIS_LANDMARKS])
            right_iris_points = np.array([[landmarks.landmark[i].x * w, landmarks.landmark[i].y * h] 
                                        for i in RIGHT_IRIS_LANDMARKS])
            
            # Calculate iris diameters
            left_iris_width = np.max(left_iris_points[:, 0]) - np.min(left_iris_points[:, 0])
            left_iris_height = np.max(left_iris_points[:, 1]) - np.min(left_iris_points[:, 1])
            right_iris_width = np.max(right_iris_points[:, 0]) - np.min(right_iris_points[:, 0])
            right_iris_height = np.max(right_iris_points[:, 1]) - np.min(right_iris_points[:, 1])
            
            # Average iris size
            left_iris_size = (left_iris_width + left_iris_height) / 2
            right_iris_size = (right_iris_width + right_iris_height) / 2
            avg_pupil_size = (left_iris_size + right_iris_size) / 2
            
            # DEBUG: Print values
            #print(f"DEBUG - Left iris: {left_iris_width:.2f}x{left_iris_height:.2f}")
            #print(f"DEBUG - Right iris: {right_iris_width:.2f}x{right_iris_height:.2f}")
            #print(f"DEBUG - Average raw size: {avg_pupil_size:.2f}")
            
            # FIXED normalization
            normalized_pupil_size = 2.0 + (avg_pupil_size / 50.0) * 3.0  # Better scaling
            normalized_pupil_size = max(2.0, min(6.0, normalized_pupil_size))

            #print(f"DEBUG - Normalized pupil size: {normalized_pupil_size:.2f}")

            return normalized_pupil_size
            
        except Exception as e:
            print(f"Pupil size error: {e}")
            return 0.0  # Default size

    def _detect_blink(self, landmarks):
        """Detect blink based on eye aspect ratio"""
        try:
            # MediaPipe face mesh eye landmarks for EAR calculation
            # Left eye: Use specific landmarks for accurate EAR
            left_eye_horizontal = [33, 133]  # Left and right corners
            left_eye_vertical_1 = [160, 144]  # Top and bottom vertical pair 1
            left_eye_vertical_2 = [159, 145]  # Top and bottom vertical pair 2
            
            # Right eye: Use specific landmarks for accurate EAR  
            right_eye_horizontal = [362, 263]  # Left and right corners
            right_eye_vertical_1 = [387, 373]  # Top and bottom vertical pair 1
            right_eye_vertical_2 = [386, 374]  # Top and bottom vertical pair 2
            
            def calculate_distance(p1_idx, p2_idx):
                """Calculate Euclidean distance between two landmark points"""
                p1 = landmarks.landmark[p1_idx]
                p2 = landmarks.landmark[p2_idx]
                return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
            
            def eye_aspect_ratio(horizontal, vertical_1, vertical_2):
                """Calculate EAR using correct MediaPipe landmarks"""
                # Two vertical distances
                A = calculate_distance(vertical_1[0], vertical_1[1])
                B = calculate_distance(vertical_2[0], vertical_2[1])
                # One horizontal distance
                C = calculate_distance(horizontal[0], horizontal[1])
                
                # EAR formula: (A + B) / (2.0 * C)
                return (A + B) / (2.0 * C)
            
            # Calculate EAR for both eyes
            left_ear = eye_aspect_ratio(left_eye_horizontal, left_eye_vertical_1, left_eye_vertical_2)
            right_ear = eye_aspect_ratio(right_eye_horizontal, right_eye_vertical_1, right_eye_vertical_2)
            avg_ear = (left_ear + right_ear) / 2.0
            
            # Blink threshold - typical EAR values: open eyes ~0.3, closed eyes ~0.1
            blink_threshold = 0.25  # Standard threshold for blink detection
            return 1.0 if avg_ear < blink_threshold else 0.0
            
        except Exception as e:
            return 0.0  # Default no blink

    def _update_blink_frequency(self, current_blink):
        """Track blink frequency over time"""
        if not hasattr(self, 'blink_history'):
            self.blink_history = []
            self.last_blink_state = 0.0
        
        # Add to history with timestamp
        current_time = time.time()
        self.blink_history.append((current_time, current_blink))
        
        # Remove old entries (keep last 60 seconds)
        self.blink_history = [(t, b) for t, b in self.blink_history if current_time - t <= 60.0]
        
        # Count blink events (transitions from 0 to 1)
        blink_count = 0
        for i in range(1, len(self.blink_history)):
            if self.blink_history[i][1] > 0.5 and self.blink_history[i-1][1] <= 0.5:
                blink_count += 1
        
        # Calculate frequency (blinks per minute)
        time_window = 60.0 if len(self.blink_history) > 1 else 1.0
        if len(self.blink_history) > 1:
            actual_time_window = min(60.0, current_time - self.blink_history[0][0])
            blink_frequency = (blink_count / actual_time_window) * 60.0  # Convert to per minute
        else:
            blink_frequency = 0.0
        
        self.last_blink_state = current_blink
        return blink_frequency

    def _predict_gaze_point(self, features):
        """Predict gaze point from features using trained model"""
        try:
            # Convert to pandas DataFrame for feature engineering (same as training)
            feature_names = FEATURE_COLUMNS
            df_features = pd.DataFrame([features], columns=feature_names)
            
            # Apply SAME feature engineering as training
            df_features['avg_iris_x'] = (df_features['norm_L_x'] + df_features['norm_R_x']) / 2
            df_features['avg_iris_y'] = (df_features['norm_L_y'] + df_features['norm_R_y']) / 2
            df_features['tvect_avg_x_inter'] = df_features['tvect_x'] * df_features['avg_iris_x']
            df_features['tvect_avg_y_inter'] = df_features['tvect_y'] * df_features['avg_iris_y']
            
            # Use same feature order as training
            engineered_features = FEATURE_COLUMNS + ['tvect_avg_x_inter', 'tvect_avg_y_inter']
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
    
    def _add_gaze_point_to_analysis(self, x, y, pupil_size, blink_frequency):
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
                current_velocity = metrics.get('current_velocity', 0)
            except:
                current_velocity = 0
            
            # Store fixation data
            if current_velocity < SACCADE_VELOCITY_THRESHOLD:  
                # print(f"DEBUG: Fixation detected! Word: {nearest_word}, Velocity: {current_velocity}")
                self.fixation_data.append({
                    'timestamp': current_time,
                    'x': x,
                    'y': y,
                    'word': nearest_word['word'] if nearest_word else "Unknown",
                    'line': nearest_word['line'] if nearest_word else None,
                    'word_index': nearest_word['word_index'] if nearest_word else None,
                    'time_from_onset': time_from_onset,
                    'page': self.current_page,
                    'pupil_size': pupil_size,
                    'blink_frequency': blink_frequency
                })
            # else:
            #    print(f"DEBUG: Saccade detected (Velocity: {current_velocity})")
    
    def _find_nearest_word(self, gaze_x, gaze_y):
        """Find the word nearest to the gaze, returning its position record or None if none is close enough"""
        min_distance = float('inf')
        nearest_word = None
        
        for word_info in self.word_positions:
            word_x = word_info['x']
            word_y = word_info['y']
            
            distance = np.sqrt((gaze_x - word_x)**2 + (gaze_y - word_y)**2)
            
            if distance < min_distance and distance < WORD_PROXIMITY_THRESHOLD:
                min_distance = distance
                nearest_word = word_info
        
        return nearest_word
    
    def _is_hebrew_word(self, word: str) -> bool:
        """Check if a word contains Hebrew characters"""
        if not word or word == "Unknown":
            return False
        # Check if any character in the word is Hebrew
        hebrew_chars = any('\u0590' <= char <= '\u05FF' for char in word)
        return hebrew_chars
    
    def _clean_hebrew_word(self, word: str) -> str:
        """Extract only Hebrew letters from a word, removing punctuation and other marks"""
        if not word or word == "Unknown":
            return word
        
        # Keep only Hebrew characters (Unicode range U+0590 to U+05FF)
        cleaned_word = ''.join(char for char in word if '\u0590' <= char <= '\u05FF')
        
        # Return cleaned word if it has Hebrew characters, otherwise return "Unknown"
        return cleaned_word if cleaned_word else "Unknown"
    
    def _answer_after_step(self, step_index):
        """Answer to the first probe after a step, the one that closes that part of the text"""
        for index in range(step_index + 1, len(self.reading_sequence)):
            if self.reading_sequence[index]['type'] == 'probe':
                return self.probe_answers.get(index, "N/A")
        return "N/A"

    def _export_probe_answers(self) -> str:
        """Save every question with its answer. Ratings stay blank for the tester to fill in"""
        rows = []
        for index, step in enumerate(self.reading_sequence):
            if step['type'] not in ('probe', 'rating'):
                continue

            answer = self.probe_answers.get(index, "N/A" if step['type'] == 'probe' else "")
            correct_answer = step['correct_answer']
            rows.append({
                'Question_Order': len(rows) + 1,
                'Text_Number': step['text'],
                'Text_Type': step['text_type'],
                'Part_Number': step['part'],
                'Question_Type': step['type'],
                'Question': step['content'],
                'Answer': answer,
                'Correct_Answer': correct_answer,
                'Is_Correct': answer == correct_answer if correct_answer and answer != "N/A" else None
            })

        os.makedirs(self.output_dir, exist_ok=True)
        filepath = os.path.join(self.output_dir, f"probe_answers_{self.session_timestamp}.csv")
        pd.DataFrame(rows).to_csv(filepath, index=False, encoding='utf-8-sig')
        print(f"Probe answers saved to: {filepath}")
        return filepath

    def _export_position_events(self) -> str:
        """Save every stretch the face spent out of the box, so drifts and pauses can be reviewed"""
        columns = ['Event_Order', 'Start_ms', 'Duration_ms', 'Screen', 'Text_Number', 'Part_Number', 'Page_Number', 'Reason', 'Paused']
        rows = [{'Event_Order': i + 1, **event} for i, event in enumerate(self.position_events)]
        df = pd.DataFrame(rows, columns=columns).astype({'Text_Number': 'Int64', 'Part_Number': 'Int64', 'Page_Number': 'Int64'})

        os.makedirs(self.output_dir, exist_ok=True)
        filepath = os.path.join(self.output_dir, f"session_events_{self.session_timestamp}.csv")
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        print(f"Position events saved to: {filepath} ({len(rows)} out of position)")
        return filepath

    def _export_fixation_data(self) -> str:
        """Export fixation data in the same format as your CSV example"""
        # Answers go first and don't depend on fixations, so a page with no gaze can't lose one
        self._export_probe_answers()
        self._export_position_events()

        # print(f"DEBUG: Exporting fixation data. Total raw fixations: {len(self.fixation_data)}")
        if not self.fixation_data:
            print("No fixation data to export")
            return None
        
        # Process fixation data to match your CSV format with Hebrew filtering
        processed_fixations = []
        fixation_order = 1
        
        # Group consecutive gaze points into fixations and filter for Hebrew words only
        current_fixation = None
        
        for data in self.fixation_data:
            # Clean the word and check if it's Hebrew
            cleaned_word = self._clean_hebrew_word(data['word'])
            
            # print(f"DEBUG: Processing word: '{data['word']}' -> Cleaned: '{cleaned_word}'")

            # Skip non-Hebrew words only if we want strict filtering, but for debugging let's keep everything
            # or at least log it. For now, let's relax it to allow "Unknown" or non-Hebrew if needed.
            # actually, let's just use the cleaned word if it exists, otherwise original
            
            final_word = cleaned_word if cleaned_word != "Unknown" else data['word']
            
            # if not self._is_hebrew_word(cleaned_word) or cleaned_word == "Unknown":
            #    # print(f"DEBUG: Skipping non-Hebrew word: {cleaned_word}")
            #    continue
            
            # Get probe answer for this page
            step = self.reading_sequence[data['page']]
            probe_ans = self._answer_after_step(data['page'])
            
            if current_fixation is None:
                current_fixation = {
                    'start_time': data['timestamp'],
                    'end_time': data['timestamp'],
                    'x_positions': [data['x']],
                    'y_positions': [data['y']],
                    'word': final_word,  # Use final_word (relaxed filtering)
                    'step': step,
                    'line': data['line'],
                    'word_index': data['word_index'],
                    'time_from_onset': data['time_from_onset'],
                    'pupil_sizes': [data.get('pupil_size', 0)],
                    'blink_frequencies': [data.get('blink_frequency', 0)],
                    'probe_answer': probe_ans
                }
            else:
                # Check if this continues the current fixation (same word position, not just the same spelling)
                time_gap = data['timestamp'] - current_fixation['end_time']
                
                # Increased threshold to 0.5s to handle slower frame rates/processing
                same_word = step is current_fixation['step'] and data['word_index'] == current_fixation['word_index']
                if time_gap < 0.5 and same_word:  # Same fixation
                    # print(f"DEBUG: Extending fixation for '{final_word}'")
                    current_fixation['end_time'] = data['timestamp']
                    current_fixation['x_positions'].append(data['x'])
                    current_fixation['y_positions'].append(data['y'])
                    current_fixation['pupil_sizes'].append(data.get('pupil_size', 0))
                    current_fixation['blink_frequencies'].append(data.get('blink_frequency', 0))
                    # Probe answer should be same for same page
                else:  # New fixation
                    # print(f"DEBUG: Ending fixation for '{current_fixation['word']}' (New word: '{final_word}', Time gap: {time_gap:.3f}s)")
                    # Process completed fixation
                    duration = (current_fixation['end_time'] - current_fixation['start_time']) * 1000
                    
                    # If duration is 0 (single point or fast samples), estimate based on sample count
                    if duration == 0:
                        duration = len(current_fixation['x_positions']) * 33.0 # Assume ~30fps
                    
                    # print(f"DEBUG: Fixation candidate duration: {duration:.2f}ms (Threshold: {FIXATION_THRESHOLD}ms)")
                    # print(f"DEBUG: Start: {current_fixation['start_time']}, End: {current_fixation['end_time']}, Diff: {current_fixation['end_time'] - current_fixation['start_time']}")

                    if duration >= FIXATION_THRESHOLD:  # Only include significant fixations
                        # print(f"DEBUG: ACCEPTED fixation for '{current_fixation['word']}'")
                        avg_x = np.mean(current_fixation['x_positions'])
                        avg_y = np.mean(current_fixation['y_positions'])
                        avg_pupil_size = np.mean(current_fixation['pupil_sizes'])
                        avg_blink_frequency = np.mean(current_fixation['blink_frequencies'])

                        processed_fixations.append({
                            'Fixation_Order': fixation_order,
                            'Text_Number': current_fixation['step']['text'],
                            'Text_Type': current_fixation['step']['text_type'],
                            'Part_Number': current_fixation['step']['part'],
                            'Page_Number': current_fixation['step']['page_number'],
                            'Line': current_fixation['line'],
                            'Word_Index': current_fixation['word_index'],
                            'Fixated_Word': current_fixation['word'],
                            'Fixation_X_Screen': round(avg_x, 2),
                            'Fixation_Y_Screen': round(avg_y, 2),
                            'Fixation_Duration': round(duration, 2),
                            'Time_from_Stimulus_Onset': round(current_fixation['time_from_onset'], 2),
                            'Pupil_Size': round(avg_pupil_size, 2), 
                            'Blink_Frequency': round(avg_blink_frequency, 2),
                            'Probe_Answer': current_fixation['probe_answer']
                        })
                    fixation_order += 1
                
                    # Start new fixation
                    current_fixation = {
                        'start_time': data['timestamp'],
                        'end_time': data['timestamp'],
                        'x_positions': [data['x']],
                        'y_positions': [data['y']],
                        'word': final_word,
                        'step': step,
                        'line': data['line'],
                        'word_index': data['word_index'],
                        'time_from_onset': data['time_from_onset'],
                        'pupil_sizes': [data.get('pupil_size', 0)],
                        'blink_frequencies': [data.get('blink_frequency', 0)],
                        'probe_answer': probe_ans
                    }
    
        # Process final fixation
        if current_fixation:
            duration = (current_fixation['end_time'] - current_fixation['start_time']) * 1000
            
            # If duration is 0 (single point or fast samples), estimate based on sample count
            if duration == 0:
                duration = len(current_fixation['x_positions']) * 33.0 # Assume ~30fps
            
            if duration >= FIXATION_THRESHOLD:
                avg_x = np.mean(current_fixation['x_positions'])
                avg_y = np.mean(current_fixation['y_positions'])
                avg_pupil_size = np.mean(current_fixation['pupil_sizes'])
                avg_blink_frequency = np.mean(current_fixation['blink_frequencies'])
                
                processed_fixations.append({
                    'Fixation_Order': fixation_order,
                    'Text_Number': current_fixation['step']['text'],
                    'Text_Type': current_fixation['step']['text_type'],
                    'Part_Number': current_fixation['step']['part'],
                    'Page_Number': current_fixation['step']['page_number'],
                    'Line': current_fixation['line'],
                    'Word_Index': current_fixation['word_index'],
                    'Fixated_Word': current_fixation['word'],
                    'Fixation_X_Screen': round(avg_x, 2),
                    'Fixation_Y_Screen': round(avg_y, 2),
                    'Fixation_Duration': round(duration, 2),
                    'Time_from_Stimulus_Onset': round(current_fixation['time_from_onset'], 2),
                    'Pupil_Size': round(avg_pupil_size,2),
                    'Blink_Frequency': round(avg_blink_frequency,2),
                    'Probe_Answer': current_fixation['probe_answer']
                })
        
        # Create DataFrame and export with consecutive word grouping
        if processed_fixations:
            # Apply consecutive word grouping to avoid duplicates
            grouped_fixations = self._group_consecutive_words(processed_fixations)
            
            # Line and Word_Index stay whole numbers, and are blank when the gaze wasn't on a word
            df = pd.DataFrame(grouped_fixations).astype({'Line': 'Int64', 'Word_Index': 'Int64'})
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"eye_movement_data_{timestamp}.csv"
            
            # Ensure output directory exists
            os.makedirs(self.output_dir, exist_ok=True)
            filepath = os.path.join(self.output_dir, filename)
            
            df.to_csv(filepath, index=False, encoding='utf-8-sig')  # The BOM lets Excel read the Hebrew
            print(f"Exported {len(grouped_fixations)} Hebrew word fixations to: {filepath}")
            print(f"Hebrew words recognized: {len(set(fix['Fixated_Word'] for fix in grouped_fixations))}")
            print(f"Grouped from {len(processed_fixations)} individual fixations")
            return filepath
        else:
            # print("DEBUG: No processed fixations found after filtering (or input data was empty).")
            return None
    
    def _group_consecutive_words(self, fixations):
        """Group consecutive fixations on the same word to avoid duplicates"""
        if not fixations:
            return []
        
        grouped_fixations = []
        current_word = None
        group_start = None
        current_line = None
        current_index = None
        accumulated_duration = 0
        accumulated_x = []
        accumulated_y = []
        word_start_time = None
        pupil_sizes = []
        blink_frequencies = []
        probe_answers = []
        current_page = None
        
        for fixation in fixations:
            word = fixation['Fixated_Word']
            word_index = fixation['Word_Index']
            page = fixation.get('Page_Number', 0)
            
            if word_index == current_index and page == current_page and current_word is not None:
                # Same word position AND same page - accumulate data
                accumulated_duration += fixation['Fixation_Duration']
                accumulated_x.append(fixation['Fixation_X_Screen'])
                accumulated_y.append(fixation['Fixation_Y_Screen'])
                pupil_sizes.append(fixation['Pupil_Size'])
                blink_frequencies.append(fixation['Blink_Frequency'])
                probe_answers.append(fixation['Probe_Answer'])
            else:
                # Different word - save previous group if exists
                if current_word is not None:
                    grouped_fixations.append({
                        'Fixation_Order': len(grouped_fixations) + 1,
                        'Text_Number': group_start['Text_Number'],
                        'Text_Type': group_start['Text_Type'],
                        'Part_Number': group_start['Part_Number'],
                        'Page_Number': current_page,
                        'Line': current_line,
                        'Word_Index': current_index,
                        'Fixated_Word': current_word,
                        'Fixation_X_Screen': round(np.mean(accumulated_x), 2),
                        'Fixation_Y_Screen': round(np.mean(accumulated_y), 2),
                        'Fixation_Duration': round(accumulated_duration, 2),
                        'Time_from_Stimulus_Onset': word_start_time,
                        'Pupil_Size': round(np.mean(pupil_sizes), 2),
                        'Blink_Frequency': round(np.mean(blink_frequencies), 2),
                        'Probe_Answer': probe_answers[0] # Should be same for group
                    })
                
                # Start new word group
                current_word = word
                group_start = fixation
                current_line = fixation['Line']
                current_index = word_index
                current_page = page
                word_start_time = fixation['Time_from_Stimulus_Onset']
                accumulated_duration = fixation['Fixation_Duration']
                accumulated_x = [fixation['Fixation_X_Screen']]
                accumulated_y = [fixation['Fixation_Y_Screen']]
                pupil_sizes = [fixation['Pupil_Size']]
                blink_frequencies = [fixation['Blink_Frequency']]
                probe_answers = [fixation['Probe_Answer']]
        
        # Don't forget the last word group
        if current_word is not None:
            grouped_fixations.append({
                'Fixation_Order': len(grouped_fixations) + 1,
                'Text_Number': group_start['Text_Number'],
                'Text_Type': group_start['Text_Type'],
                'Part_Number': group_start['Part_Number'],
                'Page_Number': current_page,
                'Line': current_line,
                'Word_Index': current_index,
                'Fixated_Word': current_word,
                'Fixation_X_Screen': round(np.mean(accumulated_x), 2),
                'Fixation_Y_Screen': round(np.mean(accumulated_y), 2),
                'Fixation_Duration': round(accumulated_duration, 2),
                'Time_from_Stimulus_Onset': word_start_time,
                'Pupil_Size': round(np.mean(pupil_sizes), 2),
                'Blink_Frequency': round(np.mean(blink_frequencies), 2),
                'Probe_Answer': probe_answers[0]
            })
        
        return grouped_fixations
    
    def run_text_reading_analysis(self):
        """Run the complete text reading analysis system"""
        if self.model is None:
            print("No model loaded. Please load a model first.")
            return
        
        self.movement_analyzer.start_reading_session()  # Reset analyzer
        
        print("Starting Text Reading Analysis System...")
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
        cv2.setMouseCallback('Text Reading Analysis', self._mouse_callback)
        
        # Start video recording
        if self.video_recorder and VIDEO_RECORDING_ENABLED:
            self.recording_paths = self.video_recorder.start_recording()
            print("=" * 60)
        
        # The reading clock starts when the first text page appears, not on the opening screen
        self.session_start_time = None

        try:
            while True:
                ret, frame = self.camera.read()
                if not ret:
                    break
                
                frame = cv2.flip(frame, 1)  # Mirror image
                
                # Record camera frame
                if self.video_recorder and RECORD_CAMERA_FEED:
                    self.video_recorder.record_frame(camera_frame=frame)
                
                # Predict gaze point
                prediction, pupil_size, blink_frequency = self._detect_face_and_predict(frame)
                
                # Create text display. A click on a probe moves on while it draws, so note the step on screen first
                shown_step = self.reading_sequence[self.current_page]
                if self.session_start_time is None and shown_step['type'] == 'page':
                    self.session_start_time = time.time()
                if OOB_CHECK_ENABLED:
                    self._update_position(shown_step)

                if self.paused:
                    text_display = self._render_reposition_screen(frame)
                else:
                    text_display = self._render_text_page()
                    if OOB_CHECK_ENABLED and shown_step.get('check_position'):
                        self._draw_position_status(text_display, frame)

                # The overlay recording always carries the gaze markers, whatever the participant sees
                recorded_display = text_display.copy()

                if prediction and not self.paused:
                    pred_x, pred_y = prediction
                    
                    # Apply smoothing
                    if self.smoothed_x is None:
                        self.smoothed_x, self.smoothed_y = pred_x, pred_y
                    else:
                        self.smoothed_x = self.smoothed_x * (1 - self.smoothing_factor) + pred_x * self.smoothing_factor
                        self.smoothed_y = self.smoothed_y * (1 - self.smoothing_factor) + pred_y * self.smoothing_factor
                    
                    smoothed_x = int(self.smoothed_x)
                    smoothed_y = int(self.smoothed_y)
                    
                    # Add to analysis only on a text page, and only with the face in the box when the check is on
                    in_position = not (OOB_CHECK_ENABLED and self.face_problem)
                    if shown_step['type'] == 'page' and in_position:
                        # print(f"DEBUG: Adding gaze point ({smoothed_x}, {smoothed_y})")
                        self._add_gaze_point_to_analysis(smoothed_x, smoothed_y, pupil_size, blink_frequency)
                    else:
                        # Optional: Add marker for probe
                        cv2.circle(recorded_display, (SCREEN_WIDTH-30, 30), 10, (0, 255, 255), -1)

                    # Draw gaze point (semi-transparent)
                    overlay = recorded_display.copy()
                    cv2.circle(overlay, (smoothed_x, smoothed_y), 12, (255, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.6, recorded_display, 0.4, 0, recorded_display)
                else:
                    # print("DEBUG: No prediction")
                    overlay = text_display.copy()
                
                # Record overlay frame (after all processing)
                if self.video_recorder and RECORD_OVERLAY_DISPLAY:
                    # print("DEBUG: Recording overlay frame")
                    self.video_recorder.record_frame(overlay_frame=recorded_display)

                if SHOW_GAZE_POINT:
                    text_display = recorded_display

                cv2.imshow('Text Reading Analysis', text_display)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    break
                elif key == ord('a') or key == ord('A'):  # Previous page
                    if self.current_page > 0 and not self.paused:
                        self.current_page -= 1
                        print(f"Moved to page {self.current_page + 1}")
                elif key == ord('d') or key == ord('D'):  # Next page
                    if self.current_page < len(self.reading_sequence) - 1 and not self.paused:
                        # Probes require a click, every other step moves on with this key.
                        # A screen that checks the position, like the opening one, waits for the face to be in the box
                        step = self.reading_sequence[self.current_page]
                        waiting = OOB_CHECK_ENABLED and step.get('check_position') and self.face_problem
                        if step['type'] != 'probe' and not waiting:
                            self.current_page += 1
                            print(f"Moved to step {self.current_page + 1}")
                elif key == ord('e') or key == ord('E'):  # Export data
                    self._export_fixation_data()
        
        finally:
            cv2.destroyAllWindows()
            self._close_position_event()

            # Stop video recording
            if self.video_recorder and VIDEO_RECORDING_ENABLED:
                self.video_metadata = self.video_recorder.stop_recording()
            
            # Final data export
            print("\n" + "=" * 60)
            print("SESSION COMPLETE - EXPORTING DATA")
            print("=" * 60)
            
            csv_file = self._export_fixation_data()
            if csv_file:
                print(f"Eye movement data exported to: {csv_file}")
                print("Data format matches your example CSV structure")
            
            # Display video recording information
            if self.video_metadata:
                print("\nVideo recordings saved:")
                if self.video_metadata.get('camera_feed_path'):
                    print(f"  Camera feed: {os.path.basename(self.video_metadata['camera_feed_path'])}")
                if self.video_metadata.get('overlay_display_path'):
                    print(f"  Overlay display: {os.path.basename(self.video_metadata['overlay_display_path'])}")
                print(f"  Resolution: {self.video_metadata.get('camera_resolution', 'N/A')} (camera), {self.video_metadata.get('overlay_resolution', 'N/A')} (overlay)")
                print(f"  FPS: {self.video_metadata.get('fps', 'N/A')}")
            
            print("Text Reading Analysis Complete!")

def main():
    """Main function for testing"""
    print("Text Reading Gaze Analysis System")
    print("=" * 50)
    
    # Check for trained model
    model_path = os.path.join(MODELS_DIR, "gaze_model.pkl")
    
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        print("Please train a model first using the main system")
        return
    
    # Create predictor and run analysis
    predictor = TextReadingGazePredictor(model_path)
    predictor.run_text_reading_analysis()

if __name__ == "__main__":
    main()
