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
from PIL import ImageFont, ImageDraw, Image
import os
import re
import threading
from queue import Queue
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
        
        # Threading for async writing (optional performance enhancement)
        self.camera_queue = Queue(maxsize=VIDEO_BUFFER_SIZE) if VIDEO_RECORDING_ENABLED else None
        self.overlay_queue = Queue(maxsize=VIDEO_BUFFER_SIZE) if VIDEO_RECORDING_ENABLED else None
        self.write_thread = None
        self.stop_event = threading.Event()
        
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
            
            if RECORD_CAMERA_FEED and self.camera_path:
                self.camera_writer = cv2.VideoWriter(
                    self.camera_path, fourcc, fps, (CAMERA_WIDTH, CAMERA_HEIGHT)
                )
                
            if RECORD_OVERLAY_DISPLAY and self.overlay_path:
                self.overlay_writer = cv2.VideoWriter(
                    self.overlay_path, fourcc, fps, (SCREEN_WIDTH, SCREEN_HEIGHT)
                )
            
            self.is_recording = True
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
        """Record frames to video files"""
        if not self.is_recording:
            return
            
        try:
            if camera_frame is not None and self.camera_writer and RECORD_CAMERA_FEED:
                self.camera_writer.write(camera_frame)
                
            if overlay_frame is not None and self.overlay_writer and RECORD_OVERLAY_DISPLAY:
                self.overlay_writer.write(overlay_frame)
                
        except Exception as e:
            print(f"Error recording frame: {e}")
            # Continue without recording rather than crash
            
    def stop_recording(self) -> dict:
        """Stop recording and return metadata"""
        if not self.is_recording:
            return {}
            
        try:
            # Stop recording flag
            self.is_recording = False
            
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
                'camera_resolution': f"{CAMERA_WIDTH}x{CAMERA_HEIGHT}",
                'overlay_resolution': f"{SCREEN_WIDTH}x{SCREEN_HEIGHT}"
            }
            
            # Save metadata file
            if self.camera_path or self.overlay_path:
                metadata_path = os.path.join(self.output_dir, f"video_metadata_{self.session_id}.json")
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                print(f"Video metadata saved: {os.path.basename(metadata_path)}")
            
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
        self.output_dir = output_dir if output_dir is not None else OUTPUT_DIR
        
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
        self.page_metadata = [] # Store page_id and question
        self.word_positions = []
        
        # Probe system
        self.state = "READING" # READING or PROBE
        self.probe_answer = None
        self.yes_button_rect = None
        self.no_button_rect = None
        
        # Placeholder for potential missing attributes reported by user
        self.gaze_prediction = None
        self.prediction = None
        self.gaze_point = None
        
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
            session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.video_recorder = VideoRecorder(self.output_dir, session_timestamp)
            self.recording_paths = None
            self.video_metadata = None
        else:
            self.video_recorder = None
        
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
        """Create adaptive text pages with proper word wrapping from READING_PAGES"""
        print("Creating adaptive text pages...")
        
        self.text_pages = []
        self.page_metadata = []
        
        # Calculate approximate character width for better estimation
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = TEXT_FONT_SIZE / 32.0  # Match the rendering scale
        thickness = 4 if TEXT_FONT_BOLD else 2
        
        # Test character to estimate average character width
        char_width = cv2.getTextSize("A", font, font_scale, thickness)[0][0]
        margin_x = int(SCREEN_WIDTH * TEXT_MARGIN_X)
        available_width = SCREEN_WIDTH - 2 * margin_x
        
        # Calculate words per row
        if TEXT_FORCE_WORDS_PER_ROW is not None:
            conservative_words_per_row = TEXT_FORCE_WORDS_PER_ROW
            print(f"Using FORCED {conservative_words_per_row} words per row")
        else:
            estimated_chars_per_row = available_width // (char_width)
            avg_word_length = 4
            conservative_words_per_row = max(6, min(TEXT_WORDS_PER_ROW, estimated_chars_per_row + 2 // avg_word_length))
            print(f"Using {conservative_words_per_row} words per row")
        
        words_per_page = TEXT_ROWS_PER_PAGE * conservative_words_per_row
        
        # Check for Hebrew in the first page to set global RTL mode
        if READING_PAGES:
            first_text = READING_PAGES[0]['text']
            words = first_text.strip().split()
            
            # Improved detection using Unicode range instead of keywords
            hebrew_pattern = re.compile(r'[\u0590-\u05FF]')
            hebrew_count = sum(1 for word in words[:20] if hebrew_pattern.search(word))
            
            self.is_rtl_text = hebrew_count >= RTL_DETECTION_THRESHOLD if RTL_AUTO_DETECT else False
            
            if self.is_rtl_text:
                print(f"Hebrew text detected - configuring RTL mode")
                self.movement_analyzer = EyeMovementAnalyzer(
                    window_width=SCREEN_WIDTH,
                    window_height=SCREEN_HEIGHT,
                    text_reading_mode=True,
                    reading_direction='rtl'
                )
            else:
                print("LTR text detected - using standard configuration")

        # Process each page from config
        for page_config in READING_PAGES:
            text = page_config['text']
            words = text.strip().split()
            
            current_page_words = []
            
            # Split this text block into visual pages
            block_pages = []
            
            for i, word in enumerate(words):
                current_page_words.append(word)
                
                if len(current_page_words) >= words_per_page or i == len(words) - 1:
                    # Create rows for this page
                    rows = []
                    current_row = []
                    
                    for w in current_page_words:
                        current_row.append(w)
                        if len(current_row) >= conservative_words_per_row:
                            rows.append(current_row)
                            current_row = []
                    
                    if current_row:
                        rows.append(current_row)
                    
                    block_pages.append(rows)
                    current_page_words = []
            
            # Add these visual pages to the main list
            for i, rows in enumerate(block_pages):
                self.text_pages.append(rows)
                
                is_last_of_block = (i == len(block_pages) - 1)
                metadata = {
                    'page_id': page_config['page_id'],
                    'sub_page': i + 1,
                    'question': page_config['question'] if is_last_of_block else None
                }
                self.page_metadata.append(metadata)
        
        self.total_pages = len(self.text_pages)
        print(f"Created {self.total_pages} pages from {len(READING_PAGES)} content blocks")
    
    def _render_text_page(self) -> np.ndarray:
        """Render current text page with precise word positioning and width checking"""
        # Create background
        background = np.full((SCREEN_HEIGHT, SCREEN_WIDTH, 3), TEXT_BACKGROUND, dtype=np.uint8)
        
        # Check if we have Hebrew text to determine layout direction
        is_hebrew_page = self.is_rtl_text
        
        # Calculate text area with RTL consideration
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
            
            # Render each row with width checking and RTL layout
            for row_idx, row_words in enumerate(page_rows):
                if row_idx >= TEXT_ROWS_PER_PAGE:
                    break
                
                # Calculate row position
                y = margin_y + row_idx * TEXT_LINE_SPACING + int(TEXT_FONT_SIZE)
                
                # Render row with width constraints and RTL support
                self._render_row_with_width_check(background, row_words, margin_x, y, font, font_scale, thickness, row_idx, text_width, is_hebrew_page)
        
        # Add navigation info
        self._draw_navigation_info(background)
        
        # Update analyzer with new word positions
        if self.movement_analyzer:
             self.movement_analyzer.set_actual_word_positions(self.word_positions)
        
        return background

    def _render_probe_window(self, frame, question):
        """Render a modal probe window with the question and Yes/No buttons"""
        # Resize frame to screen resolution if needed (e.g. if using lower res camera)
        if frame.shape[1] != SCREEN_WIDTH or frame.shape[0] != SCREEN_HEIGHT:
            frame = cv2.resize(frame, (SCREEN_WIDTH, SCREEN_HEIGHT))

        # Create overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (SCREEN_WIDTH, SCREEN_HEIGHT), (0, 0, 0), -1)
        cv2.addWeighted(overlay, PROBE_OVERLAY_ALPHA, frame, 1 - PROBE_OVERLAY_ALPHA, 0, frame)
        
        # Draw dialog box
        dialog_w, dialog_h = int(SCREEN_WIDTH * 0.6), int(SCREEN_HEIGHT * 0.4)
        dialog_x, dialog_y = (SCREEN_WIDTH - dialog_w) // 2, (SCREEN_HEIGHT - dialog_h) // 2
        
        cv2.rectangle(frame, (dialog_x, dialog_y), (dialog_x + dialog_w, dialog_y + dialog_h), PROBE_BACKGROUND_COLOR, -1)
        cv2.rectangle(frame, (dialog_x, dialog_y), (dialog_x + dialog_w, dialog_y + dialog_h), (100, 100, 100), 2)
        
        # Render Question (Hebrew support) with wrapping
        # Calculate position for text (centered in top half of dialog)
        text_y = dialog_y + int(dialog_h * 0.2)
        max_text_width = int(dialog_w * 0.9)
        
        # Wrap text logic
        words = question.split()
        lines = []
        current_line = []
        
        # Temporary font for measurement (approximation)
        font_scale = 1.0 # Base scale
        font_size = 32
        
        # We need to estimate width. Since we use PIL in _render_hebrew_text, 
        # we should ideally measure there, but for now we'll use a heuristic or simple accumulation
        # Let's just accumulate and split blindly for now, or use a simple char count estimate
        # Better: use the _render_hebrew_text to render line by line
        
        current_line_words = []
        for word in words:
            current_line_words.append(word)
            # Rough estimation: 15 pixels per char at size 32
            estimated_width = sum(len(w) for w in current_line_words) * 15 + (len(current_line_words)-1) * 10
            
            if estimated_width > max_text_width and len(current_line_words) > 1:
                # Line too long, pop last word and save line
                current_line_words.pop()
                lines.append(" ".join(current_line_words))
                current_line_words = [word]
        
        if current_line_words:
            lines.append(" ".join(current_line_words))
            
        # Render each line
        for i, line in enumerate(lines):
            line_y = text_y + i * 45 # 45 pixels line spacing
            
            # Check if line contains Hebrew
            hebrew_pattern = re.compile(r'[\u0590-\u05FF]')
            is_hebrew = bool(hebrew_pattern.search(line))
            
            # Calculate position based on alignment
            if is_hebrew:
                # Hebrew: Right Align
                # Pass the Right Boundary (Dialog Right - Margin)
                pos_x = dialog_x + dialog_w - 40
                align = True
            else:
                # English: Left Align
                # Pass the Left Boundary (Dialog Left + Margin)
                pos_x = dialog_x + 40
                align = False
            
            frame = self._render_hebrew_text(line, (pos_x, line_y), 
                                   font_size=32, color=PROBE_TEXT_COLOR, 
                                   background_image=frame, rtl_align=align)
        
        # Draw Buttons
        button_w, button_h = 120, 50
        spacing = 100
        
        # Yes Button (Right side for Hebrew?) - Let's put Yes on Right, No on Left
        yes_x = dialog_x + dialog_w // 2 + spacing // 2
        yes_y = dialog_y + int(dialog_h * 0.7)
        
        no_x = dialog_x + dialog_w // 2 - button_w - spacing // 2
        no_y = yes_y
        
        self.yes_button_rect = (yes_x, yes_y, button_w, button_h)
        self.no_button_rect = (no_x, no_y, button_w, button_h)
        
        # Draw Yes
        cv2.rectangle(frame, (yes_x, yes_y), (yes_x + button_w, yes_y + button_h), PROBE_BUTTON_COLOR, -1)
        cv2.rectangle(frame, (yes_x, yes_y), (yes_x + button_w, yes_y + button_h), (100, 100, 100), 1)
        # Center text: "כן" is approx 30px wide. To center with RTL align (which expects right edge),
        # we pass center + half_width = yes_x + 60 + 15 = yes_x + 75
        frame = self._render_hebrew_text("כן", (yes_x + button_w//2 + 15, yes_y + button_h//2 - 12), 
                               font_size=24, color=PROBE_BUTTON_TEXT_COLOR, 
                               background_image=frame, rtl_align=True)
        
        # Draw No
        cv2.rectangle(frame, (no_x, no_y), (no_x + button_w, no_y + button_h), PROBE_BUTTON_COLOR, -1)
        cv2.rectangle(frame, (no_x, no_y), (no_x + button_w, no_y + button_h), (100, 100, 100), 1)
        frame = self._render_hebrew_text("לא", (no_x + button_w//2 + 15, no_y + button_h//2 - 12), 
                               font_size=24, color=PROBE_BUTTON_TEXT_COLOR, 
                               background_image=frame, rtl_align=True)
        
        return frame

    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for probe interaction"""
        if self.state == "PROBE" and event == cv2.EVENT_LBUTTONDOWN:
            if self.yes_button_rect:
                yx, yy, yw, yh = self.yes_button_rect
                if yx <= x <= yx + yw and yy <= y <= yy + yh:
                    self.probe_answer = "Yes"
                    print("Probe Answer: Yes")
            
            if self.no_button_rect:
                nx, ny, nw, nh = self.no_button_rect
                if nx <= x <= nx + nw and ny <= y <= ny + nh:
                    self.probe_answer = "No"
                    print("Probe Answer: No")
    
    def _render_row_with_width_check(self, image, words, start_x, start_y, font, font_scale, thickness, row_idx, max_width, is_hebrew_layout=False):
        """Render a row of text with width constraints and proper word wrapping"""
        
        # Check if this row contains Hebrew
        hebrew_pattern = re.compile(r'[\u0590-\u05FF]')
        contains_hebrew = any(hebrew_pattern.search(word) for word in words)
        
        if contains_hebrew:
            # For Hebrew text, calculate right-aligned position for RTL layout
            if is_hebrew_layout:
                # Calculate right boundary for RTL text - use screen right edge minus small margin
                margin_x = int(SCREEN_WIDTH * TEXT_MARGIN_X)
                right_boundary = SCREEN_WIDTH - margin_x  # Use right edge of screen area
                
                self._render_row_with_positions(image, words, right_boundary, start_y, font, font_scale, thickness, row_idx, rtl_layout=True)
            else:
                # Standard left-aligned Hebrew (for mixed content)
                self._render_row_with_positions(image, words, start_x, start_y, font, font_scale, thickness, row_idx)
            return
        
        # For English text, check if all words fit in the available width
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
            # Check if this is Hebrew and use appropriate rendering
            hebrew_pattern = re.compile(r'[\u0590-\u05FF]')
            if hebrew_pattern.search(words[0]):
                #print(f"DEBUG: Single Hebrew word in fitted row: '{words[0]}'")
                # Use Hebrew rendering for single word
                self._render_row_with_positions(image, words, start_x, start_y, font, font_scale, thickness, row_idx)
                return
            else:
                # English word - use OpenCV
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
        
        # Check if any word is Hebrew - if so, use Hebrew rendering for entire row
        hebrew_pattern = re.compile(r'[\u0590-\u05FF]')
        contains_hebrew = any(hebrew_pattern.search(word) for word in words)
        
        if contains_hebrew:
            #print(f"DEBUG: Hebrew words in fitted row: {words}")
            # Use Hebrew rendering for entire row instead of word-by-word
            self._render_row_with_positions(image, words, start_x, start_y, font, font_scale, thickness, row_idx)
            return
        
        # English words - render with calculated spacing
        current_x = start_x
        
        for word_idx, word in enumerate(words):
            word_size = cv2.getTextSize(word, font, font_scale, thickness)[0]
            
            # Render word (English only at this point)
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

    def _render_hebrew_text(self, text, position, font_size = TEXT_FONT_SIZE, color = TEXT_COLOR, background_image = None, image_width =SCREEN_WIDTH, image_height = SCREEN_HEIGHT, rtl_align=False):
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

            # Load Hebrew-compatible font - be more aggressive about finding fonts
            hebrew_font = None
            hebrew_font_paths = [
                "C:/Windows/Fonts/david.ttf",    # Best Hebrew font
                "C:/Windows/Fonts/tahoma.ttf",   # Excellent Hebrew support
                "C:/Windows/Fonts/arial.ttf",   # Good Hebrew support
                "C:/Windows/Fonts/calibri.ttf", # Good Hebrew support
                "tahoma.ttf", "arial.ttf", "calibri.ttf"  # System font fallbacks
            ]

            font_loaded = False
            for font_path in hebrew_font_paths:
                try:
                    # Test if the font actually supports Hebrew by trying to render a Hebrew character
                    test_font = ImageFont.truetype(font_path, font_size)
                    
                    # Test render a simple Hebrew character
                    test_img = Image.new("RGB", (50, 50), (255, 255, 255))
                    test_draw = ImageDraw.Draw(test_img)
                    test_draw.text((10, 10), "ה", font=test_font, fill=(0, 0, 0))
                    
                    # If we get here without exception, the font works
                    hebrew_font = test_font
                    #print(f"DEBUG: Successfully loaded and tested Hebrew font: {os.path.basename(font_path)}")
                    font_loaded = True
                    break
                    
                except Exception as e:
                    #print(f"DEBUG: Font {font_path} failed: {e}")
                    continue
            if not font_loaded:
                #print("DEBUG: No Hebrew font found, using default - Hebrew may show as ???")
                hebrew_font = ImageFont.load_default()
                
                # Try one more fallback with a larger default font
                try:
                    hebrew_font = ImageFont.truetype("arial.ttf", font_size)
                    #print("DEBUG: Using system arial.ttf as final fallback")
                except:
                    pass            # Handle RTL text properly
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

            # Adjust position for RTL alignment if needed
            render_position = position
            if rtl_align:
                # For RTL: position[0] is the RIGHT boundary where text should END
                bbox = draw.textbbox((0, 0), display_text, font=hebrew_font)
                text_width = bbox[2] - bbox[0]
                # Calculate text start position: right_boundary - text_width
                text_start_x = position[0] - text_width
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


    def _render_row_with_positions(self, image, words, start_x, start_y, font, font_scale, thickness, row_idx, rtl_layout=False):
        
        """Render a row of text with Hebrew support and capture precise word positions"""
        
        # Check if any word contains Hebrew - be very thorough
        hebrew_pattern = re.compile(r'[\u0590-\u05FF]')
        contains_hebrew = any(hebrew_pattern.search(word) for word in words)
        
        #print(f"DEBUG: Rendering row with {len(words)} words")
        #print(f"DEBUG: First few words: {words[:3] if len(words) > 0 else 'None'}")
        #print(f"DEBUG: Contains Hebrew: {contains_hebrew}")

        # Double check - if we have Hebrew characters anywhere, force Hebrew rendering
        row_text_check = " ".join(words)
        if not contains_hebrew and hebrew_pattern.search(row_text_check):
            contains_hebrew = True
            #print("DEBUG: Force-enabled Hebrew rendering after full text check")
        
        if contains_hebrew:
            # Use PIL for Hebrew text rendering
            row_text = " ".join(words)
            
            # Calculate position for PIL (PIL uses top-left, OpenCV uses bottom-left)
            pil_y = start_y - TEXT_FONT_SIZE  # Adjust for baseline difference
            
            # For RTL layout, start_x is the RIGHT boundary where text should END
            render_x = start_x
            
            # Render entire row with PIL
            #print(f"DEBUG: About to render Hebrew row: '{row_text[:30]}...' at position ({render_x}, {pil_y}) RTL={rtl_layout}")
            rendered_image = self._render_hebrew_text(
                text=row_text,
                position=(render_x, pil_y),
                font_size=TEXT_FONT_SIZE,
                color=TEXT_COLOR,
                background_image=image,
                image_width=SCREEN_WIDTH,
                image_height=SCREEN_HEIGHT,
                rtl_align=rtl_layout
            )
            
            # Explicitly copy the rendered image back
            if rendered_image is not None:
                image[:] = rendered_image
                #print("DEBUG: Hebrew image copied successfully")
            else:
                #print("WARNING: Hebrew rendering returned None! This will cause ??? to appear.")
                #print("DEBUG: Attempting emergency Hebrew fallback...")
                # Emergency fallback - try simple PIL rendering without RTL
                try:
                    from PIL import ImageFont, ImageDraw, Image
                    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    draw = ImageDraw.Draw(pil_image)
                    try:
                        hebrew_font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", TEXT_FONT_SIZE)
                    except:
                        hebrew_font = ImageFont.load_default()
                    draw.text((start_x, pil_y), row_text, font=hebrew_font, fill=TEXT_COLOR)
                    emergency_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                    image[:] = emergency_image
                    #print("DEBUG: Emergency Hebrew rendering successful")
                except Exception as emergency_e:
                    #print(f"ERROR: Emergency Hebrew rendering also failed: {emergency_e}")
                    # Only then fall back to OpenCV (will show ???)
                    cv2.putText(image, "Hebrew text (display error)", (start_x, start_y), font, font_scale, (0, 0, 255), thickness)
            
            # Calculate word positions for Hebrew using PIL font metrics to match visual rendering
            try:
                from PIL import ImageFont, ImageDraw, Image
                # Use the same font loading logic as _render_hebrew_text
                hebrew_font = None
                hebrew_font_paths = [
                    "C:/Windows/Fonts/david.ttf", "C:/Windows/Fonts/tahoma.ttf", 
                    "C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/calibri.ttf"
                ]
                
                for font_path in hebrew_font_paths:
                    try:
                        hebrew_font = ImageFont.truetype(font_path, TEXT_FONT_SIZE)
                        break
                    except:
                        continue
                
                if hebrew_font is None:
                    hebrew_font = ImageFont.load_default()
                
                # Create a temporary draw object to measure text
                temp_image = Image.new("RGB", (100, 100), (255, 255, 255))
                draw = ImageDraw.Draw(temp_image)
                
                # Calculate RTL positions matching _render_hebrew_text logic
                if rtl_layout:
                    # For RTL: calculate total text width and position from right
                    full_text = " ".join(words)
                    
                    # Apply RTL processing like in _render_hebrew_text
                    try:
                        from bidi.algorithm import get_display
                        import arabic_reshaper
                        reshaped_text = arabic_reshaper.reshape(full_text)
                        display_text = get_display(reshaped_text)
                    except:
                        # Fallback: simple word reversal
                        display_words = list(reversed(words))
                        display_text = ' '.join(display_words)
                    
                    # Get total text width
                    bbox = draw.textbbox((0, 0), display_text, font=hebrew_font)
                    total_text_width = bbox[2] - bbox[0]
                    
                    # For RTL: start_x is the RIGHT boundary, calculate text start position
                    text_start_x = start_x - total_text_width
                    final_x = max(text_start_x, 50)  # Ensure left margin
                    
                    # Calculate individual word positions in RTL order
                    # We must iterate through the DISPLAY text to get correct visual positions
                    # The display text is visually ordered (RTL text is reversed)
                    
                    current_x = final_x
                    
                    # Split display text by spaces to get visual word units
                    # Note: This assumes spaces are preserved and separate words
                    display_words = display_text.split()
                    
                    # We need to map these visual words back to our original logical words
                    # In a simple RTL reversal, the first visual word is the last logical word
                    
                    for i, display_word in enumerate(display_words):
                        # Measure this specific word as it appears visually
                        word_bbox = draw.textbbox((0, 0), display_word, font=hebrew_font)
                        word_width = word_bbox[2] - word_bbox[0]
                        
                        # Calculate word center position
                        word_center_x = current_x + word_width // 2
                        word_center_y = start_y - TEXT_FONT_SIZE // 2
                        
                        # Map back to logical word
                        # For RTL, the first word we see (leftmost) is the last word of the logical string
                        # Wait! bidi.get_display reverses the string for display.
                        # So "Shalom Olam" -> "maloO molahS"
                        # But split() gives ["maloO", "molahS"]
                        # So the first item in the list is the RIGHTMOST word visually?
                        # No, split() splits by whitespace.
                        # If string is "maloO molahS", split is ["maloO", "molahS"]
                        # "maloO" is at index 0.
                        # When rendering "maloO molahS" at (x,y):
                        # "maloO" is drawn first (at left), then space, then "molahS" (at right).
                        # So index 0 is LEFTMOST.
                        # In Hebrew RTL, the first logical word "Shalom" should appear on the RIGHT.
                        # So the RIGHTMOST visual word corresponds to logical index 0.
                        # The LEFTMOST visual word corresponds to logical index N.
                        
                        # So: i=0 (Leftmost visual) -> Logical index = len(words) - 1
                        # i = len - 1 (Rightmost visual) -> Logical index = 0
                        
                        logical_index = len(words) - 1 - i
                        
                        if 0 <= logical_index < len(words):
                            original_word = words[logical_index]
                            
                            self.word_positions.append({
                                'word': original_word,
                                'x': word_center_x,
                                'y': word_center_y,
                                'row': row_idx,
                                'col': logical_index,
                                'width': int(word_width),
                                'height': TEXT_FONT_SIZE,
                                'page': self.current_page
                            })
                        
                        # Move x pointer
                        current_x += word_width
                        
                        # Add space width
                        if i < len(display_words) - 1:
                             space_bbox = draw.textbbox((0, 0), " ", font=hebrew_font)
                             current_x += space_bbox[2] - space_bbox[0]
                else:
                    # For LTR Hebrew (mixed content): use left-aligned positions
                    current_x = start_x
                    for word_idx, word in enumerate(words):
                        # Get actual word width using PIL
                        word_bbox = draw.textbbox((0, 0), word, font=hebrew_font)
                        word_width = word_bbox[2] - word_bbox[0]
                        
                        # Calculate word center position
                        word_center_x = current_x + word_width // 2
                        word_center_y = start_y - TEXT_FONT_SIZE // 2
                        
                        self.word_positions.append({
                            'word': word,
                            'x': word_center_x,
                            'y': word_center_y,
                            'row': row_idx,
                            'col': word_idx,
                            'width': int(word_width),
                            'height': TEXT_FONT_SIZE,
                            'page': self.current_page
                        })
                        
                        # Move to next word position
                        current_x += word_width
                        if word_idx < len(words) - 1:
                            space_bbox = draw.textbbox((0, 0), " ", font=hebrew_font)
                            current_x += space_bbox[2] - space_bbox[0]
                            

            except Exception as e:
                print(f"ERROR in Hebrew word position calculation: {e}")
                # Fallback to simple approximation
                current_x = start_x
                font_size_px = TEXT_FONT_SIZE * 0.6
                
                for word_idx, word in enumerate(words):
                    word_width = len(word) * font_size_px * 0.8 if hebrew_pattern.search(word) else len(word) * font_size_px * 0.6
                    word_center_x = current_x + word_width // 2
                    word_center_y = start_y - font_size_px // 2
                    
                    self.word_positions.append({
                        'word': word,
                        'x': word_center_x,
                        'y': word_center_y,
                        'row': row_idx,
                        'col': word_idx,
                        'width': int(word_width),
                        'height': TEXT_FONT_SIZE,
                        'page': self.current_page
                    })
                    
                    current_x += word_width + font_size_px * 0.3
        
        else:
            # Use original OpenCV rendering for English
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
            return None, None, None # gaze point, blink, pupil size
        
        face_landmarks = results.multi_face_landmarks[0]
        
        # Extract features (simplified - adapt to your feature extraction logic)
        features = self._extract_features(face_landmarks, frame.shape)
        
        
        # Calculate pupil size and blink detection
        prediction = self._predict_gaze_point(features)
        pupil_size = self._calculate_pupil_size(face_landmarks, frame.shape)
        current_blink = self._detect_blink(face_landmarks)
        blink_frequency = self._update_blink_frequency(current_blink)

        return prediction, pupil_size, current_blink

    def _predict_gaze_point(self, features):
        """Predict gaze point from features using the trained model"""
        if self.model is None or self.scaler is None:
            return None
            
        try:
            # Feature Engineering (Must match model_training.py logic)
            # features = [norm_L_x, norm_L_y, norm_R_x, norm_R_y, pitch, yaw, tvect_x, tvect_y, tvect_z]
            
            norm_L_x = features[0]
            norm_L_y = features[1]
            norm_R_x = features[2]
            norm_R_y = features[3]
            tvect_x = features[6]
            tvect_y = features[7]
            
            # Calculate average iris positions
            avg_iris_x = (norm_L_x + norm_R_x) / 2
            avg_iris_y = (norm_L_y + norm_R_y) / 2
            
            # Create interaction terms
            tvect_avg_x_inter = tvect_x * avg_iris_x
            tvect_avg_y_inter = tvect_y * avg_iris_y
            
            # Add to features list
            features_extended = features + [tvect_avg_x_inter, tvect_avg_y_inter]
            
            # Scale features
            features_scaled = self.scaler.transform([features_extended])
            
            # Predict
            prediction = self.model.predict(features_scaled)[0]
            
            # Apply smoothing
            if self.smoothed_x is None:
                self.smoothed_x = prediction[0]
                self.smoothed_y = prediction[1]
            else:
                self.smoothed_x = self.smoothed_x * (1 - self.smoothing_factor) + prediction[0] * self.smoothing_factor
                self.smoothed_y = self.smoothed_y * (1 - self.smoothing_factor) + prediction[1] * self.smoothing_factor
                
            return (self.smoothed_x, self.smoothed_y)
            
        except Exception as e:
            print(f"Prediction error: {e}") # Suppress spam
            return None

    def _detect_blink(self, landmarks):
        """Detect blink using Eye Aspect Ratio (EAR)"""
        # Simplified blink detection
        # For now, we'll return False to prevent crashes
        # A proper implementation would calculate EAR from eye landmarks
        return False

    def _update_blink_frequency(self, is_blink):
        """Update blink frequency metric"""
        # Simplified implementation
        return 0.0

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
            
            return avg_pupil_size
            
        except Exception as e:
            print(f"Error calculating pupil size: {e}")
            return 0.0

    def _export_fixation_data(self, page_id, probe_answer):
        """Export fixation data for the current page with probe answer"""
        import csv
        import os
        from datetime import datetime
        
        # Create analysis directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"eye_movement_data_{page_id}_{timestamp}.csv"
        full_path = os.path.join(self.output_dir, filename)
        
        print(f"Exporting data for page {page_id} to {full_path}")
        
        try:
            with open(full_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                # Write header
                header = [
                    'timestamp', 'x', 'y', 'duration', 'pupil_size', 
                    'fixated_word', 'page_id', 'probe_answer'
                ]
                writer.writerow(header)
                
                # Write fixation data
                for fixation in self.movement_analyzer.fixations:
                    # Get fixated word
                    word = self.movement_analyzer.get_fixated_word(fixation['x'], fixation['y'])
                    
                    # Filter out Unknown words to reduce noise
                    if word == "Unknown":
                        continue
                    
                    row = [
                        fixation.get('start_time', 0),
                        fixation['x'],
                        fixation['y'],
                        fixation['duration'],
                        fixation.get('pupil_size', 0),
                        word,
                        page_id,
                        probe_answer
                    ]
                    writer.writerow(row)
            
            print(f"Successfully exported {len(self.movement_analyzer.fixations)} fixations")
            
            # Clear data for next page
            self.movement_analyzer.fixations = []
            self.movement_analyzer.saccades = []
            self.movement_analyzer.gaze_history.clear()
            
            return True
            
        except Exception as e:
            print(f"Error exporting data: {e}")
            return False

    def run_text_reading_analysis(self):
        """Main loop for text reading analysis with probe support"""
        print("Starting text reading analysis...")
        
        # Initialize camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
            
        # Set camera resolution to match recording/training requirements
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        
        # Check model status
        if self.model is None:
            print("WARNING: No gaze prediction model loaded! Gaze tracking will not work.")
        
        # Start recording
        if self.video_recorder:
            self.video_recorder.start_recording()
        
        # Create window
        cv2.namedWindow("Reading Analysis", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Reading Analysis", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.setMouseCallback("Reading Analysis", self._mouse_callback)
        
        # Initialize state
        self.state = "READING"
        self.current_page = 0
        self.probe_answer = None
        
        # Create text pages if not already done
        if not self.text_pages:
            self._create_text_pages()
            
        if not self.text_pages:
            print("Error: No text pages created")
            return

        # Initialize movement analyzer
        self.movement_analyzer = EyeMovementAnalyzer(
            window_width=SCREEN_WIDTH,
            window_height=SCREEN_HEIGHT,
            text_reading_mode=True,
            reading_direction='rtl' if self.is_rtl_text else 'ltr'
        )
        
        # Main loop
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Flip frame for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Process frame
            prediction, pupil_size, blink = self._detect_face_and_predict(frame)
            
            # Update movement analyzer
            if prediction:
                # self.movement_analyzer.update(prediction, pupil_size, blink)
                smoothed_point = self.movement_analyzer.add_gaze_point(prediction[0], prediction[1])
                self.movement_analyzer.add_pupil_size(pupil_size)
                
                # Update prediction variable to use the Kalman-filtered coordinates
                prediction = (smoothed_point.x, smoothed_point.y)
            
            # Render based on state
            display_frame = None
            if self.state == "READING":
                # Render text page
                display_frame = self._render_text_page()
                
                # Overlay gaze point (optional, for debugging)
                if prediction:
                    cv2.circle(display_frame, (int(prediction[0]), int(prediction[1])), 10, (255, 0, 0), -1)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    break
                elif key == ord('d') or key == ord('D'):  # Next
                    # Check if we need to show a probe
                    current_metadata = self.page_metadata[self.current_page]
                    if current_metadata.get('question'):
                        self.state = "PROBE"
                        self.probe_answer = None
                        print(f"Switching to PROBE state for page {self.current_page}")
                    else:
                        # No question, just export and move on
                        # Ensure any open fixation is recorded
                        self.movement_analyzer.end_reading_session()
                        self._export_fixation_data(current_metadata['page_id'], "N/A")
                        self.current_page += 1
                        if self.current_page >= len(self.text_pages):
                            print("All pages completed")
                            break
                elif key == ord('a') or key == ord('A'):  # Previous
                    if self.current_page > 0:
                        self.current_page -= 1
                        # Clear data when going back? Maybe not.
                
            elif self.state == "PROBE":
                # Render probe window
                current_metadata = self.page_metadata[self.current_page]
                display_frame = self._render_probe_window(frame.copy(), current_metadata['question'])
                
                # Check if answer received
                if self.probe_answer:
                    # Export data
                    # Ensure any open fixation is recorded
                    self.movement_analyzer.end_reading_session()
                    self._export_fixation_data(current_metadata['page_id'], self.probe_answer)
                    
                    # Move to next page
                    self.current_page += 1
                    self.state = "READING"
                    self.probe_answer = None
                    
                    if self.current_page >= len(self.text_pages):
                        print("All pages completed")
                        break
                
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    break
            
            # Record frame
            if display_frame is not None:
                if self.video_recorder:
                    self.video_recorder.record_frame(camera_frame=frame, overlay_frame=display_frame)
                cv2.imshow("Reading Analysis", display_frame)
            
        # Cleanup
        if self.video_recorder:
            self.video_recorder.stop_recording()
        cap.release()
        cv2.destroyAllWindows()