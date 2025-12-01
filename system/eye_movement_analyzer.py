import cv2
import numpy as np
import time
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import math
from config import *

@dataclass
class GazePoint:
    """Data class for storing gaze point information"""
    x: float
    y: float
    timestamp: float
    velocity: float = 0.0
    is_fixation: bool = False

@dataclass
class MovementMetrics:
    """Data class for storing movement analysis results"""
    average_velocity: float = 0.0
    fixation_count: int = 0
    total_fixation_duration: float = 0.0
    saccade_count: int = 0
    reading_speed_wpm: float = 0.0
    current_velocity: float = 0.0

class KalmanFilter:
    """Enhanced Kalman filter optimized for text reading gaze prediction"""
    def __init__(self, text_reading_mode=False, reading_direction="ltr"):
        # State: [x, y, vx, vy] - position and velocity
        self.state = np.zeros(4)
        self.text_reading_mode = text_reading_mode
        self.reading_direction = reading_direction
        self.is_rtl = (reading_direction == "rtl")
        
        # Text reading mode parameters (same for LTR and RTL)
        self.P = np.eye(4) * 500   # Lower initial uncertainty for more responsive tracking
        self.Q = np.eye(4) * 0.1  # Reduced process noise for smoother reading patterns (was 0.1)
        self.R = np.eye(2) * 5    # Measurement noise (lower = more responsive)
            
        self.F = np.array([[1, 0, 1, 0],  # State transition model
                          [0, 1, 0, 1],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]])
        self.H = np.array([[1, 0, 0, 0],  # Observation model
                          [0, 1, 0, 0]])
        self.initialized = False
        
        # Text reading specific enhancements - adjusted for reading direction
        if self.is_rtl:
            self.reading_direction_bias = -1.2  # Bias towards leftward movement for RTL
            self.expected_saccade_direction = -1  # Negative for leftward saccades
        else:
            self.reading_direction_bias = 1.0   # Bias towards rightward movement for LTR
            self.expected_saccade_direction = 1   # Positive for rightward saccades
            
        self.line_return_detected = False
        self.last_prediction = None

    def predict(self):
        """Enhanced prediction with text reading optimizations"""
        if not self.initialized:
            return None
        
        # Standard Kalman prediction
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        predicted_x, predicted_y = self.state[0], self.state[1]
        
        # Text reading specific enhancements
        if self.text_reading_mode:
            # Apply reading direction bias (slight rightward movement preference)
            predicted_x += self.reading_direction_bias
            
            # Detect potential line returns (large leftward + downward movement)
            if self.last_prediction:
                dx = predicted_x - self.last_prediction[0]
                dy = predicted_y - self.last_prediction[1]
                
                # If moving significantly left and down, likely a line return
                if dx < -100 and dy > 20:
                    self.line_return_detected = True
                    # Reduce horizontal velocity prediction for line return
                    self.state[2] *= 0.3  # Reduce horizontal velocity
                else:
                    self.line_return_detected = False
        
        result = (predicted_x, predicted_y)
        self.last_prediction = result
        return result

    def update(self, measurement):
        """Update filter with new measurement"""
        if not self.initialized:
            # Initialize with first measurement
            self.state[0] = measurement[0]
            self.state[1] = measurement[1]
            self.initialized = True
            return
        
        # Update step
        y = np.array(measurement) - self.H @ self.state  # Innovation
        S = self.H @ self.P @ self.H.T + self.R  # Innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain
        
        self.state = self.state + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

class EyeMovementAnalyzer:
    """Comprehensive eye movement analysis system"""

    def __init__(self, window_width=SCREEN_WIDTH, window_height=SCREEN_HEIGHT, text_reading_mode=True, reading_direction='ltr'):
        self.window_width = window_width
        self.window_height = window_height

        self.text_reading_mode = text_reading_mode
        
        # Movement tracking
        self.gaze_history = deque(maxlen=300)  # Store last 10 seconds at 30fps
        self.velocity_buffer = deque(maxlen=30)  # Store last 1 second of velocities
        
        # Kalman filter for prediction only (optimized for text reading)
        self.kalman_filter = KalmanFilter(text_reading_mode=True, reading_direction=reading_direction)
        
        # Fixation detection parameters
        self.fixation_threshold_pixels = 30  # Max movement for fixation
        self.fixation_min_duration = FIXATION_THRESHOLD / 1000.0  # Convert ms to seconds from config
        self.current_fixation_start = None
        self.current_fixation_position = None
        
        # Reading analysis
        self.fixations = []
        self.saccades = []
        self.reading_start_time = None
        
        # Performance metrics
        self.metrics = MovementMetrics()
        
        # Text analysis (for reading speed calculation)
        self.estimated_words_per_line = 12  # Approximate words per line
        self.estimated_lines_read = 0
        
        # Text reading specific variables for detailed analysis
        self.text_content = None  # Store current page text for word detection
        self.word_positions = []  # Store word positions for mapping fixations to words
        self.blink_count = 0
        self.last_blink_time = 0
        self.pupil_size_history = deque(maxlen=100)  # Store recent pupil sizes
        self.text_onset_time = None  # Time when text was first displayed
        
        # Velocity Tracking for Adaptive Kalman Filter
        self.last_raw_x = None
        self.last_raw_y = None
        self.last_timestamp = None
        self.smoothed_velocity = 0.0
        
    def add_gaze_point(self, x: float, y: float) -> GazePoint:
        """Add new gaze point and analyze movement"""
        current_time = time.time()
        
        # 1. Calculate Smoothed Velocity from RAW data
        # This detects intent to move without being fooled by filter lag or raw noise
        raw_velocity = 0.0
        if self.last_raw_x is not None and self.last_timestamp is not None:
            dt = current_time - self.last_timestamp
            if dt > 0:
                dx = x - self.last_raw_x
                dy = y - self.last_raw_y
                dist = (dx**2 + dy**2)**0.5
                raw_velocity = dist / dt
        
        # Update history
        self.last_raw_x = x
        self.last_raw_y = y
        self.last_timestamp = current_time
        
        # Apply EMA to velocity (Alpha 0.2 = smooths out noise spikes)
        # If raw_velocity is noise (jitter), it fluctuates and averages low.
        # If raw_velocity is reading, it stays consistent and average rises.
        self.smoothed_velocity = 0.2 * raw_velocity + 0.8 * self.smoothed_velocity
        
        # Create gaze point object
        gaze_point = GazePoint(x=x, y=y, timestamp=current_time, velocity=self.smoothed_velocity)
        self.velocity_buffer.append(self.smoothed_velocity)
        
        # 2. Update Kalman Filter
        # Predict step is required to maintain covariance
        self.kalman_filter.predict()
        self.kalman_filter.update([x, y])
        
        # Use filtered position
        filtered_pos = self.kalman_filter.state[:2]
        if filtered_pos is not None:
            gaze_point.x = filtered_pos[0]
            gaze_point.y = filtered_pos[1]
        
        # 3. Update Adaptive Logic using SMOOTHED VELOCITY
        self._adaptive_kalman_update(self.smoothed_velocity)
        
        # Detect fixations
        self._detect_fixation(gaze_point)
        
        # Add to history
        self.gaze_history.append(gaze_point)
        
        # Update metrics
        self._update_metrics()
        
        return gaze_point
    
    def predict_next_gaze(self) -> Optional[Tuple[float, float]]:
        """Predict next gaze position using enhanced Kalman filter"""
        return self.kalman_filter.predict()
    
    def _adaptive_kalman_update(self, velocity: float):
        """Adaptively adjust Kalman filter parameters based on SMOOTHED velocity"""
        # Use the smoothed velocity directly to determine the state
        # This prevents noise from triggering 'saccade mode'
        
        # Adaptive parameter adjustment based on reading behavior
        if velocity < 50:  # Slow movement (fixation/careful reading)
            # Lower process noise for more stable predictions
            self.kalman_filter.Q = np.eye(4) * 0.005
            self.kalman_filter.reading_direction_bias = 0.5
        elif velocity > 200:  # Fast movement (saccades/scanning)
            # Higher process noise to respond quickly to changes
            self.kalman_filter.Q = np.eye(4) * 0.1
            self.kalman_filter.reading_direction_bias = 2.0
        else:  # Normal reading speed
            # Standard parameters
            self.kalman_filter.Q = np.eye(4) * 0.05
            self.kalman_filter.reading_direction_bias = 1.0
    
    def _calculate_distance(self, x1: float, y1: float, x2: float, y2: float) -> float:
        """Calculate Euclidean distance between two points"""
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    def _detect_fixation(self, current_point: GazePoint):
        """Detect if current point is part of a fixation"""
        if len(self.gaze_history) == 0:
            # First point - start potential fixation
            self.current_fixation_start = current_point.timestamp
            self.current_fixation_position = (current_point.x, current_point.y)
            return
        
        # Check if we're still within fixation threshold
        if self.current_fixation_position:
            distance = self._calculate_distance(
                current_point.x, current_point.y,
                self.current_fixation_position[0], self.current_fixation_position[1]
            )
            
            # Check both distance AND velocity
            # If velocity is high (>350), it's a saccade, not a fixation
            # Increased threshold from 100 to 350 to be less harsh on noise
            is_stable = (distance <= self.fixation_threshold_pixels) and (self.smoothed_velocity < 350)
            
            if is_stable:
                # Still in fixation - update position (use centroid)
                self.current_fixation_position = (
                    (self.current_fixation_position[0] + current_point.x) / 2,
                    (self.current_fixation_position[1] + current_point.y) / 2
                )
                
                # Check if we are in a valid fixation (duration exceeded)
                if self.current_fixation_start:
                    fixation_duration = current_point.timestamp - self.current_fixation_start
                    if fixation_duration >= self.fixation_min_duration:
                        current_point.is_fixation = True
                        
            else:
                # Fixation ended - check if it was long enough
                if self.current_fixation_start:
                    fixation_duration = current_point.timestamp - self.current_fixation_start
                    if fixation_duration >= self.fixation_min_duration:
                        # Valid fixation
                        self._record_fixation(
                            self.current_fixation_position[0],
                            self.current_fixation_position[1],
                            self.current_fixation_start,
                            current_point.timestamp
                        )
                        # Mark recent points as fixation (retroactively)
                        for point in self.gaze_history:
                            if point.timestamp >= self.current_fixation_start:
                                point.is_fixation = True
                
                # Start new potential fixation ONLY if velocity is reasonably low
                if self.smoothed_velocity < 350:
                    self.current_fixation_start = current_point.timestamp
                    self.current_fixation_position = (current_point.x, current_point.y)
                else:
                    self.current_fixation_start = None
                    self.current_fixation_position = None
    
    def end_reading_session(self):
        """End the current reading session and record any open fixation"""
        if self.current_fixation_start and self.current_fixation_position:
            current_time = time.time()
            fixation_duration = current_time - self.current_fixation_start
            
            if fixation_duration >= self.fixation_min_duration:
                self._record_fixation(
                    self.current_fixation_position[0],
                    self.current_fixation_position[1],
                    self.current_fixation_start,
                    current_time
                )
        
        self.current_fixation_start = None
        self.current_fixation_position = None
    
    def _record_fixation(self, x: float, y: float, start_time: float, end_time: float):
        """Record a completed fixation"""
        fixation = {
            'x': x,
            'y': y,
            'start_time': start_time,
            'end_time': end_time,
            'duration': end_time - start_time
        }
        self.fixations.append(fixation)
        
        # Update reading progress estimation
        self._estimate_reading_progress(y)
    
    def _estimate_reading_progress(self, y_position: float):
        """Estimate reading progress based on vertical position"""
        # Simple estimation: assume reading progresses top to bottom
        line_height = self.window_height / 20  # Assume ~20 lines of text
        estimated_line = y_position / line_height
        
        if estimated_line > self.estimated_lines_read:
            self.estimated_lines_read = estimated_line
    
    def _update_metrics(self):
        """Update movement metrics"""
        if len(self.velocity_buffer) > 0:
            self.metrics.current_velocity = self.velocity_buffer[-1]
            self.metrics.average_velocity = sum(self.velocity_buffer) / len(self.velocity_buffer)
        
        self.metrics.fixation_count = len(self.fixations)
        self.metrics.total_fixation_duration = sum(f['duration'] for f in self.fixations)
        
        # Calculate reading speed (words per minute)
        if self.reading_start_time and len(self.fixations) > 0:
            time_elapsed = time.time() - self.reading_start_time
            if time_elapsed > 0:
                words_read = self.estimated_lines_read * self.estimated_words_per_line
                self.metrics.reading_speed_wpm = (words_read / time_elapsed) * 60
    
    def start_reading_session(self):
        """Start a new reading session"""
        self.reading_start_time = time.time()
        self.gaze_history.clear()
        self.velocity_buffer.clear()
        self.fixations.clear()
        self.saccades.clear()
        self.estimated_lines_read = 0
        self.metrics = MovementMetrics()
        self.kalman_filter = KalmanFilter(text_reading_mode=True)  # Reset with text reading optimization
    
    def get_fixation_heatmap_data(self) -> List[Tuple[float, float, float]]:
        """Get fixation data for heatmap visualization"""
        heatmap_data = []
        for fixation in self.fixations:
            # Return x, y, intensity (based on duration)
            intensity = min(fixation['duration'] * 10, 1.0)  # Normalize to 0-1
            heatmap_data.append((fixation['x'], fixation['y'], intensity))
        return heatmap_data
    
    def get_reading_path(self) -> List[Tuple[float, float]]:
        """Get the reading path for visualization"""
        if len(self.gaze_history) < 2:
            return []
        
        # Return path of recent gaze points
        path = [(point.x, point.y) for point in list(self.gaze_history)[-50:]]
        return path
    
    def get_velocity_graph_data(self) -> List[float]:
        """Get velocity data for graph visualization"""
        return list(self.velocity_buffer)
    
    def is_currently_fixating(self) -> bool:
        """Check if user is currently fixating"""
        if len(self.gaze_history) == 0:
            return False
        return self.gaze_history[-1].is_fixation
    
    def get_current_fixation_duration(self) -> float:
        """Get duration of current fixation"""
        if not self.current_fixation_start:
            return 0.0
        return time.time() - self.current_fixation_start
    
    def classify_movement_type(self) -> str:
        """Classify current movement type"""
        if len(self.velocity_buffer) == 0:
            return "No movement"
        
        current_velocity = self.velocity_buffer[-1]
        
        if current_velocity < 30:  # pixels/second
            return "Fixation"
        elif current_velocity < SACCADE_VELOCITY_THRESHOLD:  
            return "Smooth pursuit"
        else:
            return "Saccade"
    
    def get_analysis_summary(self) -> Dict:
        """Get comprehensive analysis summary"""
        if not self.reading_start_time:
            return {"status": "No reading session active"}
        
        session_duration = time.time() - self.reading_start_time
        
        summary = {
            "session_duration": session_duration,
            "total_fixations": len(self.fixations),
            "average_fixation_duration": (
                self.metrics.total_fixation_duration / len(self.fixations) 
                if len(self.fixations) > 0 else 0
            ),
            "reading_speed_wpm": self.metrics.reading_speed_wpm,
            "average_velocity": self.metrics.average_velocity,
            "current_velocity": self.metrics.current_velocity,
            "movement_type": self.classify_movement_type(),
            "lines_read_estimate": self.estimated_lines_read,
            "fixation_rate": len(self.fixations) / session_duration if session_duration > 0 else 0
        }
        
        return summary
    
    def show_analysis_summary_gui(self):
        """Display analysis summary in a GUI popup window"""
        try:
            import tkinter as tk
            from tkinter import messagebox
            
            # Get the analysis data
            summary = self.get_analysis_summary()
            
            if "status" in summary:
                messagebox.showinfo("Analysis", summary["status"])
                return
            
            # Format the analysis for display
            analysis_text = f"""READING ANALYSIS COMPLETE!

Session Overview:
• Duration: {summary['session_duration']:.2f} seconds
• Total Fixations: {summary['total_fixations']}
• Average Fixation Duration: {summary['average_fixation_duration']:.3f} seconds

Reading Performance:
• Reading Speed: {summary['reading_speed_wpm']:.1f} Words Per Minute
• Average Eye Velocity: {summary['average_velocity']:.1f} pixels/second
• Lines Read (estimated): {summary['lines_read_estimate']:.1f}
• Fixation Rate: {summary['fixation_rate']:.2f} fixations/second

Performance Assessment:
"""
            
            # Add performance interpretation
            if summary['reading_speed_wpm'] > 250:
                analysis_text += "• Fast Reader - Above average reading speed\n"
            elif summary['reading_speed_wpm'] > 200:
                analysis_text += "• Good Reader - Average reading speed\n"
            else:
                analysis_text += "• Careful Reader - Taking time to process text\n"
            
            if summary['average_fixation_duration'] > 0.3:
                analysis_text += "• Detailed Processing - Longer fixations indicate careful reading"
            else:
                analysis_text += "• Quick Processing - Shorter fixations indicate efficient reading"
            
            # Display in popup
            messagebox.showinfo("Reading Analysis Results", analysis_text)
            print("Analysis summary displayed successfully!")
            
        except ImportError:
            print("GUI summary not available (tkinter not found)")
        except Exception as e:
            print(f"Error showing GUI summary: {e}")
    
    def set_text_content(self, text_content: str, text_start_x: int, text_start_y: int, 
                        text_width: int, line_spacing: int = 35, font_scale: float = 0.6):
        """Set the current text content and calculate word positions for fixation mapping"""
        import cv2
        
        self.text_content = text_content
        self.word_positions = []
        self.text_onset_time = time.time()
        
        # Calculate approximate word positions based on text layout
        words = text_content.split()
        current_line = ""
        y = text_start_y
        line_number = 0
        
        font = cv2.FONT_HERSHEY_DUPLEX
        thickness = 1
        
        for word in words:
            test_line = current_line + (" " if current_line else "") + word
            text_size = cv2.getTextSize(test_line, font, font_scale, thickness)[0]
            
            if text_size[0] <= text_width:
                current_line = test_line
            else:
                # Start new line
                if current_line:
                    line_number += 1
                    y += line_spacing
                current_line = word
            
            # Calculate word position
            current_line_size = cv2.getTextSize(current_line, font, font_scale, thickness)[0]
            word_size = cv2.getTextSize(word, font, font_scale, thickness)[0]
            
            # Approximate word center position
            line_x = text_start_x + (text_width - current_line_size[0]) // 2
            word_start_in_line = current_line.rfind(word)
            chars_before_word = current_line[:word_start_in_line]
            chars_before_size = cv2.getTextSize(chars_before_word, font, font_scale, thickness)[0]
            
            word_x = line_x + chars_before_size[0] + word_size[0] // 2
            word_y = y
            
            self.word_positions.append({
                'word': word,
                'x': word_x,
                'y': word_y,
                'line': line_number,
                'width': word_size[0],
                'height': word_size[1]
            })
    
    def set_actual_word_positions(self, word_positions: List[Dict]):
        """UNIFIED PIPELINE: Set actual word positions captured during rendering with cleaned words"""
        # Clean the words when storing positions
        cleaned_positions = []
        for pos in word_positions:
            # Try to clean if it's Hebrew, otherwise keep as is
            if self.is_hebrew_word(pos['word']):
                cleaned_word = self.clean_hebrew_word(pos['word'])
            else:
                cleaned_word = pos['word']
                
            # Store all words, not just Hebrew ones
            if cleaned_word and cleaned_word.strip():
                cleaned_pos = pos.copy()
                cleaned_pos['word'] = cleaned_word
                cleaned_positions.append(cleaned_pos)
        
        self.word_positions = cleaned_positions
        self.text_onset_time = time.time()
        
        # Extract text content from cleaned positions for compatibility
        if cleaned_positions:
            self.text_content = " ".join([pos['word'] for pos in cleaned_positions])
        else:
            self.text_content = ""
    
    def get_fixated_word(self, fixation_x: float, fixation_y: float) -> str:
        """Determine which word is being fixated based on gaze position (words are already cleaned)"""
        if not self.word_positions:
            return "Unknown"
        
        # Find closest word to fixation point
        min_distance = float('inf')
        closest_word = "Unknown"
        
        for word_info in self.word_positions:
            # Calculate distance from fixation to word center
            distance = math.sqrt((fixation_x - word_info['x'])**2 + (fixation_y - word_info['y'])**2)
            
            # Consider word as "fixated" if within reasonable distance
            word_threshold = max(word_info['width'], word_info['height']) * 0.8
            if distance < word_threshold and distance < min_distance:
                min_distance = distance
                closest_word = word_info['word']
        
        return closest_word
    
    def add_pupil_size(self, pupil_size: float):
        """Add pupil size measurement for analysis"""
        self.pupil_size_history.append(pupil_size)
    
    def record_blink(self):
        """Record a blink event"""
        current_time = time.time()
        self.blink_count += 1
        self.last_blink_time = current_time
    
    def get_blink_frequency(self) -> float:
        """Calculate blink frequency (blinks per minute)"""
        if not self.text_onset_time:
            return 0.0
        
        session_duration = time.time() - self.text_onset_time
        if session_duration > 0:
            return (self.blink_count / session_duration) * 60  # blinks per minute
        return 0.0
    
    def get_average_pupil_size(self) -> float:
        """Get average pupil size from recent measurements"""
        if not self.pupil_size_history:
            return 0.0
        return sum(self.pupil_size_history) / len(self.pupil_size_history)
    
    def clean_hebrew_word(self, word: str) -> str:
        """Extract only Hebrew letters from a word, removing punctuation and other marks"""
        if not word or word == "Unknown":
            return word
        
        # Keep only Hebrew characters (Unicode range U+0590 to U+05FF)
        cleaned_word = ''.join(char for char in word if '\u0590' <= char <= '\u05FF')
        
        # Return cleaned word if it has Hebrew characters, otherwise return "Unknown"
        return cleaned_word if cleaned_word else "Unknown"
    
    def is_hebrew_word(self, word: str) -> bool:
        """Check if a word contains Hebrew characters"""
        if not word or word == "Unknown":
            return False
        # Check if any character in the word is Hebrew
        hebrew_chars = any('\u0590' <= char <= '\u05FF' for char in word)
        return hebrew_chars
    
    def export_text_reading_csv(self, filename: str = None):
        """Export text reading specific CSV with fixation-level data for Hebrew words only"""
        import csv
        import os
        from datetime import datetime
        
        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"eye_movement_data_{timestamp}.csv"
        
        # Ensure CSV extension
        if not filename.endswith('.csv'):
            filename += '.csv'
        
        # Create data directory if it doesn't exist
        data_dir = "movement_data"
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        
        full_path = os.path.join(data_dir, filename)
        
        # NOTE: This method is DEPRECATED and never called. 
        # CSV export is now handled by prediction.py with proper grouping.
        # Prepare fixation-level data with Hebrew words only and grouping
        csv_data = []
        last_word = None
        last_word_start_time = None
        accumulated_duration = 0
        accumulated_x = 0
        accumulated_y = 0
        fixation_count_for_word = 0
        
        for i, fixation in enumerate(self.fixations):
            # Get the word being fixated
            fixated_word = self.get_fixated_word(fixation['x'], fixation['y'])
            
            # Skip Unknown words and non-Hebrew words completely
            if fixated_word == "Unknown" or not self.is_hebrew_word(fixated_word):
                continue
            
            # Additional safety check - ensure word contains only Hebrew letters
            cleaned_word = self.clean_hebrew_word(fixated_word)
            if cleaned_word == "Unknown" or not self.is_hebrew_word(cleaned_word):
                continue
            
            # Use the cleaned word for processing
            fixated_word = cleaned_word
            
            # Check if this is the same word as the previous one
            if fixated_word == last_word and last_word is not None:
                # Same word - accumulate data
                accumulated_duration += fixation['duration']
                accumulated_x += fixation['x']
                accumulated_y += fixation['y']
                fixation_count_for_word += 1
            else:
                # Different word - first save the previous word if it exists
                if last_word is not None:
                    # Calculate averages for the accumulated word
                    avg_x = accumulated_x / fixation_count_for_word
                    avg_y = accumulated_y / fixation_count_for_word
                    
                    # Calculate time from stimulus onset
                    time_from_onset = 0
                    if self.text_onset_time:
                        time_from_onset = last_word_start_time - self.text_onset_time
                    
                    csv_row = {
                        'Fixation_Order': len(csv_data) + 1,
                        'Fixated_Word': last_word,
                        'Fixation_X_Screen': round(avg_x, 2),
                        'Fixation_Y_Screen': round(avg_y, 2),
                        'Fixation_Duration': round(accumulated_duration * 1000, 2),  # Convert to milliseconds
                        'Time_from_Stimulus_Onset': round(time_from_onset * 1000, 2),  # Convert to milliseconds
                        'Pupil_Size': round(self.get_average_pupil_size(), 2),
                        'Blink_Frequency': round(self.get_blink_frequency(), 2)
                    }
                    
                    csv_data.append(csv_row)
                
                # Start new word accumulation
                last_word = fixated_word
                last_word_start_time = fixation['start_time']
                accumulated_duration = fixation['duration']
                accumulated_x = fixation['x']
                accumulated_y = fixation['y']
                fixation_count_for_word = 1
        
        # Don't forget the last word
        if last_word is not None and self.is_hebrew_word(last_word):
            avg_x = accumulated_x / fixation_count_for_word
            avg_y = accumulated_y / fixation_count_for_word
            
            time_from_onset = 0
            if self.text_onset_time:
                time_from_onset = last_word_start_time - self.text_onset_time
            
            csv_row = {
                'Fixation_Order': len(csv_data) + 1,
                'Fixated_Word': last_word,
                'Fixation_X_Screen': round(avg_x, 2),
                'Fixation_Y_Screen': round(avg_y, 2),
                'Fixation_Duration': round(accumulated_duration * 1000, 2),  # Convert to milliseconds
                'Time_from_Stimulus_Onset': round(time_from_onset * 1000, 2),  # Convert to milliseconds
                'Pupil_Size': round(self.get_average_pupil_size(), 2),
                'Blink_Frequency': round(self.get_blink_frequency(), 2)
            }
            
            csv_data.append(csv_row)
        
        # Write to CSV
        if csv_data:
            fieldnames = ['Fixation_Order', 'Fixated_Word', 'Fixation_X_Screen', 'Fixation_Y_Screen', 
                         'Fixation_Duration', 'Time_from_Stimulus_Onset', 'Pupil_Size', 'Blink_Frequency']
            
            with open(full_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(csv_data)
            
            print(f"Hebrew text reading data exported to: {full_path}")
            print(f"Total Hebrew word fixations: {len(csv_data)}")
            print(f"Hebrew words recognized: {len(set(row['Fixated_Word'] for row in csv_data))}")
            print(f"Blink frequency: {self.get_blink_frequency():.1f} blinks/min")
            
            return full_path
        else:
            print("No Hebrew word fixation data available for export")
            return None
    
    def export_data_to_csv(self, filename: str = None):
        """Export all collected movement data to CSV file for model training"""
        import csv
        import os
        from datetime import datetime
        
        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"eye_movement_data_{timestamp}.csv"
        
        # Ensure CSV extension
        if not filename.endswith('.csv'):
            filename += '.csv'
        
        # Create data directory if it doesn't exist
        data_dir = "movement_data"
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        
        full_path = os.path.join(data_dir, filename)
        
        # Prepare data for CSV export
        csv_data = []
        
        # Session metadata
        session_duration = time.time() - self.reading_start_time if self.reading_start_time else 0
        summary = self.get_analysis_summary()
        
        # Export each gaze point with comprehensive data
        for i, point in enumerate(self.gaze_history):
            # Find corresponding fixation data if this point is part of a fixation
            fixation_data = None
            for fixation in self.fixations:
                if (fixation['start_time'] <= point.timestamp <= fixation['end_time'] and
                    abs(fixation['x'] - point.x) < self.fixation_threshold_pixels and
                    abs(fixation['y'] - point.y) < self.fixation_threshold_pixels):
                    fixation_data = fixation
                    break
            
            # Calculate relative position in window (normalized 0-1)
            norm_x = point.x / self.window_width
            norm_y = point.y / self.window_height
            
            # Calculate time since session start
            time_from_start = point.timestamp - self.reading_start_time if self.reading_start_time else 0
            
            # Movement direction (if not first point)
            movement_direction_x = 0
            movement_direction_y = 0
            if i > 0:
                prev_point = list(self.gaze_history)[i-1]
                movement_direction_x = point.x - prev_point.x
                movement_direction_y = point.y - prev_point.y
            
            # Acceleration (if we have enough points)
            acceleration = 0
            if i > 1:
                prev_point = list(self.gaze_history)[i-1]
                prev_prev_point = list(self.gaze_history)[i-2]
                if prev_point.velocity > 0:
                    acceleration = point.velocity - prev_point.velocity
            
            csv_row = {
                # Basic gaze data
                'point_id': i,
                'timestamp': point.timestamp,
                'time_from_start': time_from_start,
                'x_pixel': point.x,
                'y_pixel': point.y,
                'x_normalized': norm_x,
                'y_normalized': norm_y,
                
                # Movement characteristics
                'velocity_px_per_sec': point.velocity,
                'movement_direction_x': movement_direction_x,
                'movement_direction_y': movement_direction_y,
                'acceleration': acceleration,
                'movement_type': self._classify_point_movement(point.velocity),
                
                # Fixation data
                'is_fixation': point.is_fixation,
                'fixation_duration': fixation_data['duration'] if fixation_data else 0,
                'fixation_center_x': fixation_data['x'] if fixation_data else 0,
                'fixation_center_y': fixation_data['y'] if fixation_data else 0,
                
                # Reading context
                'estimated_line': point.y / (self.window_height / 20),  # Estimate line number
                'estimated_word_position': point.x / (self.window_width / self.estimated_words_per_line),
                
                # Session metrics (same for all points)
                'session_duration': session_duration,
                'total_fixations': len(self.fixations),
                'average_velocity': self.metrics.average_velocity,
                'reading_speed_wpm': self.metrics.reading_speed_wpm,
                'fixation_rate': summary.get('fixation_rate', 0)
            }
            
            csv_data.append(csv_row)
        
        # Write to CSV
        if csv_data:
            fieldnames = csv_data[0].keys()
            
            with open(full_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(csv_data)
            
            print(f"Eye movement data exported to: {full_path}")
            print(f"Data points: {len(csv_data)}")
            print(f"Session duration: {session_duration:.2f} seconds")
            print(f"Total fixations: {len(self.fixations)}")
            print(f"Reading speed: {self.metrics.reading_speed_wpm:.1f} WPM")
            
            return full_path
        else:
            print("No data to export")
            return None
    
    def _classify_point_movement(self, velocity: float) -> str:
        """Classify individual point movement type based on velocity"""
        if velocity < 30:  # pixels/second
            return "fixation"
        elif velocity < 300:
            return "smooth_pursuit"
        else:
            return "saccade"
    
    def finish_analysis_session(self, auto_export: bool = True) -> str:
        """Finish the analysis session and optionally export data"""
        if not self.reading_start_time:
            print("No active reading session to finish")
            return None
        
        # Final metrics update
        self._update_metrics()
        
        # Print session summary
        summary = self.get_analysis_summary()
        print("\n" + "="*50)
        print("EYE MOVEMENT ANALYSIS COMPLETE")
        print("="*50)
        print(f"Session Duration: {summary['session_duration']:.2f} seconds")
        print(f"Total Fixations: {summary['total_fixations']}")
        print(f"Average Fixation Duration: {summary['average_fixation_duration']:.3f} seconds")
        print(f"Reading Speed: {summary['reading_speed_wpm']:.1f} WPM")
        print(f"Average Velocity: {summary['average_velocity']:.1f} px/sec")
        print(f"Lines Read (estimated): {summary['lines_read_estimate']:.1f}")
        print(f"Fixation Rate: {summary['fixation_rate']:.2f} fixations/sec")
        print("="*50)
        
        # Auto-export to CSV
        csv_file = None
        if auto_export:
            csv_file = self.export_data_to_csv()
        
        return csv_file
