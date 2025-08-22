import cv2
import numpy as np
import time
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import math

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
    def __init__(self, text_reading_mode=False):
        # State: [x, y, vx, vy] - position and velocity
        self.state = np.zeros(4)
        self.text_reading_mode = text_reading_mode
        
        if text_reading_mode:
            # Optimized parameters for text reading
            self.P = np.eye(4) * 500   # Lower initial uncertainty for more responsive tracking
            self.Q = np.eye(4) * 0.05  # Lower process noise for smoother reading patterns
            self.R = np.eye(2) * 15    # Slightly higher measurement noise to reduce jitter
        else:
            # Standard parameters
            self.P = np.eye(4) * 1000  # Error covariance matrix
            self.Q = np.eye(4) * 0.1   # Process noise
            self.R = np.eye(2) * 10    # Measurement noise
            
        self.F = np.array([[1, 0, 1, 0],  # State transition model
                          [0, 1, 0, 1],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]])
        self.H = np.array([[1, 0, 0, 0],  # Observation model
                          [0, 1, 0, 0]])
        self.initialized = False
        
        # Text reading specific enhancements
        self.reading_direction_bias = 1.0  # Bias towards rightward movement
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
    
    def __init__(self, window_width=1920, window_height=1080):
        self.window_width = window_width
        self.window_height = window_height
        
        # Movement tracking
        self.gaze_history = deque(maxlen=300)  # Store last 10 seconds at 30fps
        self.velocity_buffer = deque(maxlen=30)  # Store last 1 second of velocities
        
        # Kalman filter for prediction only (optimized for text reading)
        self.kalman_filter = KalmanFilter(text_reading_mode=True)
        
        # Fixation detection parameters
        self.fixation_threshold_pixels = 30  # Max movement for fixation
        self.fixation_min_duration = 0.1     # Minimum fixation duration (seconds)
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
        
    def add_gaze_point(self, x: float, y: float) -> GazePoint:
        """Add new gaze point and analyze movement"""
        current_time = time.time()
        
        # Create gaze point
        gaze_point = GazePoint(x=x, y=y, timestamp=current_time)
        
        # Calculate velocity if we have previous points
        if len(self.gaze_history) > 0:
            prev_point = self.gaze_history[-1]
            distance = self._calculate_distance(x, y, prev_point.x, prev_point.y)
            time_delta = current_time - prev_point.timestamp
            
            if time_delta > 0:
                velocity = distance / time_delta  # pixels per second
                gaze_point.velocity = velocity
                self.velocity_buffer.append(velocity)
        
        # Update Kalman filter for prediction with adaptive parameters
        self._adaptive_kalman_update(gaze_point)
        self.kalman_filter.update([x, y])
        
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
    
    def _adaptive_kalman_update(self, current_point: GazePoint):
        """Adaptively adjust Kalman filter parameters based on reading patterns"""
        if len(self.gaze_history) < 5:  # Need some history for adaptation
            return
        
        # Analyze recent movement pattern
        recent_points = list(self.gaze_history)[-5:]
        velocities = [point.velocity for point in recent_points if point.velocity > 0]
        
        if len(velocities) < 3:
            return
        
        avg_velocity = sum(velocities) / len(velocities)
        
        # Adaptive parameter adjustment based on reading behavior
        if avg_velocity < 50:  # Slow movement (fixation/careful reading)
            # Lower process noise for more stable predictions
            self.kalman_filter.Q = np.eye(4) * 0.02
            self.kalman_filter.reading_direction_bias = 0.5
        elif avg_velocity > 200:  # Fast movement (saccades/scanning)
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
            
            if distance <= self.fixation_threshold_pixels:
                # Still in fixation - update position (use centroid)
                self.current_fixation_position = (
                    (self.current_fixation_position[0] + current_point.x) / 2,
                    (self.current_fixation_position[1] + current_point.y) / 2
                )
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
                        # Mark recent points as fixation
                        for point in self.gaze_history:
                            if point.timestamp >= self.current_fixation_start:
                                point.is_fixation = True
                
                # Start new potential fixation
                self.current_fixation_start = current_point.timestamp
                self.current_fixation_position = (current_point.x, current_point.y)
    
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
        elif current_velocity < 300:
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
            
            print(f"✅ Eye movement data exported to: {full_path}")
            print(f"📊 Data points: {len(csv_data)}")
            print(f"⏱️  Session duration: {session_duration:.2f} seconds")
            print(f"👁️  Total fixations: {len(self.fixations)}")
            print(f"📖 Reading speed: {self.metrics.reading_speed_wpm:.1f} WPM")
            
            return full_path
        else:
            print("❌ No data to export")
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
            print("❌ No active reading session to finish")
            return None
        
        # Final metrics update
        self._update_metrics()
        
        # Print session summary
        summary = self.get_analysis_summary()
        print("\n" + "="*50)
        print("📈 EYE MOVEMENT ANALYSIS COMPLETE")
        print("="*50)
        print(f"⏱️  Session Duration: {summary['session_duration']:.2f} seconds")
        print(f"👁️  Total Fixations: {summary['total_fixations']}")
        print(f"⚡ Average Fixation Duration: {summary['average_fixation_duration']:.3f} seconds")
        print(f"📖 Reading Speed: {summary['reading_speed_wpm']:.1f} WPM")
        print(f"🏃 Average Velocity: {summary['average_velocity']:.1f} px/sec")
        print(f"📄 Lines Read (estimated): {summary['lines_read_estimate']:.1f}")
        print(f"🎯 Fixation Rate: {summary['fixation_rate']:.2f} fixations/sec")
        print("="*50)
        
        # Auto-export to CSV
        csv_file = None
        if auto_export:
            csv_file = self.export_data_to_csv()
        
        return csv_file

# Example usage and testing
if __name__ == "__main__":
    # Test the eye movement analyzer
    analyzer = EyeMovementAnalyzer()
    analyzer.start_reading_session()
    
    print("🔬 Testing Eye Movement Analyzer (Silent Mode)...")
    print("📊 Collecting data in background...")
    
    # Simulate realistic reading pattern
    import random
    
    # Simulate reading multiple lines of text
    for line in range(5):  # 5 lines of text
        for word_pos in range(12):  # 12 words per line
            # Simulate fixation on each word (3-5 points per word)
            fixation_points = random.randint(3, 5)
            base_x = word_pos * 80 + 100  # Word position
            base_y = line * 60 + 150      # Line position
            
            for point in range(fixation_points):
                # Add small random movement within word area
                x = base_x + random.randint(-10, 10)
                y = base_y + random.randint(-5, 5)
                analyzer.add_gaze_point(x, y)
                time.sleep(0.05)  # Simulate realistic fixation timing
            
            # Simulate saccade to next word (faster movement)
            if word_pos < 11:  # Not last word in line
                next_x = (word_pos + 1) * 80 + 100
                # Quick movement to next word
                analyzer.add_gaze_point(next_x, base_y)
                time.sleep(0.02)
        
        # Simulate return sweep to next line
        if line < 4:  # Not last line
            analyzer.add_gaze_point(100, (line + 1) * 60 + 150)
            time.sleep(0.1)
    
    # Finish analysis and export data
    print("\n🏁 Finishing analysis session...")
    csv_file = analyzer.finish_analysis_session(auto_export=True)
    
    if csv_file:
        print(f"\n✨ Ready for model training! Data saved to: {csv_file}")
    else:
        print("\n❌ Failed to export data")
