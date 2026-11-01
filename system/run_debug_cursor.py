import cv2
import numpy as np
import time
import os
import sys
from prediction import TextReadingGazePredictor
from config import *

class DebugGazePredictor(TextReadingGazePredictor):
    def __init__(self):
        # Initialize parent without model
        # This will create the session directories and load text pages
        super().__init__(model_path=None)
        
        # Mouse state
        self.mouse_x = 0
        self.mouse_y = 0
        
        # Mock model components to avoid errors if accessed
        self.model = "DEBUG_MODE" 
        
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            self.mouse_x = x
            self.mouse_y = y

    def run_debug_session(self):
        print("="*60)
        print("Starting Debug Session with Mouse Cursor")
        print("="*60)
        print("INSTRUCTIONS:")
        print("1. Move the mouse cursor over the text to simulate reading")
        print("2. The green dot represents your 'gaze'")
        print("3. Stop on words to trigger fixations")
        print("4. Press 'A'/'D' to change pages")
        print("5. Press 'ESC' to exit and save data")
        print("="*60)
        
        self.movement_analyzer.start_reading_session()
        
        # Window setup
        window_name = 'Text Reading Debug (Cursor Mode)'
        cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.setMouseCallback(window_name, self.mouse_callback)
        
        self.session_start_time = time.time()
        
        try:
            while True:
                # 1. Render Text Page
                text_display = self._render_text_page()
                
                # 2. Get "Gaze" (Mouse) Position
                x, y = self.mouse_x, self.mouse_y
                
                # 3. Add to Analysis
                # Mock pupil size (fixed) and blink (randomly simulated)
                pupil_size = 3.5 + np.sin(time.time()) * 0.1
                blink_freq = 12.0
                
                # Call the actual data collection method
                self._add_gaze_point_to_analysis(x, y, pupil_size, blink_freq)
                
                # 4. Visual Feedback
                # Draw cursor/gaze point
                cv2.circle(text_display, (x, y), 10, (0, 255, 0), -1) 
                
                # Show debug info on screen
                nearest_word = self._find_nearest_word(x, y)
                
                # Get current velocity from analyzer
                try:
                    metrics = self.movement_analyzer.get_analysis_summary()
                    velocity = metrics.get('current_velocity', 0)
                except:
                    velocity = 0
                
                info_text = f"Word: {nearest_word} | Vel: {velocity:.1f}"
                cv2.putText(text_display, info_text, (x + 15, y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                cv2.putText(text_display, "DEBUG MODE - CURSOR CONTROL", (20, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                cv2.imshow(window_name, text_display)
                
                # 5. Controls
                key = cv2.waitKey(1) & 0xFF
                if key == 27: # ESC
                    break
                elif key == ord('a') or key == ord('A'):
                    if self.current_page > 0: 
                        self.current_page -= 1
                        print(f"Moved to page {self.current_page + 1}")
                elif key == ord('d') or key == ord('D'):
                    if self.current_page < self.total_pages - 1: 
                        self.current_page += 1
                        print(f"Moved to page {self.current_page + 1}")
                elif key == ord('e') or key == ord('E'):
                    self._export_fixation_data()
                    
        except Exception as e:
            print(f"Error in debug session: {e}")
            import traceback
            traceback.print_exc()
            
        finally:
            cv2.destroyAllWindows()
            print("\nSession ended. Exporting data...")
            csv_path = self._export_fixation_data()
            if csv_path:
                print(f"\nSUCCESS! Data exported to:\n{csv_path}")
            else:
                print("\nWARNING: No data was exported.")

if __name__ == "__main__":
    try:
        debugger = DebugGazePredictor()
        debugger.run_debug_session()
    except Exception as e:
        print(f"Failed to start debugger: {e}")
        input("Press Enter to exit...")
