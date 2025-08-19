# main.py
from tracker import EyeTracker

if __name__ == "__main__":
    print("Starting Eye Tracking System...")
    tracker = EyeTracker()
    tracker.run_full_pipeline()