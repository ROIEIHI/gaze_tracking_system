import numpy as np
import cv2

def test_3d_model_validity():
    """Test if our 3D model points make geometric sense"""
    
    # Our current 3D model points
    current_3d_points = np.array([
        (0, 0, 0),              # Nose tip (index 1)
        (0, -330, -65),         # Chin (index 152)
        (-225, 170, -135),      # Left eye left corner (index 33)
        (225, 170, -135),       # Right eye right corner (index 362)
        (-150, -150, -125),     # Left mouth corner (index 61)
        (150, -150, -125)       # Right mouth corner (index 291)
    ], dtype=np.float64)
    
    # Alternative 3D model from MediaPipe documentation
    mediapipe_standard = np.array([
        (0.0, 0.0, 0.0),          # Nose tip
        (0.0, -330.0, -65.0),     # Chin
        (-225.0, 170.0, -135.0),  # Left eye left corner  
        (225.0, 170.0, -135.0),   # Right eye right corner
        (-150.0, -150.0, -125.0), # Left mouth corner
        (150.0, -150.0, -125.0)   # Right mouth corner
    ], dtype=np.float64)
    
    # OpenCV tutorial standard model
    opencv_standard = np.array([
        (0.0, 0.0, 0.0),          # Nose tip
        (0.0, -330.0, -65.0),     # Chin
        (-165.0, 170.0, -135.0),  # Left eye left corner
        (165.0, 170.0, -135.0),   # Right eye right corner  
        (-150.0, -150.0, -125.0), # Left mouth corner
        (150.0, -150.0, -125.0)   # Right mouth corner
    ], dtype=np.float64)
    
    print("=== 3D Model Analysis ===")
    print()
    
    models = {
        "Current": current_3d_points,
        "MediaPipe Standard": mediapipe_standard, 
        "OpenCV Standard": opencv_standard
    }
    
    for name, points in models.items():
        print(f"{name} Model:")
        print(f"  Nose tip: {points[0]}")
        print(f"  Chin: {points[1]}")
        print(f"  Left eye: {points[2]}")
        print(f"  Right eye: {points[3]}")
        print(f"  Left mouth: {points[4]}")
        print(f"  Right mouth: {points[5]}")
        
        # Calculate distances
        nose_to_chin = np.linalg.norm(points[1] - points[0])
        eye_distance = np.linalg.norm(points[3] - points[2])
        mouth_distance = np.linalg.norm(points[5] - points[4])
        
        print(f"  Distances:")
        print(f"    Nose to chin: {nose_to_chin:.1f}mm")
        print(f"    Eye separation: {eye_distance:.1f}mm")
        print(f"    Mouth width: {mouth_distance:.1f}mm")
        print()

def create_better_3d_model():
    """Create a more accurate 3D head model based on anthropometric data"""
    
    # Based on average adult head measurements (mm)
    # Nose tip at origin (0,0,0)
    better_model = np.array([
        (0.0, 0.0, 0.0),          # Nose tip - reference point
        (0.0, -70.0, -65.0),      # Chin - 70mm below nose, 65mm back
        (-35.0, 15.0, -15.0),     # Left eye inner corner - 35mm left, 15mm up, 15mm back
        (35.0, 15.0, -15.0),      # Right eye inner corner - 35mm right, 15mm up, 15mm back  
        (-25.0, -25.0, -10.0),    # Left mouth corner - 25mm left, 25mm down, 10mm back
        (25.0, -25.0, -10.0)      # Right mouth corner - 25mm right, 25mm down, 10mm back
    ], dtype=np.float64)
    
    print("=== Improved 3D Model ===")
    print("Based on anthropometric measurements:")
    for i, point in enumerate(better_model):
        labels = ["Nose tip", "Chin", "Left eye", "Right eye", "Left mouth", "Right mouth"]
        print(f"  {labels[i]}: {point}")
    
    return better_model

if __name__ == "__main__":
    test_3d_model_validity()
    better_model = create_better_3d_model()
    
    # Save the better model for testing
    np.save("better_3d_model.npy", better_model)
    print("\nSaved improved 3D model to 'better_3d_model.npy'")
