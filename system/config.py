"""
Configuration file for the Gaze Tracking Reading Analysis System
Contains all system constants and settings
"""

import os 

# ============================================================================
# SYSTEM CONFIGURATION
# ============================================================================

# Screen and Display Settings
SCREEN_WIDTH = 1080
SCREEN_HEIGHT = 720
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_INDEX = 0

# File and Directory Settings
CALIBRATION_DATA_DIR = "calibration_data"
MODELS_DIR = "models"
OUTPUT_DIR = "eye_tracking_data"
USER_DATA_DIR = "user_data"

# Ensure directories exist
for directory in [CALIBRATION_DATA_DIR, MODELS_DIR, OUTPUT_DIR, USER_DATA_DIR]:
    os.makedirs(directory, exist_ok=True)

# ============================================================================
# VIDEO RECORDING SETTINGS
# ============================================================================

# Video Recording Configuration
VIDEO_RECORDING_ENABLED = True
VIDEO_FPS = 30
VIDEO_CODEC = 'mp4v'  # H.264 compatible codec
RECORD_CAMERA_FEED = True
RECORD_OVERLAY_DISPLAY = True
VIDEO_QUALITY = 80  # 0-100 quality setting
VIDEO_BUFFER_SIZE = 30  # Maximum frames to buffer

# ============================================================================
# CALIBRATION SETTINGS
# ============================================================================

# Face Detection and Tracking
FACE_BOUNDARY_MARGIN_X = 0.3  # 30% margin from left/right edges
FACE_BOUNDARY_MARGIN_Y = 0.15  # 15% margin from top/bottom edges
FACE_STABILITY_THRESHOLD = 1.0  # seconds to wait for stable face detection
FACE_BOUNDARY_SENSITIVITY = 0.1  # sensitivity for boundary warnings
MIN_FACE_SIZE = 100  # minimum face bounding box size in pixels

# MediaPipe Face Detection Confidence Settings
FACE_DETECTION_CONFIDENCE = 0.95    # Detection confidence
FACE_TRACKING_CONFIDENCE = 0.95     # Tracking confidence

# Calibration Process
BASELINE_FRAMES = 30  # frames for pitch/yaw baseline calculation
CALIBRATION_FRAMES = 11  # frames per calibration point
CALIBRATION_GRID_SIZE = 4  # 4x4 grid = 16 points
CALIBRATION_POINT_SIZE = 20  # radius of calibration points
CALIBRATION_MARGIN_X = 150  # pixels from screen edges
CALIBRATION_MARGIN_Y = 200  # pixels from screen edges

# Animation Settings
ANIMATION_SHRINK_FRAMES = 10  # frames for point shrinking animation
FADE_IN_FRAMES = 10  # frames for point fade in

# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

# Feature Column Names (FIXED ORDER - DO NOT CHANGE)
FEATURE_COLUMNS = [
    'norm_L_x', 'norm_L_y',  # Left eye normalized iris position
    'norm_R_x', 'norm_R_y',  # Right eye normalized iris position  
    'pitch', 'yaw',  # Head pose angles
    'tvect_x', 'tvect_y', 'tvect_z'  # Translation vectors
]

# PnP Algorithm Settings
PNP_3D_MODEL_POINTS = [
    (0.0, 0.0, 0.0),         # Nose tip
    (0.0, -330.0, -65.0),    # Chin
    (-225.0, 170.0, -135.0), # Left eye left corner
    (225.0, 170.0, -135.0),  # Right eye right corner
    (-150.0, -150.0, -125.0), # Left mouth corner
    (150.0, -150.0, -125.0)   # Right mouth corner
]

# Corresponding MediaPipe face mesh indices
PNP_LANDMARK_INDICES = [1, 152, 33, 263, 61, 291]

# Iris detection landmarks
LEFT_IRIS_LANDMARKS = [474, 475, 476, 477]
RIGHT_IRIS_LANDMARKS = [469, 470, 471, 472]

# Face boundary landmarks
FACE_BOUNDARY_LANDMARKS = {
    'forehead': [10, 151, 9, 8],
    'chin': [152, 175],
    'left_ear': [234, 93],
    'right_ear': [454, 323],
    'left_eye_outer': [33],
    'right_eye_outer': [263],
    'nose_tip': [1],
    'mouth_left': [61],
    'mouth_right': [291]
}

# ============================================================================
# MODEL TRAINING SETTINGS
# ============================================================================

# Data Processing
TEST_SIZE = 0.2  # 20% for testing
RANDOM_STATE = 42
NOISE_LEVEL = 5  # pixels of AGWN for training data

# Model Hyperparameters for GridSearchCV
RANDOM_FOREST_PARAMS = {
    'n_estimators': [300, 350, 400],
    'max_depth': [2, 4, 8, None],
    'min_samples_split': [2, 3, 5],
    'min_samples_leaf': [2, 3, 4],
    'max_features': ['sqrt', 'log2']
}

XGB_PARAMS = {
    'n_estimators': [300, 350, 400],
    'max_depth': [2, 4, 8],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0]
}

# Cross-validation settings
CV_FOLDS = 3

SCORING_METRIC = 'neg_mean_squared_error'

# ============================================================================
# UI COLORS AND FONTS
# ============================================================================

# Colors (BGR format for OpenCV)
RED = (0, 0, 255)
GREEN = (100, 255, 100)
BLUE = (255, 0, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
YELLOW = (0, 255, 255)
CYAN = (255, 255, 0)

# Text settings
FONT_SCALE = 1.0
FONT_THICKNESS = 2
TEXT_COLOR = WHITE
ERROR_COLOR = RED
SUCCESS_COLOR = GREEN

# ============================================================================
# SYSTEM MESSAGES
# ============================================================================

MESSAGES = {
    'face_not_detected': "FACE NOT DETECTED - Please position your face in view",
    'face_out_of_bounds': "FACE OUT OF BOUNDS - Please move closer to the camera",
    'face_too_small': "FACE TOO SMALL - Please move closer to the camera",
    'calibration_start': "CALIBRATION - Click on red points and keep gaze steady",
    'baseline_instruction': "Look at the center point and click when ready",
    'calibration_complete': "Calibration completed successfully!",
    'model_training_start': "Training gaze prediction model...",
    'model_training_complete': "Model training completed!",
    'press_space': "Press SPACE to continue",
    'press_esc': "Press ESC to exit"
}

# ============================================================================
# OUTLIER DETECTION
# ============================================================================

OUTLIER_METHOD = 'iqr'  # 'iqr' or 'zscore'
IQR_MULTIPLIER = 1.5
ZSCORE_THRESHOLD = 3.0

# ============================================================================
# EYE ANALYSIS AND READING SETTINGS
# ============================================================================
# Text Display Control
TEXT_ROWS_PER_PAGE = 3             # Number of text rows per page
TEXT_WORDS_PER_ROW = 10            # Maximum words per row (used in automatic calculation)
TEXT_FORCE_WORDS_PER_ROW = 10        # Set to a number (e.g., 5) to force exact words per row, None for automatic
TEXT_FONT = "arial"                 # Font type (arial, times, calibri, etc.)
TEXT_FONT_SIZE = 40               # Font size in points
TEXT_FONT_BOLD = True               # Bold text (True/False)
TEXT_COLOR = (0, 0, 0)              # Text color (R, G, B) - Black
TEXT_BACKGROUND = (255, 255, 255)   # Background color (R, G, B) - White
TEXT_LINE_SPACING = 80              # Spacing between lines in pixels
TEXT_MARGIN_X = 0.10                # Horizontal margin (10% from edges)
TEXT_MARGIN_Y = 0.35                # Vertical margin (35% from edges)

# Reading Text Content (100+ words - easily customizable)
# Reading Content - Pages and Probes
PAGES = {
    1: """זהו הטקסט של העמוד הראשון כדי לבדוק את רנדרינג של המערכת ולאחר מכן לבדוק את קריאת הנתונים מהטקסט""",
    
    2: """בטקסט של העמוד השני נכניס כמה מילים באנגלית כמו THE, THIS, THAT, THESE, THOSE כדי שנוכל לבדוק איך הוא מתמודד עם טקסט באנגלית""",
    
    3: """ בעמוד השלישי נכניס כמה מילים בעברית כמו את, על, עם, בין, אם, מה, זה, הוא ואולי גם כמה סימנים?  """,
    
    4: """זהו הטקסט של העמוד האחרון, תודה שקראתם!"""
}

PROBES = {
    1: "האם הטקסט ברור?",
    2: "האם הבנת את הטקסט?",
    3: "האם הטקסט ברור?"
}

# Eye Movement Analysis Settings
FIXATION_THRESHOLD = 10              # Minimum duration (ms) to count as fixation
SACCADE_VELOCITY_THRESHOLD = 200    # Minimum velocity (px/s) for saccade detection
WORD_PROXIMITY_THRESHOLD = 100       # Maximum distance (px) to associate gaze with word
ANALYSIS_EXPORT_FORMAT = "detailed"  # "detailed" or "simple"

# CSV Export Settings
CSV_COLUMNS = [
    "Fixation_Order",
    "Fixated_Word", 
    "Fixation_X_Screen",
    "Fixation_Y_Screen",
    "Fixation_Duration",
    "Time_from_Stimulus_Onset",
    "Pupil_Size",
    "Blink_Frequency"
]

# ============================================================================
# HEBREW TEXT SUPPORT
# ============================================================================

# Hebrew/RTL text settings
HEBREW_SUPPORT = True
HEBREW_FONT_PATH = None  # Will auto-detect system Hebrew font
HEBREW_FONT_SIZE = 32
TEXT_DIRECTION = "auto"  # "auto", "ltr", "rtl"


# Auto language detection (simple keyword-based)
HEBREW_KEYWORDS = ["של", "את", "על", "אל", "עם", "בין", "אם", "מה", "זה", "הוא"]

# RTL (Right-to-Left) Text Configuration
RTL_READING_DIRECTION = "rtl"  # Direction for RTL text processing
RTL_AUTO_DETECT = True         # Auto-detect RTL text and adjust eye tracking
RTL_DETECTION_THRESHOLD = 1    # Minimum Hebrew words to consider text as RTL

