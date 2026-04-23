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
FACE_DETECTION_CONFIDENCE = 0.9    # Detection confidence
FACE_TRACKING_CONFIDENCE = 0.9     # Tracking confidence

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
TEXT_FONT_SIZE = 25               # Font size in points
TEXT_FONT_BOLD = True               # Bold text (True/False)
TEXT_COLOR = (0, 0, 0)              # Text color (R, G, B) - Black
TEXT_BACKGROUND = (255, 255, 255)   # Background color (R, G, B) - White
TEXT_LINE_SPACING = 80              # Spacing between lines in pixels
TEXT_MARGIN_X = 0.2                # Horizontal margin (10% from edges)
TEXT_MARGIN_Y = 0.35                # Vertical margin (35% from edges)

# Reading Text Content (100+ words - easily customizable)
# Reading Content - Pages and Probes
PAGES = {
    1: """גידול כלב מביא שמחה והרגשה טובה לבני המשפחה בכל יום.  טיולים חשובים לבריאות הכלב, למצב הרוח שלו ולמפגש עם כלבים אחרים. עדיף לצאת לכמה טיולים קצרים ביום ולא לטיול אחד ארוך,  כך הכלב רגוע יותר ומרגיש בטוח בסביבה הקרובה והמוכרת לו. משחק כל יום עוזר לכלב להוציא אנרגיה, ליהנות מהזמן שלו, ולהרגיש טוב ורגוע יותר במהלך היום, לכן כדאי לעשות את לעיתים קרובות. המשחק עוזר ליצור קשר טוב עם בני הבית, לכן חשוב להכין לכלב מקום שקט ונוח לישון בו ולתת לו צעצועים פשוטים ובטוחים. """,
    
    2: """ רצוי לחשוף את הכלב לאנשים ולרעשים לאט ובהדרגה ולתת לכלב אוכל ומים בכל יום. את האוכל נותנים בכמה ארוחות קטנות.  ביקור אצל וטרינר עוזר לשמור על בריאות הכלב ואיכות חייו. מומלץ לטפל בכלב באהבה ובסבלנות, ולשמור על שגרה קבועה.""",
    
    3: """עד כמה נדדת מ 1-5?""",

    4: """ביטחון עצמי נתפס בספרות כתהליך מתמשך, המתעצב מתוך חוויות ופרשנויות אישיות בהקשרים משתנים. תהליך זה מוגדר כמרכיב דינמי בזהות, ולא כתכונה קבועה ובלתי משתנה לאורך זמן. המחקר מבחין בין ביטחון כללי לבין ביטחון תחומי בהקשרים כמו עבודה, למידה או הורות, בעוד שביטחון כללי מכוון התנהלות רחבה.  """,

    5: """ביטחון תחומי קשור לתפקוד ולהערכת ביצועים ספציפיים. בלי תחושת ביטחון, אנשים נוטים לדפוסי הימנעות, לעומת זאת, ביטחון מבוסס עשוי לעודד מחויבות, התמדה והרחבת תחומי פעולה.  שינויים בביטחון מתוארים כתהליכים, הכוללים ניטור עצמי ובחינת דפוסי חשיבה המשוב הפנימי  מופעל,  ומכוון לבחירות ואסטרטגיות התנהלות. הספרות מראה שספק מתון תורם לחשיבה, ללמידה מתמשכת ולהעמקת ההבנה במצבים חברתיים שונים.""",

    6: """עד כמה נדדת במחשבותיך מ1-5?""",

    7: """כל אדם אשר יופיע במסמכי פתיחת החשבון כמיופה סמכות רחבה לפעול בשמי ובהתאם לשיקול דעתו המקצועי ייחשב מורשה מלא לכל דבר ועניין, בכפוף לאישור מוקדם ומפורש של הבנק ובהתאם לדרישותיו הפנימיות.  הרשאה זו תחול על כלל ערוצי השירות הקיימים, לרבות ערוצים דיגיטליים, ותהיה בתוקף גם ביחס לערוצים עתידיים שיתווספו מבלי לפגוע בזכות הבנק לקבוע תנאים נוספים לפי שיקול דעתו.  כל פעולה שתבוצע בחשבון תיחשב כפעולה שבוצעה בשמי ובאחריותי המלאה גם במקרה של חריגה מסמכות מוצהרת, אלא אם נמסרה הודעה כתובה מתועדת ומאושרת כדין הצטרפות לשירותים נוספים.""",

    8: """כל שינוי בתנאים יחייב אותי ללא הסתייגות. אני מתחייב להימנע מהתקשרויות סותרות ולשפות את הבנק בגין כל נזק שייגרם כתוצאה מהפרת הוראות אלו. בכל מקרה תחול עליי אחריות משפטית מלאה ומצטברת לכל פעולה עתידית שתתבצע במסגרת ניהול החשבון האמור.""",

    9: """עד כמה נדדת במחשבותיך מ1-5?""",

    10: """חוכמה לאור הנר נתפסת במחקר האמנותי כתהליך מתמשך, המתעצב מתוך הקשרים חזותיים ופרשנויות תרבותיות. תהליך זה מוגדר כמרכיב דינמי במשמעות, ולא כייצוג קבוע ובלתי משתנה. המחקר מבחין בין ייצוג כללי לבין ייצוג תחומי בהקשרים מדעיים ואמנותיים בעוד ייצוג כללי מכוון תפיסה רעיונית רחבה, ייצוג תחומי קשור להערכת סמלים שני היבטי הייצוג ניתנים לפיתוח באמצעות ניסוי חוזר ורפלקציה בהיעדר ייצוג מגובש, פרשנויות נוטות לעמימות ולהגבלת הבנה לעומת זאת, ייצוג מבוסס עשוי לעודד העמקה והרחבת פרשנות.""",

    11: """עד כמה נדדת במחשבותיך מ 1-5?""",

    12: """במסורת האמנותית, נושא הבשורה מתפתח כתהליך מתמשך הנשען על הקשרים תרבותיים ועל פרשנויות תאולוגיות שונות. המשמעות נבנית כאן כתהליך מתפתח וגמיש, ולא כייצוג סגור ובלתי משתנה. ניכרת הבחנה בין משמעות כללית לבין משמעות תחומית בהקשרים דוקטרינריים. המשמעות הכללית מכוונת להבנה תאולוגית רחבה, ואילו המשמעות התחומית עוסקת בהערכת סמלים. היבטים אלה של המשמעות ניתנים לפיתוח באמצעות פרשנות חוזרת ורפלקציה. מצב זה מוביל לכך שפרשנויות נוטות לעמימות ולהגבלת הבנה. כך מתאפשרים תהליכי העמקה, הרחבת פרשנות והבנה מורכבת יותר של המשמעויות. התהליך מתואר כתהליך הכולל ניטור חזותי והערכה שיטתית. שבהם מופעל משוב רעיוני המכוון פרשנות. ועשויה לתרום לאיזון בין בהירות, גמישות פרשנית והעמקה רעיונית, ולהבנה רעיונית לאורך הקשרים תרבותיים משתנים נוספים.""",

    13: """עד כמה נדדת במחשבותיך מ 1-5?""",

    14: """הצדדים מסכימים כי מסמך זה מהווה הסכם מחייב כולל ואחיד אשר הוראותיו יוצרות מארג נורמטיבי שלם המחייב את הצדדים בכל היבטיו. פרשנות ההסכם תיעשה כמכלול אינטגרטיבי וכותרות הסעיפים נועדו לנוחות בלבד ואינן גורעות מהוראה מהותית כלשהי.  הספק מצהיר כי השירותים ניתנים ללא הקניית זכות קניינית, ללא יצירת מעמד מתמשך ובהתאם לדין כל תשלום שישולם ייחשב כתמורה חוזית בלבד ולא יקנה זכות עתידית או תרופה אוטומטית.""",

    15: """הלקוח מוותר מראש על כל טענה ומצהיר כי השקעותיו נעשות באחריותו הבלעדית. הוראות ההסכם גוברות על כל מסמך אחר וכל מחלוקת תפורש בהתאם ללשון ההסכם ותתברר לפי הדין החל במסגרת ההתקשרות החוזית המחייבת וללא אפשרות לסטייה פרשנית נוספת מצד מי מהצדדים בכל שלב מימוש ההסכם או יישומו המעשי בפועל לאורך תקופת ההתקשרות המוסכמת והמחייבת בין הצדדים כולם במלואה ובתוקף מלא.""",

    16: """עד כמה נדדת במחשבותיך מ 1-5?""",

    17: """משמעות החיים מתוארת כתהליך מתמשך שנבנה מתוך חוויות אישיות ודרכי פרשנות שונות במצבים משתנים. תהליך זה מתואר כמבנה דינמי של ערכים, ולא כעמדה קבועה שאינה משתנה עם הזמן.  המחקר מבחין בין משמעות כללית לבין משמעות תחומית בהקשרים כמו עבודה, קהילה או משפחה,  בעוד שמשמעות כללית מכוונת תפיסת חיים רחבה. שני היבטי המשמעות יכולים להתפתח לאורך החיים דרך התנסויות חוזרות, חשיבה על חוויות ושיקוף עצמי. בלי תחושת משמעות, אנשים עלולים לחוות בלבול, להפחית יוזמה ולהימנע ממעורבות חברתית, לעומת זאת, תחושת משמעות יציבה עשויה לחזק מחויבות, לעודד התמדה ולהרחיב תחומי עשייה. שינויים במשמעות מתוארים כתהליכים של התבוננות עצמית ובחינה של דפוסי חשיבה.  תהליך זה עוזר להכווין בחירות ודרכי פעולה. מחקרים מצביעים על כך שספק קיומי מתון יכול לתרום להתפתחות אישית ולהעמקת ההבנה העצמית לאורך זמן.""",

    18: """עד כמה נדדת במחשבותיך מ 1-5?""", 

    19: """שינה טובה חשובה מאוד לבריאות ולהרגשה הכללית של הגוף והנפש. כשישנים מספיק שעות בלילה, הגוף והמוח נחים, ויש יותר כוח ליום החדש. מבוגרים צריכים לישון מספר קבוע של שעות בכל לילה, וילדים צריכים יותר שעות שינה כדי לגדול, ללמוד ולהתפתח.""",

    20: """שינה רצופה ואיכותית עוזרת לקום בבוקר ערניים, רגועים ומוכנים לפעילויות היום, לכן כדאי ללכת לישון בשעה מוקדמת ולא לדחות את שעת השינה. לפני השינה כדאי לא לשתות קפה, לא לאכול אוכל כבד או להשתמש במסכים,  מכיוון שדברים אלו מקשים על ההרדמות.  רצוי לעשות פעילות גופנית קלה במהלך היום, ולשמור על שגרה קבועה, שתעזור לגוף להירגע ולישון טוב. הקפדה על הרגלי שינה טובים, יוצרת הרגשה טובה. עם הרגלי שינה קבועים, מרגישים פחות עייפים, ומצב הרוח משתפר. מומלץ לשמור על הרגלים קבועים כדי לשפר ריכוז וסבלנות במשימות יום יומיות.""",

    21: """עד כמה נדדת במחשבותיך מ 1-5?""",

    22: """תודה על השתתפותך!"""
}

PROBES = {
    1: "האם נדדת במחשבותיך?",
    2: "האם חשוב לטייל ולשחק עם הכלב כל יום?",
    3: "מוכן להמשיך?",
    4: "האם נדדת במחשבותיך?",
    5: "האם נדדת במחשבותיך?",
    6: "האם הביטחון העצמי יכול להשתנות עם הזמן?",
    7: "האם נדדת במחשבותיך?",
    8: "האם ניתן להגביל אחריות לערוצים עתידיים?",
    9: "מוכן להמשיך?",
    10: "האם נדדת במחשבותיך?",
    11: "האם עמימות מוחלטת מעמיקה הבנה רעיונית?",
    12: "האם נדדת במחשבותיך?",
    13: "האם בהירות מלאה הכרחית להעמקה פרשנית?",
    14: "האם נדדת במחשבותיך?",
    15: "האם נדדת במחשבותיך?",
    16: "האם ניתן לשנות את יישום ההסכם בפרשנות חיצונית?",
    17: "האם נדדת במחשבותיך?",
    18: "האם ספק קיומי פוגע בהתפתחות האישית?",
    19: "האם נדדת במחשבותיך?",
    20: "האם נדדת במחשבותיך?",
    21: "האם הריכוז ומצב הרוח תלויים רק בשינה?",
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

