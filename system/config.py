"""
Configuration file for the Gaze Tracking Reading Analysis System

The sections most often changed come first. Search for a number to jump there, for example "# 3."

     1. SESSION OPTIONS        Gaze dot, accuracy check, video on or off
     2. LAB SETUP              Screen size, viewing distance, camera
     3. TEXT DISPLAY           Font, rows, line spacing, margins
     4. GAZE ANALYSIS          Fixation, saccade and word matching limits
     5. PARTICIPANT POSITION   Face box, face detection, position check during reading
     6. CALIBRATION            Calibration dots, timing, training preview
     7. VIDEO RECORDING        What is recorded and how
     8. MODEL TRAINING         Data split, outliers, hyperparameters
     9. COLORS AND MESSAGES    On-screen colors, font and instructions
    10. FILE LOCATIONS         Where data is saved
    11. FACE AND EYE MODEL     Landmarks and features, do not change
    12. NOT IN USE             Settings no code reads, changing them has no effect
    13. EXPERIMENT SCRIPT      Opening screen, texts and questions, in the order shown
"""

import os

# ============================================================================
# 1. SESSION OPTIONS
# ============================================================================

SHOW_GAZE_POINT = False             # Show the live gaze point to the participant during reading (gaze is recorded either way)
VIDEO_RECORDING_ENABLED = True      # Record the session videos, see section 7

# Accuracy check after training
ERROR_THRESHOLD_ENABLED = False     # Offer recalibration when the error is over the threshold (False ignores it, the error is still saved)
ERROR_THRESHOLD_DEGREES = 2.5       # Maximum gaze error in degrees of visual angle

# ============================================================================
# 2. LAB SETUP
# ============================================================================

# Physical setup, used to convert the gaze error to degrees
SCREEN_DIAGONAL_CM = 35.56          # Physical screen diagonal (14 inch)
VIEWING_DISTANCE_CM = 40            # Eye-to-screen distance

# Size of the image shown to the participant, stretched to fill the screen
SCREEN_WIDTH = 1080
SCREEN_HEIGHT = 720

# Camera
CAMERA_INDEX = 0
CAMERA_WIDTH = 640                  # Size requested from the camera, it may deliver a different one
CAMERA_HEIGHT = 480

# ============================================================================
# 3. TEXT DISPLAY
# ============================================================================

# Layout. The program stops before calibration with a clear error if the text can't fit
TEXT_ROWS_PER_PAGE = 4              # Number of text rows per page
TEXT_FONT_SIZE = 40                 # Font size in points
TEXT_LINE_SPACING = 120             # Spacing between lines in pixels
TEXT_MARGIN_X = 0.1                 # Horizontal margin (10% from each edge)
TEXT_MARGIN_Y = 0.22                # Top margin (22% of the screen height), the first row starts here on every page.
                                    # 0.22 centres 4 rows at font 40 and spacing 120, adjust it if those change

# Appearance
TEXT_COLOR = (0, 0, 0)              # Text color (R, G, B) - Black
TEXT_BACKGROUND = (255, 255, 255)   # Background color (R, G, B) - White
INSTRUCTION_FONT_SIZE = 28          # Font size of the instructions on the opening screen
TEXT_FONT_PATHS = [                 # Hebrew fonts, the first one that loads is used
    "C:/Windows/Fonts/david.ttf",
    "C:/Windows/Fonts/tahoma.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/calibri.ttf"
]

RTL_READING_DIRECTION = "rtl"       # Direction for RTL text processing

# ============================================================================
# 4. GAZE ANALYSIS
# ============================================================================

FIXATION_THRESHOLD = 10             # Minimum duration (ms) to count as fixation
SACCADE_VELOCITY_THRESHOLD = 200    # Minimum velocity (px/s) for saccade detection
WORD_PROXIMITY_THRESHOLD = 100      # Maximum distance (px) to associate gaze with word

# ============================================================================
# 5. PARTICIPANT POSITION
# ============================================================================

# Face box shown during setup
FACE_BOUNDARY_MARGIN_X = 0.35       # 35% margin from left/right edges
FACE_BOUNDARY_MARGIN_Y = 0.2375     # 23.75% margin from top/bottom edges
FACE_BOUNDARY_SENSITIVITY = 0.075   # Sensitivity for boundary warnings

# Face detection
MIN_FACE_SIZE = 200                 # Minimum face bounding box size in pixels 200 ~ 40 cm distance from the camera
FACE_STABILITY_THRESHOLD = 1.0      # Seconds to wait for stable face detection
FACE_DETECTION_CONFIDENCE = 0.9     # MediaPipe detection confidence
FACE_TRACKING_CONFIDENCE = 0.9      # MediaPipe tracking confidence

# Position check during reading, using the face box above
OOB_CHECK_ENABLED = True            # Pause reading when the face leaves the box, and hold the opening screen until it's inside
OOB_PAUSE_AFTER_SECONDS = 2.0       # Seconds out of the box before reading pauses. Shorter drifts only leave a gap in the data
OOB_RESUME_AFTER_SECONDS = 1.0      # Seconds steady inside the box before reading resumes

# ============================================================================
# 6. CALIBRATION
# ============================================================================

# Calibration process
BASELINE_FRAMES = 30                # Frames for pitch/yaw baseline calculation
CALIBRATION_FRAMES = 11             # Frames per calibration point

# Calibration points
CALIBRATION_GRID_SIZE = 4           # 4x4 grid = 16 points, added to the 21 fixed points
CALIBRATION_POINT_SIZE = 20         # Radius of calibration points
CALIBRATION_MARGIN_X = 150          # Grid distance from the left/right edges, in pixels
CALIBRATION_MARGIN_Y = 200          # Grid distance from the top/bottom edges, in pixels
ANIMATION_SHRINK_FRAMES = 10        # Frames for point shrinking animation

# Training preview (self-view shown while the model trains)
TRAINING_PREVIEW_FPS = 12           # Preview rate cap
TRAINING_PREVIEW_CHECK_INTERVAL = 1.5   # Seconds between face position checks

# ============================================================================
# 7. VIDEO RECORDING (turned on or off in section 1)
# ============================================================================

RECORD_CAMERA_FEED = True           # Save the webcam video
RECORD_OVERLAY_DISPLAY = True       # Save the screen video with the gaze markers
VIDEO_FPS = 30                      # Playback rate of the saved videos
VIDEO_CODEC = 'mp4v'                # MPEG-4 codec
VIDEO_BUFFER_SIZE = 30              # Maximum frames waiting to be saved

# ============================================================================
# 8. MODEL TRAINING
# Changing these changes the trained model, so sessions may stop being comparable
# ============================================================================

# Data processing
TEST_SIZE = 0.2                     # 20% for testing
RANDOM_STATE = 42
NOISE_LEVEL = 5                     # Pixels of AGWN for training data

# Outlier detection
OUTLIER_METHOD = 'iqr'              # 'iqr' or 'zscore'
IQR_MULTIPLIER = 1.5                # Multiplier for the interquartile range when using the IQR method
ZSCORE_THRESHOLD = 3.0              # Threshold when using the z-score method

# Hyperparameters for GridSearchCV
XGB_PARAMS = {
    'n_estimators': [300, 350, 400],
    'max_depth': [2, 4, 8],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0]
}
CV_FOLDS = 3                        # Cross-validation folds

# ============================================================================
# 9. COLORS AND MESSAGES (setup and calibration screens)
# ============================================================================

# Colors (BGR format for OpenCV)
RED = (0, 0, 255)
GREEN = (100, 255, 100)
BLUE = (255, 0, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
YELLOW = (0, 255, 255)

# OpenCV font
FONT_SCALE = 1.0
FONT_THICKNESS = 1

# Instructions shown on screen
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
# 10. FILE LOCATIONS (always inside this folder, wherever the program is launched from)
# ============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
USER_DATA_DIR = os.path.join(BASE_DIR, "user_data")
CALIBRATION_DATA_DIR = os.path.join(BASE_DIR, "calibration_data")  # Only used by main.py
MODELS_DIR = os.path.join(BASE_DIR, "models")                      # Only used by main.py

# ============================================================================
# 11. FACE AND EYE MODEL
# ============================================================================

# Feature column names (FIXED ORDER - DO NOT CHANGE)
FEATURE_COLUMNS = [
    'norm_L_x', 'norm_L_y',  # Left eye normalized iris position
    'norm_R_x', 'norm_R_y',  # Right eye normalized iris position
    'pitch', 'yaw',  # Head pose angles
    'tvect_x', 'tvect_y', 'tvect_z'  # Translation vectors
]

# PnP algorithm settings
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
# 13. EXPERIMENT SCRIPT
# ============================================================================

# Opening screen, shown before the first text. The tester presses D to start reading
OPENING_TITLE = "הכל מוכן! אפשר להתחיל..."
OPENING_INSTRUCTIONS = [            # One paragraph per line, long lines wrap on screen
    "בחלק זה יוצגו לך כמה טקסטים קצרים לקריאה.",
    "יש לשמור על תנוחה יציבה בלי לזוז הרבה עם הראש בדומה לתהליך הקליברציה.",
    "אחרי כל חלק בטקסט תופיע שאלה, עונים עליה בלחיצה על YES או NO.",
    "בסוף כל טקסט יש לדרג עד כמה הבנת אותו מ 1-5, ולסמן במספר אצבעות מול המצלמה.",
    "אם תנוחת הראש תשתנה, הקריאה תיעצר ותצטרך לחזור למקום. הקריאה תמשיך מעצמה.",
    "בהצלחה!",
]

# Shown while reading is paused because the face left the box
REPOSITION_MESSAGE = "נא לחזור למרכז המסגרת. הקריאה תמשיך מעצמה"

# Experiment script, in the order it is shown. Each text lists its steps from top to bottom:
#   ("text", content)                 part of the text, the only step whose gaze goes to the CSV
#   ("probe", question)               yes/no question answered by clicking
#   ("probe", question, "Yes"/"No")   same, with the correct answer so the response can be scored
#   ("rating", question)              no buttons, the tester records the answer and presses D to continue
# "type" is saved with the data, participants never see it
UNDERSTOOD_PROBE = "האם הבנת את הטקסט?"
UNDERSTANDING_RATING = "עד כמה הבנת את הטקסט מ 1-5?"

EXPERIMENT = [
    {"number": 1, "type": "קל", "steps": [
        ("text", """גידול עציצים מביא שמחה והרגשה טובה לבני המשפחה בכל יום. אור שמש חשוב לבריאות העציץ, לצמיחה שלו ולהופעה של עלים חדשים. עדיף לתת לעציץ כמה השקיות קטנות בשבוע ולא השקיה אחת גדולה, כך העציץ חזק יותר וגדל טוב באדמה הלחה והמתאימה לו. טיפול כל יום עוזר לעציץ לגדול, ליהנות מהמקום שלו, ולהיראות ירוק ובריא יותר במהלך השנה, לכן כדאי לעשות אותו לעיתים קרובות. הטיפול עוזר ליצור אווירה טובה לבני הבית, לכן חשוב להכין לעציץ מקום מואר ונוח לגדול בו ולתת לו אדמה פשוטה ואיכותית."""),
        ("probe", UNDERSTOOD_PROBE),
        ("text", """רצוי להעביר את העציץ לחדרים ולמקומות חדשים לאט ובהדרגה ולבדוק את העלים ואת האדמה בכל יום. את הדשן נותנים בכמה מנות קטנות. ביקור במשתלה עוזר לשמור על בריאות העציץ ואורך חייו. מומלץ לטפל בעציץ באהבה ובסבלנות, ולשמור על שגרה קבועה."""),
        ("probe", UNDERSTOOD_PROBE),
        ("rating", UNDERSTANDING_RATING),
        ("probe", "האם חשוב להשקות ולטפל בעציץ באופן קבוע?", "Yes"),
    ]},
    {"number": 2, "type": "מעניין", "steps": [
        ("text", """מוטיבציה נתפסת בספרות כתהליך מתמשך. זהו תהליך המתעצב מתוך חוויות ופרשנויות אישיות בהקשרים משתנים. תהליך זה מוגדר כמרכיב דינמי באישיות, ולא כתכונה קבועה ובלתי משתנה לאורך זמן. המחקר מבחין בין מוטיבציה כללית לבין מוטיבציה תחומית בהקשרים כמו עבודה, לימודים או ספורט. בעוד שמוטיבציה כללית מכוונת התנהלות רחבה, מוטיבציה תחומית קשורה להשקעה ולהשגת מטרות ספציפיות."""),
        ("probe", UNDERSTOOD_PROBE),
        ("text", """בלי תחושת מוטיבציה, אנשים נוטים לדפוסי דחיינות, ולעומת זאת, מוטיבציה מבוססת עשויה לעודד מחויבות, התמדה והרחבת תחומי עניין. שינויים במוטיבציה מתוארים כתהליכים, הכוללים ניטור עצמי ובחינת דפוסי חשיבה. המשוב הפנימי מופעל ומכוון את הבחירות ואת דרכי ההתנהלות. הספרות מראה שאתגר מתון תורם לחשיבה וללמידה מתמשכת. כמו כן, אתגר מתון תורם להעמקת ההבנה במצבים חברתיים שונים."""),
        ("probe", UNDERSTOOD_PROBE),
        ("rating", UNDERSTANDING_RATING),
        ("probe", "האם מוטיבציה יכולה להשתנות עם הזמן?", "Yes"),
    ]},
    {"number": 3, "type": "קשה", "steps": [
        ("text", """כל אדם אשר יופיע במסמכי ההצטרפות לפוליסה כמיופה כוח רחב לפעול בשמי ובהתאם לשיקול דעתו האישי ייחשב מורשה מלא לכל דבר ועניין, בכפוף לאישור מוקדם ומפורש של המבטח ובהתאם לנהליו הפנימיים. הרשאה זו תחול על כלל ערוצי ההתקשרות הקיימים, לרבות ערוצים מקוונים, ותהיה בתוקף גם ביחס לערוצים עתידיים שיתווספו מבלי לפגוע בזכות המבטח לקבוע תנאים נוספים לפי שיקול דעתו. כל פעולה שתבוצע בפוליסה תיחשב כפעולה שבוצעה בשמי ובאחריותי המלאה גם במקרה של חריגה מהרשאה מוצהרת, אלא אם נמסרה למבטח מראש הודעה כתובה, מתועדת ומאושרת כדין על ביטול ההרשאה."""),
        ("probe", UNDERSTOOD_PROBE),
        ("text", """כל עדכון בתנאים יחייב אותי ללא הסתייגות. אני מתחייב להימנע מהתקשרויות מקבילות ולשפות את המבטח בגין כל נזק שייגרם כתוצאה מהפרת הוראות אלו. בכל מקרה תחול עליי אחריות משפטית מלאה ומצטברת לכל פעולה עתידית שתתבצע במסגרת ניהול הפוליסה האמורה."""),
        ("probe", UNDERSTOOD_PROBE),
        ("rating", UNDERSTANDING_RATING),
        ("probe", "האם ההרשאה חלה רק על ערוצי ההתקשרות הקיימים?", "No"),
    ]},
    {"number": 4, "type": "משעמם", "steps": [
        ("text", """דימוי הצל בציור נתפס במחקר האמנותי כתהליך מתמשך, המתעצב מתוך הקשרים חזותיים ופרשנויות סגנוניות. תהליך זה מוגדר כמרכיב דינמי בקומפוזיציה, ולא כייצוג קבוע ובלתי משתנה. המחקר מבחין בין דימוי כללי לבין דימוי תחומי בהקשרים היסטוריים ואמנותיים. בעוד שדימוי כללי מכוון תפיסה צורנית רחבה, דימוי תחומי קשור להערכת מוטיבים. שני היבטי הדימוי ניתנים לפיתוח באמצעות התבוננות חוזרת ורפלקציה. בהיעדר דימוי מגובש, פרשנויות נוטות לעמימות ולהגבלת הבנה. לעומת זאת, דימוי מבוסס עשוי לעודד העמקה והרחבת פרשנות."""),
        ("probe", UNDERSTOOD_PROBE),
        ("text", """שינויים בדימוי מתוארים כתהליכים הכוללים ניטור חזותי ובחינת דפוסים. במסגרת זו מופעל משוב סגנוני המכוון הבנה והערכה. עם זאת, הספרות מציעה כי עמימות מתונה עשויה לשמש גורם מווסת. היא גם תורמת להעמקת הבנה צורנית לאורך הקשרים משתנים."""),
        ("probe", UNDERSTOOD_PROBE),
        ("rating", UNDERSTANDING_RATING),
        ("probe", "האם עמימות מוחלטת מעמיקה הבנה צורנית?", "No"),
    ]},
    {"number": 5, "type": "משעמם", "steps": [
        ("text", """במסורת הציור, נושא הים מתפתח כתהליך מתמשך הנשען על הקשרים סגנוניים ועל פרשנויות אסתטיות שונות. המשמעות נבנית כאן כתהליך מתפתח וגמיש, ולא כייצוג סגור ובלתי משתנה. ניכרת הבחנה בין משמעות כללית לבין משמעות תחומית בהקשרים אקדמיים. המשמעות הכללית מכוונת להבנה אסתטית רחבה, ואילו המשמעות התחומית עוסקת בהערכת מוטיבים. היבטים אלה של המשמעות ניתנים לפיתוח באמצעות התבוננות חוזרת ורפלקציה. מצב זה מוביל לכך שפרשנויות נוטות לעמימות ולהגבלת הבנה. כך מתאפשרים תהליכי העמקה, הרחבת פרשנות והבנה מורכבת יותר של המשמעויות. התהליך מתואר כתהליך הכולל ניטור חזותי והערכה שיטתית. במהלכו מופעל משוב סגנוני המכוון פרשנות. תהליך זה עשוי לתרום לאיזון בין בהירות, גמישות פרשנית והעמקה אסתטית. הוא גם תורם להבנה אסתטית לאורך הקשרים משתנים."""),
        ("probe", UNDERSTOOD_PROBE),
        ("rating", UNDERSTANDING_RATING),
        ("probe", "האם המשמעות של נושא הים מתוארת כייצוג סגור ובלתי משתנה?", "No"),
    ]},
    {"number": 6, "type": "קשה", "steps": [
        ("text", """הצדדים מסכימים כי מסמך זה מהווה חוזה מחייב כולל ואחיד אשר תניותיו יוצרות מארג נורמטיבי שלם המחייב את הצדדים בכל היבטיו. פרשנות החוזה תיעשה כמכלול קוהרנטי, וכותרות הפרקים נועדו לנוחות בלבד ואינן גורעות מתניה מהותית כלשהי. המשכיר מצהיר כי הנכס מושכר ללא הקניית זכות קניינית, ללא יצירת מעמד סטטוטורי ובהתאם לדין, וכי כל תשלום שישולם ייחשב כדמי שכירות בלבד ולא יקנה זכות עתידית או חזקה אוטומטית."""),
        ("probe", UNDERSTOOD_PROBE),
        ("text", """השוכר מוותר מראש על כל טענה ומצהיר כי השימוש בנכס נעשה באחריותו הבלעדית. הוראות החוזה גוברות על כל מסמך אחר, וכל מחלוקת תפורש בהתאם ללשון החוזה ותתברר לפי הדין החל במסגרת ההתקשרות החוזית המחייבת וללא אפשרות לסטייה פרשנית נוספת מצד מי מהצדדים בכל שלב של מימוש החוזה או יישומו המעשי בפועל לאורך תקופת השכירות המוסכמת והמחייבת בין הצדדים, במלואה ובתוקף מלא."""),
        ("probe", UNDERSTOOD_PROBE),
        ("rating", UNDERSTANDING_RATING),
        ("probe", "האם השוכר אחראי באופן בלעדי לשימוש שלו בנכס?", "Yes"),
    ]},
    {"number": 7, "type": "מעניין", "steps": [
        ("text", """חברות היא נושא בעל חשיבות עבור רבים. היא מתוארת כקשר מתמשך שנבנה מתוך חוויות משותפות ודרכי תקשורת שונות במצבים משתנים. קשר זה מתואר כמבנה דינמי של אמון, ולא כמצב קבוע. המחקר מבחין בין חברות שטחית לבין חברות עמוקה בהקשרים כמו עבודה, לימודים או שכונה. שני סוגי החברות יכולים להתפתח לאורך החיים דרך מפגשים חוזרים, שיחות על חוויות ותמיכה הדדית. בלי תחושת חברות, אנשים עלולים לחוות בדידות, להפחית יוזמה ולהימנע ממעורבות חברתית. לעומת זאת, תחושת חברות יציבה עשויה לחזק ביטחון, לעודד פתיחות ולהרחיב תחומי עשייה. שינויים בחברות מתוארים כתהליכים של התקרבות הדדית ובחינה של דפוסי התנהגות. תהליך זה עוזר להכווין בחירות ודרכי פעולה. מחקרים מצביעים על כך שמחלוקת מתונה יכולה לתרום לחיזוק הקשר ולהעמקת ההיכרות ההדדית לאורך זמן."""),
        ("probe", UNDERSTOOD_PROBE),
        ("rating", UNDERSTANDING_RATING),
        ("probe", "האם מחלוקת מתונה בהכרח פוגעת בחברות?", "No"),
    ]},
    {"number": 8, "type": "קל", "steps": [
        ("text", """שתיית מים חשובה מאוד לבריאות ולהרגשה הכללית של הגוף והנפש. כששותים מספיק מים ביום, הגוף והמוח פועלים היטב, ויש יותר כוח לפעילויות. מבוגרים צריכים לשתות כמות קבועה של כוסות בכל יום, וילדים צריכים לשתות לעיתים קרובות כדי לגדול, ללמוד ולהתפתח."""),
        ("probe", UNDERSTOOD_PROBE),
        ("text", """שתייה מספקת ומסודרת עוזרת להרגיש בבוקר ערניים, רגועים ומוכנים לפעילויות היום, לכן כדאי לשתות כוס מים אחרי הקימה ולא לחכות לתחושת צמא. בימים חמים כדאי לא לצאת בלי בקבוק, לא לוותר על הפסקות או לשכוח לשתות, מכיוון שדברים אלו מקשים על הגוף. רצוי לאכול פירות טריים במהלך היום, ולשמור על שגרה קבועה, שתעזור לגוף להתרענן ולהרגיש טוב. הקפדה על הרגלי שתייה טובים יוצרת הרגשה טובה. עם הרגלי שתייה קבועים, מרגישים פחות עייפים, וכאבי הראש פוחתים. מומלץ להקפיד על הרגלים קבועים כדי לשפר ריכוז ויעילות במשימות יומיומיות."""),
        ("probe", UNDERSTOOD_PROBE),
        ("rating", UNDERSTANDING_RATING),
        ("probe", "האם כדאי לשתות כוס מים אחרי הקימה בבוקר?", "Yes"),
    ]},
]

CLOSING_MESSAGE = "תודה על ההשתתפות!"  # Shown after the last text, ESC ends the session


# ============================================================================
# 12. LEGACY SETTINGS
# ============================================================================

VIDEO_QUALITY = 80                  # 0-100 quality setting
FADE_IN_FRAMES = 10                 # Frames for point fade in
ANALYSIS_EXPORT_FORMAT = "detailed" # "detailed" or "simple"
SCORING_METRIC = 'neg_mean_squared_error'
CYAN = (255, 255, 0)
ERROR_COLOR = RED
SUCCESS_COLOR = GREEN

RANDOM_FOREST_PARAMS = {
    'n_estimators': [300, 350, 400],
    'max_depth': [2, 4, 8, None],
    'min_samples_split': [2, 3, 5],
    'min_samples_leaf': [2, 3, 4],
    'max_features': ['sqrt', 'log2']
}