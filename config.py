# config.py
# Eye Tracker Configuration Constants
# This file contains all the configuration constants and settings for the eye-tracking application.
# By centralizing these values, you can easily tweak the application's behavior without
# modifying the core logic.

# --- Display and Camera Settings ---
WINDOW_WIDTH = 1080
WINDOW_HEIGHT = 720

# --- Data Capture and Processing ---
CAPTURE_FRAMES = 10  # Number of frames to capture for each calibration point
SMOOTHING_FACTOR = 0.2  # Smoothing factor for real-time gaze prediction (0.0-1.0)

# --- User Positioning Boundary (Normalized Coordinates) ---
# This defines the target area where the user's face must be positioned.
# Values are percentages of the screen width/height.
BOUNDARY_LEFT = 0.375
BOUNDARY_RIGHT = 0.625
BOUNDARY_TOP = 0.3
BOUNDARY_BOTTOM = 0.7

# --- Monitoring and Quality Thresholds ---
# (Note: FACE_DETECTION_THRESHOLD is defined but not currently used in the main logic)
FACE_DETECTION_THRESHOLD = 0.85

# --- Calibration Target Generation ---
# This section programmatically generates the list of calibration targets.
# Modifying the margins will change the layout of the calibration points.

# Define margins
edge_margin = 20  # Small margin from the absolute screen edges in pixels
grid_margin_percent = 0.2  # 20% margin from the edges for the main grid

# Calculate grid boundaries based on window dimensions
grid_left = int(WINDOW_WIDTH * grid_margin_percent)
grid_right = int(WINDOW_WIDTH * (1 - grid_margin_percent))
grid_top = int(WINDOW_HEIGHT * grid_margin_percent)
grid_bottom = int(WINDOW_HEIGHT * (1 - grid_margin_percent))

# 1. Four corners of the screen
corners = [
    (edge_margin, edge_margin),                            # Top-left corner
    (WINDOW_WIDTH - edge_margin, edge_margin),             # Top-right corner
    (edge_margin, WINDOW_HEIGHT - edge_margin),            # Bottom-left corner
    (WINDOW_WIDTH - edge_margin, WINDOW_HEIGHT - edge_margin)  # Bottom-right corner
]

# 2. Four edges (middle of each edge)
edges = [
    (WINDOW_WIDTH // 2, edge_margin),                      # Top edge center
    (WINDOW_WIDTH // 2, WINDOW_HEIGHT - edge_margin),      # Bottom edge center
    (edge_margin, WINDOW_HEIGHT // 2),                     # Left edge center
    (WINDOW_WIDTH - edge_margin, WINDOW_HEIGHT // 2)       # Right edge center
]

# 3. Enhanced 13-point grid in the middle
grid_points = []
original_grid = []
for row in range(3):
    for col in range(3):
        x = grid_left + col * (grid_right - grid_left) // 2
        y = grid_top + row * (grid_bottom - grid_top) // 2
        original_grid.append((x, y))
        grid_points.append((x, y))

# Add intermediate points between specific grid targets for more comprehensive data
tr_x, tr_y = original_grid[2]   # Top-Right
mc_x, mc_y = original_grid[4]   # Middle-Center
tl_x, tl_y = original_grid[0]   # Top-Left
bl_x, bl_y = original_grid[6]   # Bottom-Left
br_x, br_y = original_grid[8]   # Bottom-Right

grid_points.append(((tr_x + mc_x) // 2, (tr_y + mc_y) // 2))
grid_points.append(((tl_x + mc_x) // 2, (tl_y + mc_y) // 2))
grid_points.append(((bl_x + mc_x) // 2, (bl_y + mc_y) // 2))
grid_points.append(((br_x + mc_x) // 2, (br_y + mc_y) // 2))

# Combine all calibration targets into a single list
CALIBRATION_TARGETS = corners + edges + grid_points

# Labels for each target to be displayed during calibration
TARGET_LABELS = (
    ["Corner TL", "Corner TR", "Corner BL", "Corner BR"] +
    ["Edge Top", "Edge Bottom", "Edge Left", "Edge Right"] +
    ["Grid TL", "Grid TC", "Grid TR", "Grid ML", "Grid MC", "Grid MR", "Grid BL", "Grid BC", "Grid BR"] +
    ["Mid TR-MC", "Mid TL-MC", "Mid BL-MC", "Mid BR-MC"]
)

# Additional Calibration Settings
TARGET_RADIUS = 30
STABLE_DURATION = 1.0

# Model Settings
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5
