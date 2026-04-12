"""Cau hinh chung cho he thong Action Recognition - Jump / Bend."""

# --- Camera ---
CAMERA_ID = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

# --- Display ---
WINDOW_NAME = "Action Recognition - Dodge Game"
GAME_PANEL_WIDTH = 1280
FPS_SMOOTHING = 0.9
SHOW_DEBUG = True

# --- Pose Estimation (MediaPipe Tasks API) ---
POSE_MIN_DETECTION_CONFIDENCE = 0.5
POSE_MIN_TRACKING_CONFIDENCE = 0.5

# --- Action Recognition (ML Model) ---
MODEL_PATH = "model.pkl"
MODEL_FEATURE_WINDOW = 5
MODEL_MIN_READY_FRAMES = 2
MODEL_CONFIDENCE_THRESHOLD = 0.70

# --- Game Character ---
CHAR_GROUND_Y = 500
CHAR_JUMP_VELOCITY = -20
CHAR_GRAVITY = 2
CHAR_COLOR = (0, 255, 100)

# --- UI Colors ---
COLOR_ACTION = (0, 255, 255)
COLOR_FPS = (0, 200, 0)
COLOR_STATUS = (200, 200, 200)
COLOR_TITLE = (255, 255, 255)
