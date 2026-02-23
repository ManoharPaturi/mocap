# Camera Settings
CAMERA_ID = 0  # Device ID for cv2.VideoCapture (integer: 0, 1, 2, ...)
NETWORK_CAMERA_ID = 'cam_0'  # Network identifier (string: 'cam_0', 'cam_1', ...)
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FPS = 30

# Detection Settings
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# Model Selection
# Options: 'LITE' (Fastest), 'FULL' (Balanced), 'HEAVY' (Most Accurate)
POSE_MODEL_COMPLEXITY = 'FULL' 

# Multi-Person Settings
NUM_POSES = 5
NUM_FACES = 5
NUM_HANDS = 5

# Internal Paths (Do not edit unless moving files)
MODEL_PATHS = {
    'LITE': 'models/pose_landmarker_lite.task',
    'FULL': 'models/pose_landmarker_full.task',
    'HEAVY': 'models/pose_landmarker_heavy.task',
    'FACE': 'models/face_landmarker.task',
    'HAND': 'models/hand_landmarker.task'
}

# Visualization Settings
DRAW_LANDMARKS = True
SHOW_FPS = True
THEME_COLOR = (255, 100, 0)

# Database Settings
DB_TYPE = 'sqlite'  # 'sqlite' or 'postgres'
DB_PATH = 'mocap_data.db'  # For SQLite
POSTGRES_CONNECTION = {  # For PostgreSQL (set DB_TYPE='postgres' to use)
    'host': 'localhost',
    'port': 5432,
    'database': 'mocap',
    'user': 'postgres',
    'password': 'your_password'
}

# ============================================
# VS5 ADVANCED CONFIGURATION
# ============================================

# --- Image Preprocessing ---
# Frame Resizing
MAX_FRAME_WIDTH = 1280  # Resize frames wider than this for consistency

# Gamma Correction
GAMMA_DEFAULT = 1.0     # Default gamma value (1.0 = disabled)
GAMMA_MIN = 1.0         # Minimum gamma value
GAMMA_MAX = 1.4         # Maximum gamma value

# Face-Guided Exposure
FACE_EXPOSURE_TARGET = 140.0    # Target brightness for face region (0-255)
FACE_EXPOSURE_MIN_GAIN = 0.5    # Minimum gain factor (prevents over-darkening)
FACE_EXPOSURE_MAX_GAIN = 2.5    # Maximum gain factor (prevents noise amplification)
FACE_EXPOSURE_MIN_BRIGHTNESS = 10.0  # Minimum face brightness to avoid div-by-zero

# CLAHE (Contrast Limited Adaptive Histogram Equalization)
CLAHE_CLIP_LIMIT = 2.0      # Clip limit (1.0-4.0, higher = more contrast)
CLAHE_TILE_GRID_SIZE = (8, 8)  # Tile grid size for adaptive equalization

# --- Physics Engine ---
# Visibility Thresholds
VISIBILITY_HARD_GATE = 0.5      # Landmarks below this are ignored (latched to previous)
VISIBILITY_MIN_METRIC = 0.5     # Minimum visibility for metric calculations

# 1 Euro Filter Parameters (Optimized for 30 FPS)
FILTER_MIN_CUTOFF = 1.0     # Minimum cutoff frequency (Hz) - smooth slow motion
FILTER_BETA = 0.005         # Speed coefficient - responsiveness during fast motion
FILTER_D_CUTOFF = 1.0       # Derivative cutoff frequency (Hz)

# Physics Constraints
CALIBRATION_FRAMES = 30     # Number of frames to learn skeleton (1 sec @ 30fps)
BONE_LENGTH_STRICTNESS_HIGH_VIS = 0.85  # Constraint strength when visibility > 0.5
BONE_LENGTH_STRICTNESS_LOW_VIS = 0.99   # Constraint strength when visibility < 0.5

# --- Metric Calculations ---
# Outlier Rejection
ANGLE_OUTLIER_BASE_THRESHOLD = 50.0     # Base threshold for angle spikes (degrees)
ANGLE_OUTLIER_VELOCITY_COEFF = 0.1      # Velocity-dependent coefficient

# Velocity Limits
MAX_LINEAR_VELOCITY = 6.0   # Maximum limb velocity (m/s) - sanity check

# Smoothing
SMOOTHING_ALPHA_DEFAULT = 0.5   # Default EMA alpha for metrics
SMOOTHING_ALPHA_FACE = 0.3      # Stronger smoothing for facial metrics

# Normalization
BODY_HEIGHT_MULTIPLIER = 2.0    # Multiply hip-ankle distance by this for full height
IPD_DEFAULT = 0.065             # Default interpupillary distance (meters) if detection fails

# --- Detector Toggles (Default States) ---
ENABLE_FACE_DETECTION = True
ENABLE_HAND_DETECTION = True
ENABLE_FACE_EXPOSURE = False    # Face-guided exposure disabled by default

# --- ROI-Based Pose Cropping ---
# Crop to person bounding box for higher effective resolution
ENABLE_ROI_CROPPING = True          # Enable person-centered ROI cropping
ROI_EXPANSION_FACTOR = 0.25         # Expand bbox by 25% (0.2-0.3 recommended)
ROI_MIN_SIZE = 200                  # Minimum ROI size (pixels)
ROI_TARGET_SIZE = 640               # Resize ROI to this size for inference
ROI_SMOOTHING_ALPHA = 0.7           # Smooth ROI position (0.5-0.8)

# --- Performance ---
# Frame Skip (for low-end hardware)
FRAME_SKIP = 0  # Process every Nth frame (0 = no skip, 1 = every other frame)

# --- Presets ---
# You can define presets for different scenarios
PRESETS = {
    'indoor': {
        'gamma': 1.2,
        'face_exposure': True,
        'clahe_clip': 2.0
    },
    'outdoor': {
        'gamma': 1.0,
        'face_exposure': False,
        'clahe_clip': 2.0
    },
    'low_light': {
        'gamma': 1.3,
        'face_exposure': True,
        'clahe_clip': 3.0
    },
    'high_speed': {
        'gamma': 1.0,
        'face_exposure': False,
        'clahe_clip': 2.0,
        'enable_face': False,
        'enable_hand': False,
        'filter_beta': 0.01  # More responsive
    }
}

# ============================================
# MULTI-CAMERA SETTINGS
# ============================================

# Multi-Camera Mode
ENABLE_MULTI_CAMERA = False
CAMERA_ROLE = 'single'  # 'single', 'server', or 'master'
NUM_CAMERAS = 2         # Total number of cameras in setup

# --- GUI Integration Mode (NEW) ---
# Set this based on which laptop you're using:
# 'single' - Normal single-camera mode (default)
# 'server' - Full GUI + broadcasts to network (PC1)
# 'master' - Full GUI + receives from network + dual display (PC2)
MULTI_CAMERA_MODE = 'single'
REMOTE_CAMERA_IP = None  # IP of other laptop (e.g., '10.51.179.228')

# Network Configuration
MASTER_IP = '192.168.1.100'  # IP address of master coordinator
DISCOVERY_PORT = 6000        # Port for camera discovery broadcasts
DATA_PORT = 6001            # Port for frame data transmission
NETWORK_PROTOCOL = 'tcp'     # 'udp' (faster) or 'tcp' (reliable)

# Frame Synchronization
SYNC_TIME_THRESHOLD_MS = 100.0  # Max time difference for frame matching (ms)
FRAME_BUFFER_SIZE = 10         # Number of frames to buffer per camera

# Calibration
CALIBRATION_FILE = 'calibration.json'
AUTO_LOAD_CALIBRATION = True

# Data Compression
COMPRESS_NETWORK_DATA = True  # Use msgpack compression

# Multi-View Fusion
TRIANGULATION_MIN_VIEWS = 2      # Minimum cameras to see landmark for triangulation
REPROJECTION_ERROR_THRESHOLD = 15.0  # Max reprojection error (pixels)
CONFIDENCE_WEIGHT_VISIBILITY = 0.6   # Weight for visibility in confidence calculation
CONFIDENCE_WEIGHT_REPROJ = 0.4       # Weight for reprojection error in confidence

