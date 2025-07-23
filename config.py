# Configuration parameters for Critical Region Object Throw Detection System

# Motion detection thresholds
MOTION_DETECTION_THRESHOLD = 25
TEMPORAL_CONSISTENCY_FRAMES = 4  # Number of frames for temporal check

# Object size limits (in pixels)
MIN_OBJECT_AREA = 100
MAX_OBJECT_AREA = 2500

# Speed thresholds (in pixels/frame)
MIN_SPEED_THRESHOLD = 50

# Morphological operation kernel sizes
MORPH_KERNEL_SIZE = (5, 5)

# Alert settings
ALERT_SNAPSHOT_SAVE = True
ALERT_SNAPSHOT_DIR = 'snapshots'

# ROI settings
ROI_SAVE_PATH = 'roi_coords.npy'

# Video settings
TARGET_FPS = 20

# Video source settings
DEFAULT_SOURCE_TYPE = "webcam"
RTSP_CONNECTION_TIMEOUT = 10
RTSP_RECONNECT_ATTEMPTS = 3
RTSP_RECONNECT_DELAY = 5
VIDEO_FILE_LOOP = True
SUPPORTED_VIDEO_FORMATS = ['.mp4', '.avi', '.mov', '.mkv', '.flv']
UPLOADED_VIDEOS_DIR = 'uploaded_videos'
RECORDED_STREAMS_DIR = 'recorded_streams'

# Add detection cooldown for event logging
DETECTION_COOLDOWN = 2  # seconds

# Default zone points (rectangle in normalized coordinates, can be adjusted as needed)
DEFAULT_ZONE_POINTS = [
    [180, 135],
    [720, 135],
    [720, 540],
    [180, 540]
]

# Confidence threshold for detection (legacy, for compatibility)
CONFIDENCE_THRESHOLD = 0.5

# Directory for saving screenshots
SCREENSHOT_DIR = 'screenshots' 