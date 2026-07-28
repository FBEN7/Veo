"""Centralized constants for football match analysis pipeline."""

# ============================================================================
# Pitch dimensions (in metres)
# ============================================================================
PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0

# ============================================================================
# Detection thresholds and parameters
# ============================================================================
PERSON_CLASS_ID = 0
BALL_CLASS_ID = 32  # "sports ball" in COCO

# YOLO confidence thresholds
DEFAULT_CONF_PLAYER = 0.25
DEFAULT_CONF_BALL = 0.08

# Tracking parameters
TRACK_ACTIVATION_THRESHOLD = 0.08
LOST_TRACK_BUFFER = 180
MINIMUM_MATCHING_THRESHOLD = 0.78
MIN_DETECTION_HEIGHT_PX = 16.0

# ============================================================================
# Pitch masking and filtering
# ============================================================================
# HSV range for grass detection (in OpenCV HSV colorspace)
GRASS_H_MIN, GRASS_H_MAX = 28, 95
GRASS_S_MIN, GRASS_S_MAX = 35, 255
GRASS_V_MIN, GRASS_V_MAX = 25, 255

# Pitch bounds with tolerance (in metres)
PITCH_X_MIN, PITCH_X_MAX = -5.0, PITCH_LENGTH_M + 5.0
PITCH_Y_MIN, PITCH_Y_MAX = -5.0, PITCH_WIDTH_M + 5.0

# ============================================================================
# Physical stats calculation
# ============================================================================
SPRINT_THRESHOLD_KMH = 20.0
MAX_PLAUSIBLE_SPEED_KMH = 40.0
MIN_TRACK_LENGTH_FRAMES = 20
SPEED_CONVERSION_FACTOR = 3.6  # m/s to km/h

# ============================================================================
# Re-identification (ReID) parameters
# ============================================================================
REID_TEMPORAL_GAP_MAX_S = 120.0
REID_TEMPORAL_BONUS_MAX_S = 30.0
REID_DISTANCE_THRESHOLD = 1.35
REID_HUE_WRAP = 180.0

# Feature distance weights for ReID
REID_WEIGHTS = {
    "hue": 2.3,
    "saturation": 1.1,
    "value": 0.8,
    "std_dev": 0.5,
    "height": 0.6,
    "size": 0.4,
    "spatial": 0.7,
}

# ============================================================================
# Team assignment (color clustering)
# ============================================================================
N_TEAM_CLUSTERS = 3  # Two teams + "other" (refs, bench)
KMEANS_N_INIT = 10
MIN_TRACKS_FOR_CLUSTERING = 3
MIN_TRACK_LENGTH_FOR_COLOR = 10

# ============================================================================
# Event detection parameters
# ============================================================================
POSSESSION_RADIUS_M = 4.4
POSSESSION_GAP_FILL_S = 1.4
MIN_PASS_DISTANCE_M = 1.2
MAX_PASS_INTERVAL_S = 4.5
PASS_UNKNOWN_BRIDGE_S = 1.4
SHOT_MIN_KMH = 16.0
SHOT_MAX_DIST_FROM_GOAL_M = 40.0
SHOT_COOLDOWN_S = 1.8
CARRY_MIN_TIME_S = 2.0
CARRY_MIN_DISTANCE_M = 8.0
RECOVERY_MIN_LOOSE_S = 0.8
TACKLE_MAX_PLAYER_DIST_M = 2.5
TACKLE_MAX_BALL_SPEED_KMH = 28.0

# Goal geometry
GOAL_WIDTH_M = 7.32
GOAL_HALF_WIDTH_M = GOAL_WIDTH_M / 2.0
GOAL_DEPTH_M = 2.0
GOAL_CENTER_Y = PITCH_WIDTH_M / 2.0
LEFT_GOAL_X_MAX = GOAL_DEPTH_M
LEFT_GOAL_Y_MIN = GOAL_CENTER_Y - GOAL_HALF_WIDTH_M
LEFT_GOAL_Y_MAX = GOAL_CENTER_Y + GOAL_HALF_WIDTH_M
RIGHT_GOAL_X_MIN = PITCH_LENGTH_M - GOAL_DEPTH_M
RIGHT_GOAL_Y_MIN = GOAL_CENTER_Y - GOAL_HALF_WIDTH_M
RIGHT_GOAL_Y_MAX = GOAL_CENTER_Y + GOAL_HALF_WIDTH_M

# ============================================================================
# Player rating parameters
# ============================================================================
MIN_MINUTES_TRACKED_FOR_RATING = 5.0
GOAL_BONUS = 1.50
RATING_WEIGHTS = {
    "distance_m": 0.20,
    "n_sprints": 0.15,
    "n_passes": 0.25,
    "n_shots": 0.20,
    "possession_pct": 0.10,
}

# ============================================================================
# Quality assessment thresholds
# ============================================================================
BALL_DETECTION_RATE_THRESHOLD = 0.5
MIN_PLAYERS_PER_FRAME = 8.0
MIN_LONG_TRACK_RATIO = 0.6
MIN_PASSES_PER_MINUTE = 4.0

# ============================================================================
# Video processing
# ============================================================================
DEFAULT_FPS = 25.0
DEFAULT_IMG_SIZE = 640
DEFAULT_MODEL = "yolov8m.pt"
DEFAULT_STRIDE = 1
