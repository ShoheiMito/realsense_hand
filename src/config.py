"""Configuration constants for RealSense pose estimation system."""

from dataclasses import dataclass


# Camera settings
CAMERA_WIDTH: int = 640
CAMERA_HEIGHT: int = 480
CAMERA_FPS: int = 30

# Queue settings
FRAME_QUEUE_SIZE: int = 2
RESULT_QUEUE_SIZE: int = 2

# Depth filter parameters
SPATIAL_FILTER_MAGNITUDE: int = 1
SPATIAL_FILTER_ALPHA: float = 0.5
SPATIAL_FILTER_DELTA: int = 20

TEMPORAL_FILTER_ALPHA: float = 0.4
TEMPORAL_FILTER_DELTA: int = 20

HOLE_FILLING_MODE: int = 1  # 0: fill from left, 1: farthest from around, 2: nearest

# Depth fallback settings
DEPTH_FALLBACK_KERNEL_SIZE: int = 5

# Expression recognition settings
EXPRESSION_SKIP_FRAMES: int = 5  # Run expression recognition every N frames


@dataclass
class OneEuroFilterParams:
    """Parameters for One Euro Filter (temporal smoothing)."""

    min_cutoff: float = 1.0
    beta: float = 0.007
    d_cutoff: float = 1.0


# MediaPipe Pose model settings
POSE_MODEL_PATH: str = "models/pose_landmarker_full.task"
POSE_MIN_DETECTION_CONFIDENCE: float = 0.5
POSE_MIN_TRACKING_CONFIDENCE: float = 0.5
POSE_NUM_POSES: int = 1

# MediaPipe FaceLandmarker settings
FACE_MODEL_PATH: str = "models/face_landmarker.task"
FACE_MIN_DETECTION_CONFIDENCE: float = 0.5
FACE_MIN_TRACKING_CONFIDENCE: float = 0.5

# Depth valid range (meters)
DEPTH_MIN_DISTANCE: float = 0.25
DEPTH_MAX_DISTANCE: float = 4.0

# Visualization settings
SKELETON_LINE_THICKNESS: int = 3
SKELETON_CIRCLE_RADIUS: int = 5
WINDOW_NAME: str = "RealSense 3D Pose"
