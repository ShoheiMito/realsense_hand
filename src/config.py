"""Configuration constants for RealSense hand control system."""

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


@dataclass
class OneEuroFilterParams:
    """Parameters for One Euro Filter (temporal smoothing)."""

    min_cutoff: float = 1.0
    beta: float = 0.007
    d_cutoff: float = 1.0


# MediaPipe HandLandmarker settings
HAND_MODEL_PATH: str = "models/hand_landmarker.task"
HAND_MIN_DETECTION_CONFIDENCE: float = 0.5
HAND_MIN_TRACKING_CONFIDENCE: float = 0.5
HAND_NUM_HANDS: int = 2
HAND_SKIP_FRAMES: int = 1  # Run hand detection every N frames

# Depth valid range (meters)
DEPTH_MIN_DISTANCE: float = 0.25
DEPTH_MAX_DISTANCE: float = 4.0

# Visualization settings
SKELETON_LINE_THICKNESS: int = 3
SKELETON_CIRCLE_RADIUS: int = 5
WINDOW_NAME: str = "RealSense Hand Control"

# ---------------------------------------------------------------------------
# Hand controller settings
# ---------------------------------------------------------------------------
CONTROL_ACTIVE_REGION: float = 0.7       # カメラ画面の中央何%を操作領域にするか
CONTROL_SENSITIVITY: float = 2.5        # 相対カーソル感度（カメラ1px→画面何px）
CONTROL_DEADZONE_PX: int = 2             # この画素数以下の移動は無視（カメラピクセル）
CONTROL_PINCH_THRESHOLD_PX: int = 45     # ピンチ検出閾値（ピクセル距離）
CONTROL_PINCH_RELEASE_THRESHOLD_PX: int = 60  # ピンチ解除閾値（ヒステリシス）
CONTROL_CLICK_MAX_DURATION_S: float = 0.3     # クリック判定の最大保持時間
CONTROL_CLICK_MAX_MOVE_PX: int = 20      # クリック判定の最大移動量
CONTROL_DRAG_MIN_HOLD_S: float = 0.25    # ドラッグ判定の最小保持時間
CONTROL_SCROLL_SENSITIVITY: float = 3.0  # スクロール感度
CONTROL_HAND_LOST_FRAMES: int = 5        # 手消失判定フレーム数
CONTROL_GESTURE_CONFIRM_FRAMES: int = 3  # ジェスチャー確定フレーム数
CONTROL_MIRROR_X: bool = True            # X軸ミラーリング

# クラッチ制御
CONTROL_CLUTCH_ENABLED: bool = True      # クラッチ方式を有効にする

# 3Dピンチ検出（メートル単位）
CONTROL_PINCH_THRESHOLD_3D_M: float = 0.040       # 40mm ピンチ検出
CONTROL_PINCH_RELEASE_THRESHOLD_3D_M: float = 0.055  # 55mm ピンチ解除
CONTROL_USE_3D_PINCH: bool = False       # 実機チューニング後にTrueにする

# カーソルOne Euroフィルターパラメータ
CURSOR_ONE_EURO_MIN_CUTOFF: float = 0.5
CURSOR_ONE_EURO_BETA: float = 0.01
CURSOR_ONE_EURO_D_CUTOFF: float = 1.0
