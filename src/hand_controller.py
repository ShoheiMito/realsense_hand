"""Hand gesture recognition and mouse control.

Translates MediaPipe hand landmarks into PC screen operations:
- Cursor movement (index finger tip tracking)
- Click (quick pinch: thumb + index finger)
- Drag (pinch and hold while moving)
- Scroll (index + middle finger extended, vertical movement)
"""

from __future__ import annotations

import ctypes
import enum
import logging
import math
import time
from dataclasses import dataclass

from src import config
from src.depth_utils import OneEuroFilter
from src.processor import HandResult, PoseKeypoint3D

try:
    from pynput.mouse import Button, Controller as MouseController

    _PYNPUT_AVAILABLE = True
except ImportError:
    _PYNPUT_AVAILABLE = False

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Gesture state machine
# ---------------------------------------------------------------------------


class GestureState(enum.Enum):
    """Gesture states for the hand controller."""

    IDLE = "idle"
    NEUTRAL = "neutral"      # 手は見えているがカーソル固定
    CURSOR = "cursor"
    CLICK_DOWN = "click_down"
    DRAGGING = "dragging"
    SCROLLING = "scrolling"


@dataclass
class GestureInfo:
    """Snapshot of current gesture state for visualization."""

    state: GestureState
    cursor_screen: tuple[int, int] | None = None
    index_tip_px: tuple[int, int] | None = None
    pinch_distance: float = 0.0
    pinch_distance_3d: float | None = None
    is_pinching: bool = False
    control_active: bool = True


# ---------------------------------------------------------------------------
# GestureDetector — stateless per-frame feature extraction
# ---------------------------------------------------------------------------


class GestureDetector:
    """Stateless per-frame gesture feature extraction from hand landmarks."""

    @staticmethod
    def pinch_distance(landmarks_2d: list[tuple[int, int]]) -> float:
        """Compute pixel distance between thumb tip and index finger tip.

        Args:
            landmarks_2d: 21 hand landmarks in pixel coordinates.

        Returns:
            Euclidean distance in pixels.
        """
        thumb_tip = landmarks_2d[4]
        index_tip = landmarks_2d[8]
        dx = thumb_tip[0] - index_tip[0]
        dy = thumb_tip[1] - index_tip[1]
        return math.sqrt(dx * dx + dy * dy)

    @staticmethod
    def is_finger_extended(
        landmarks_2d: list[tuple[int, int]],
        tip_idx: int,
        pip_idx: int,
    ) -> bool:
        """Check if a finger is extended (tip above PIP joint in image space).

        Args:
            landmarks_2d: 21 hand landmarks in pixel coordinates.
            tip_idx: Tip landmark index.
            pip_idx: PIP joint landmark index.

        Returns:
            True if the finger tip Y is above (less than) PIP Y.
        """
        return landmarks_2d[tip_idx][1] < landmarks_2d[pip_idx][1]

    @staticmethod
    def is_scroll_pose(landmarks_2d: list[tuple[int, int]]) -> bool:
        """Detect scroll gesture: index + middle extended, others folded.

        Args:
            landmarks_2d: 21 hand landmarks in pixel coordinates.

        Returns:
            True if scroll pose is detected.
        """
        index_ext = GestureDetector.is_finger_extended(landmarks_2d, 8, 6)
        middle_ext = GestureDetector.is_finger_extended(landmarks_2d, 12, 10)
        ring_ext = GestureDetector.is_finger_extended(landmarks_2d, 16, 14)

        return index_ext and middle_ext and not ring_ext

    @staticmethod
    def is_pointing_pose(landmarks_2d: list[tuple[int, int]]) -> bool:
        """Detect index-only pointing: index extended, middle+ring folded.

        Args:
            landmarks_2d: 21 hand landmarks in pixel coordinates.

        Returns:
            True if only the index finger is extended.
        """
        index_ext = GestureDetector.is_finger_extended(landmarks_2d, 8, 6)
        middle_ext = GestureDetector.is_finger_extended(landmarks_2d, 12, 10)
        ring_ext = GestureDetector.is_finger_extended(landmarks_2d, 16, 14)
        return index_ext and not middle_ext and not ring_ext

    @staticmethod
    def is_open_hand(landmarks_2d: list[tuple[int, int]]) -> bool:
        """Detect open hand: index, middle, and ring all extended.

        Args:
            landmarks_2d: 21 hand landmarks in pixel coordinates.

        Returns:
            True if 3+ fingers are extended (open palm).
        """
        index_ext = GestureDetector.is_finger_extended(landmarks_2d, 8, 6)
        middle_ext = GestureDetector.is_finger_extended(landmarks_2d, 12, 10)
        ring_ext = GestureDetector.is_finger_extended(landmarks_2d, 16, 14)
        return index_ext and middle_ext and ring_ext

    @staticmethod
    def pinch_distance_3d(
        keypoints_3d: list[PoseKeypoint3D] | None,
    ) -> float | None:
        """Compute 3D distance in metres between thumb tip and index tip.

        Args:
            keypoints_3d: 21 hand landmarks in 3D world coordinates, or None.

        Returns:
            Distance in metres, or None if 3D data is unavailable.
        """
        if keypoints_3d is None or len(keypoints_3d) < 9:
            return None
        thumb = keypoints_3d[4]
        index = keypoints_3d[8]
        if thumb.z <= 0 or index.z <= 0:
            return None
        dx = thumb.x - index.x
        dy = thumb.y - index.y
        dz = thumb.z - index.z
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    @staticmethod
    def get_index_tip(landmarks_2d: list[tuple[int, int]]) -> tuple[int, int]:
        """Return index finger tip coordinates."""
        return landmarks_2d[8]

    @staticmethod
    def get_scroll_center(landmarks_2d: list[tuple[int, int]]) -> tuple[int, int]:
        """Return midpoint between index and middle finger tips."""
        ix, iy = landmarks_2d[8]
        mx, my = landmarks_2d[12]
        return ((ix + mx) // 2, (iy + my) // 2)


# ---------------------------------------------------------------------------
# CoordinateMapper — camera pixel → screen coordinates
# ---------------------------------------------------------------------------


class CoordinateMapper:
    """Relative coordinate mapper: camera movement delta → cursor movement.

    Works like a trackpad: hand movement is translated into cursor displacement,
    not mapped to an absolute screen position.
    """

    def __init__(
        self,
        camera_width: int = config.CAMERA_WIDTH,
        camera_height: int = config.CAMERA_HEIGHT,
        sensitivity: float = config.CONTROL_SENSITIVITY,
        deadzone_px: int = config.CONTROL_DEADZONE_PX,
        mirror_x: bool = config.CONTROL_MIRROR_X,
    ) -> None:
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.sensitivity = sensitivity
        self.deadzone_px = deadzone_px
        self.mirror_x = mirror_x

        # Auto-detect screen resolution (Windows)
        try:
            user32 = ctypes.windll.user32  # type: ignore[attr-defined]
            user32.SetProcessDPIAware()
            self.screen_width: int = user32.GetSystemMetrics(0)
            self.screen_height: int = user32.GetSystemMetrics(1)
        except Exception:
            self.screen_width = 1920
            self.screen_height = 1080
            logger.warning(
                "Could not detect screen resolution, using %dx%d",
                self.screen_width, self.screen_height,
            )

        # One Euro filter on camera input (smooth before delta calculation)
        self._filter_params = config.OneEuroFilterParams(
            min_cutoff=config.CURSOR_ONE_EURO_MIN_CUTOFF,
            beta=config.CURSOR_ONE_EURO_BETA,
            d_cutoff=config.CURSOR_ONE_EURO_D_CUTOFF,
        )
        self._filter_x = OneEuroFilter(self._filter_params)
        self._filter_y = OneEuroFilter(self._filter_params)

        # Previous smoothed camera position (for delta calculation)
        self._prev_cam_x: float | None = None
        self._prev_cam_y: float | None = None

        # Current screen cursor position
        self._cursor_x: float = self.screen_width / 2.0
        self._cursor_y: float = self.screen_height / 2.0

    def map(
        self, cam_x: int, cam_y: int, timestamp: float | None = None,
    ) -> tuple[int, int]:
        """Convert camera movement delta to cursor displacement.

        Args:
            cam_x: X pixel in camera frame.
            cam_y: Y pixel in camera frame.
            timestamp: Current time in seconds. Uses time.monotonic() if None.

        Returns:
            (screen_x, screen_y) coordinates.
        """
        if timestamp is None:
            timestamp = time.monotonic()

        # One Euro smoothing on camera coordinates
        smooth_x = self._filter_x(float(cam_x), timestamp)
        smooth_y = self._filter_y(float(cam_y), timestamp)

        # First frame: initialize, no movement
        if self._prev_cam_x is None or self._prev_cam_y is None:
            self._prev_cam_x = smooth_x
            self._prev_cam_y = smooth_y
            return self.get_cursor_pos()

        # Calculate camera delta
        delta_cx = smooth_x - self._prev_cam_x
        delta_cy = smooth_y - self._prev_cam_y
        self._prev_cam_x = smooth_x
        self._prev_cam_y = smooth_y

        # Dead zone — ignore tiny camera movements
        if abs(delta_cx) < self.deadzone_px and abs(delta_cy) < self.deadzone_px:
            return self.get_cursor_pos()

        # Mirror X axis (camera is mirrored)
        if self.mirror_x:
            delta_cx = -delta_cx

        # Apply sensitivity and move cursor
        self._cursor_x += delta_cx * self.sensitivity
        self._cursor_y += delta_cy * self.sensitivity

        # Clamp to screen bounds
        self._cursor_x = max(0.0, min(float(self.screen_width - 1), self._cursor_x))
        self._cursor_y = max(0.0, min(float(self.screen_height - 1), self._cursor_y))

        return self.get_cursor_pos()

    def get_cursor_pos(self) -> tuple[int, int]:
        """Return current cursor position without moving it."""
        return (int(self._cursor_x), int(self._cursor_y))

    def sync_cursor(self) -> None:
        """Sync internal cursor position with actual mouse position."""
        try:
            user32 = ctypes.windll.user32  # type: ignore[attr-defined]
            from ctypes import wintypes

            point = wintypes.POINT()
            user32.GetCursorPos(ctypes.byref(point))
            self._cursor_x = float(point.x)
            self._cursor_y = float(point.y)
        except Exception:
            pass

    def reset(self) -> None:
        """Reset filter state. Call when re-entering CURSOR mode."""
        self._filter_x = OneEuroFilter(self._filter_params)
        self._filter_y = OneEuroFilter(self._filter_params)
        self._prev_cam_x = None
        self._prev_cam_y = None
        # カーソル位置はリセットしない（現在位置から継続）


# ---------------------------------------------------------------------------
# HandController — orchestrator
# ---------------------------------------------------------------------------


class HandController:
    """Gesture state machine + mouse control via pynput.

    Call update() once per frame with the detected hands. Returns GestureInfo
    for visualization.
    """

    def __init__(self, preferred_hand: str = "Right") -> None:
        """Initialize the hand controller.

        Args:
            preferred_hand: Preferred handedness ('Left' or 'Right').

        Raises:
            RuntimeError: If pynput is not installed.
        """
        if not _PYNPUT_AVAILABLE:
            raise RuntimeError(
                "pynput is not installed. Run: pip install pynput"
            )

        self._preferred_hand = preferred_hand
        self._mouse = MouseController()
        self._detector = GestureDetector()
        self._mapper = CoordinateMapper()

        self._state = GestureState.IDLE
        self._control_active = True

        # Debounce counters
        self._hand_detected_frames: int = 0
        self._hand_lost_frames: int = 0
        self._scroll_confirm_frames: int = 0
        self._open_hand_frames: int = 0

        # Click/drag state
        self._pinch_start_time: float = 0.0
        self._pinch_start_screen: tuple[int, int] = (0, 0)
        self._is_pinching: bool = False

        # Scroll state
        self._scroll_prev_y: int | None = None

        # Last known info
        self._last_index_tip_px: tuple[int, int] | None = None
        self._last_pinch_dist: float = 0.0

    @property
    def control_active(self) -> bool:
        """Whether mouse control is currently active."""
        return self._control_active

    def toggle_control(self) -> bool:
        """Toggle mouse control on/off. Returns new state."""
        self._control_active = not self._control_active
        if not self._control_active:
            self._release_all()
            self._state = GestureState.IDLE
        logger.info("Control mode: %s", "ON" if self._control_active else "OFF")
        return self._control_active

    def update(self, hands: list[HandResult] | None) -> GestureInfo:
        """Process one frame of hand data and execute mouse actions.

        Args:
            hands: Detected hands from the processor, or None.

        Returns:
            GestureInfo for visualization.
        """
        hand = self._select_hand(hands)

        if hand is None:
            return self._handle_no_hand()

        landmarks = hand.landmarks_2d
        self._hand_lost_frames = 0
        self._hand_detected_frames += 1

        index_tip = self._detector.get_index_tip(landmarks)
        pinch_dist_2d = self._detector.pinch_distance(landmarks)
        is_scroll = self._detector.is_scroll_pose(landmarks)
        is_pointing = self._detector.is_pointing_pose(landmarks)
        is_open = self._detector.is_open_hand(landmarks)

        self._last_index_tip_px = index_tip
        self._last_pinch_dist = pinch_dist_2d

        # 3Dピンチ検出（利用可能な場合）、2Dフォールバック
        pinch_dist_3d = self._detector.pinch_distance_3d(hand.keypoints_3d)
        if config.CONTROL_USE_3D_PINCH and pinch_dist_3d is not None:
            if self._is_pinching:
                pinching = pinch_dist_3d < config.CONTROL_PINCH_RELEASE_THRESHOLD_3D_M
            else:
                pinching = pinch_dist_3d < config.CONTROL_PINCH_THRESHOLD_3D_M
            logger.debug(
                "Pinch 3D: %.1fmm thresh=%.0fmm pinching=%s",
                pinch_dist_3d * 1000,
                (config.CONTROL_PINCH_RELEASE_THRESHOLD_3D_M if self._is_pinching
                 else config.CONTROL_PINCH_THRESHOLD_3D_M) * 1000,
                pinching,
            )
        else:
            if self._is_pinching:
                pinching = pinch_dist_2d < config.CONTROL_PINCH_RELEASE_THRESHOLD_PX
            else:
                pinching = pinch_dist_2d < config.CONTROL_PINCH_THRESHOLD_PX
            logger.debug(
                "Pinch 2D: %.1fpx thresh=%dpx pinching=%s state=%s",
                pinch_dist_2d,
                (config.CONTROL_PINCH_RELEASE_THRESHOLD_PX if self._is_pinching
                 else config.CONTROL_PINCH_THRESHOLD_PX),
                pinching,
                self._state.value,
            )

        # Debounce hand detection: IDLE → NEUTRAL
        if self._state == GestureState.IDLE:
            if self._hand_detected_frames >= config.CONTROL_GESTURE_CONFIRM_FRAMES:
                self._state = GestureState.NEUTRAL
            return GestureInfo(
                state=self._state,
                index_tip_px=index_tip,
                pinch_distance=pinch_dist_2d,
                pinch_distance_3d=pinch_dist_3d,
                is_pinching=pinching,
                control_active=self._control_active,
            )

        # State transitions
        if self._state == GestureState.NEUTRAL:
            # NEUTRALではカーソルを動かさない
            screen_pos = self._mapper.get_cursor_pos()

            if config.CONTROL_CLUTCH_ENABLED:
                if is_pointing:
                    self._state = GestureState.CURSOR
                    self._mapper.reset()
                    self._mapper.sync_cursor()
                elif pinching:
                    # NEUTRALからのピンチ→カーソルはその場でクリック
                    self._enter_click_down(screen_pos)
            else:
                self._state = GestureState.CURSOR
                self._mapper.reset()
                self._mapper.sync_cursor()

        elif self._state == GestureState.CURSOR:
            screen_pos = self._mapper.map(index_tip[0], index_tip[1])
            if pinching:
                self._enter_click_down(screen_pos)
            elif is_scroll:
                self._scroll_confirm_frames += 1
                if self._scroll_confirm_frames >= config.CONTROL_GESTURE_CONFIRM_FRAMES:
                    self._enter_scrolling(landmarks)
            elif config.CONTROL_CLUTCH_ENABLED and is_open and not pinching:
                self._open_hand_frames += 1
                if self._open_hand_frames >= config.CONTROL_GESTURE_CONFIRM_FRAMES:
                    self._scroll_confirm_frames = 0
                    self._open_hand_frames = 0
                    self._state = GestureState.NEUTRAL
            else:
                self._scroll_confirm_frames = 0
                self._open_hand_frames = 0
                if is_pointing or not config.CONTROL_CLUTCH_ENABLED:
                    self._move_cursor(screen_pos)

        elif self._state == GestureState.CLICK_DOWN:
            screen_pos = self._mapper.get_cursor_pos()
            if not pinching:
                self._fire_click()
                if config.CONTROL_CLUTCH_ENABLED:
                    self._state = GestureState.NEUTRAL
                else:
                    self._state = GestureState.CURSOR
            else:
                elapsed = time.monotonic() - self._pinch_start_time
                move_dist = math.sqrt(
                    (screen_pos[0] - self._pinch_start_screen[0]) ** 2
                    + (screen_pos[1] - self._pinch_start_screen[1]) ** 2
                )
                if (
                    elapsed > config.CONTROL_DRAG_MIN_HOLD_S
                    or move_dist > config.CONTROL_CLICK_MAX_MOVE_PX
                ):
                    self._enter_dragging(screen_pos)

        elif self._state == GestureState.DRAGGING:
            screen_pos = self._mapper.map(index_tip[0], index_tip[1])
            if not pinching:
                self._release_drag()
                if config.CONTROL_CLUTCH_ENABLED:
                    self._state = GestureState.NEUTRAL
                else:
                    self._state = GestureState.CURSOR
            else:
                self._move_cursor(screen_pos)

        elif self._state == GestureState.SCROLLING:
            screen_pos = self._mapper.get_cursor_pos()
            if not is_scroll:
                self._scroll_confirm_frames = 0
                self._scroll_prev_y = None
                self._state = GestureState.CURSOR
            else:
                self._do_scroll(landmarks)

        self._is_pinching = pinching

        return GestureInfo(
            state=self._state,
            cursor_screen=screen_pos,
            index_tip_px=index_tip,
            pinch_distance=pinch_dist_2d,
            pinch_distance_3d=pinch_dist_3d,
            is_pinching=pinching,
            control_active=self._control_active,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _select_hand(self, hands: list[HandResult] | None) -> HandResult | None:
        """Select the preferred hand from detected hands."""
        if not hands:
            return None
        if len(hands) == 1:
            return hands[0]
        for h in hands:
            if h.handedness == self._preferred_hand:
                return h
        return hands[0]

    def _handle_no_hand(self) -> GestureInfo:
        """Handle frames where no hand is detected."""
        self._hand_detected_frames = 0
        self._hand_lost_frames += 1

        if self._hand_lost_frames >= config.CONTROL_HAND_LOST_FRAMES:
            if self._state in (GestureState.DRAGGING, GestureState.CLICK_DOWN):
                self._release_all()
            if self._state != GestureState.IDLE:
                self._state = GestureState.IDLE
            self._is_pinching = False
            self._scroll_prev_y = None
            self._scroll_confirm_frames = 0

        return GestureInfo(
            state=self._state,
            index_tip_px=self._last_index_tip_px,
            pinch_distance=self._last_pinch_dist,
            control_active=self._control_active,
        )

    def _move_cursor(self, screen_pos: tuple[int, int]) -> None:
        """Move the mouse cursor to screen position."""
        if self._control_active:
            self._mouse.position = screen_pos

    def _enter_click_down(self, screen_pos: tuple[int, int]) -> None:
        """Transition to CLICK_DOWN state."""
        self._state = GestureState.CLICK_DOWN
        self._pinch_start_time = time.monotonic()
        self._pinch_start_screen = screen_pos
        self._is_pinching = True

    def _fire_click(self) -> None:
        """Execute a mouse click."""
        if self._control_active:
            elapsed = time.monotonic() - self._pinch_start_time
            if elapsed <= config.CONTROL_CLICK_MAX_DURATION_S:
                self._mouse.click(Button.left, 1)
                logger.debug("Click fired")
        self._is_pinching = False

    def _enter_dragging(self, screen_pos: tuple[int, int]) -> None:
        """Transition to DRAGGING state."""
        self._state = GestureState.DRAGGING
        if self._control_active:
            self._mouse.position = screen_pos
            self._mouse.press(Button.left)
            logger.debug("Drag started")

    def _release_drag(self) -> None:
        """Release drag."""
        if self._control_active:
            self._mouse.release(Button.left)
            logger.debug("Drag released")
        self._is_pinching = False

    def _enter_scrolling(self, landmarks: list[tuple[int, int]]) -> None:
        """Transition to SCROLLING state."""
        self._state = GestureState.SCROLLING
        self._scroll_prev_y = self._detector.get_scroll_center(landmarks)[1]

    def _do_scroll(self, landmarks: list[tuple[int, int]]) -> None:
        """Execute scroll based on vertical movement."""
        center = self._detector.get_scroll_center(landmarks)
        if self._scroll_prev_y is not None and self._control_active:
            delta_y = self._scroll_prev_y - center[1]  # 上移動=正=上スクロール
            if abs(delta_y) > 3:  # ノイズフィルタ
                scroll_amount = int(delta_y * config.CONTROL_SCROLL_SENSITIVITY / 30.0)
                if scroll_amount != 0:
                    self._mouse.scroll(0, scroll_amount)
        self._scroll_prev_y = center[1]

    def _release_all(self) -> None:
        """Release any held mouse buttons."""
        if self._control_active:
            try:
                self._mouse.release(Button.left)
            except Exception:  # noqa: BLE001
                pass
