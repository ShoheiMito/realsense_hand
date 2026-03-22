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
from dataclasses import dataclass, field

from src import config
from src.processor import HandResult

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
    """Maps camera pixel coordinates to screen coordinates with smoothing."""

    def __init__(
        self,
        camera_width: int = config.CAMERA_WIDTH,
        camera_height: int = config.CAMERA_HEIGHT,
        active_region: float = config.CONTROL_ACTIVE_REGION,
        smoothing_alpha: float = config.CONTROL_SMOOTHING_ALPHA,
        deadzone_px: int = config.CONTROL_DEADZONE_PX,
        mirror_x: bool = config.CONTROL_MIRROR_X,
    ) -> None:
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.active_region = active_region
        self.smoothing_alpha = smoothing_alpha
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

        # EMA state
        self._prev_sx: float = self.screen_width / 2.0
        self._prev_sy: float = self.screen_height / 2.0
        self._initialized: bool = False

        # Pre-compute active region bounds
        margin_x = (1.0 - active_region) / 2.0 * camera_width
        margin_y = (1.0 - active_region) / 2.0 * camera_height
        self._min_x = margin_x
        self._max_x = camera_width - margin_x
        self._min_y = margin_y
        self._max_y = camera_height - margin_y

    def map(self, cam_x: int, cam_y: int) -> tuple[int, int]:
        """Convert camera pixel position to smoothed screen coordinates.

        Args:
            cam_x: X pixel in camera frame.
            cam_y: Y pixel in camera frame.

        Returns:
            (screen_x, screen_y) coordinates.
        """
        # Clamp to active region
        cx = max(self._min_x, min(self._max_x, float(cam_x)))
        cy = max(self._min_y, min(self._max_y, float(cam_y)))

        # Normalize to [0, 1]
        norm_x = (cx - self._min_x) / (self._max_x - self._min_x)
        norm_y = (cy - self._min_y) / (self._max_y - self._min_y)

        # Mirror X axis (camera is mirrored)
        if self.mirror_x:
            norm_x = 1.0 - norm_x

        # Scale to screen
        raw_sx = norm_x * self.screen_width
        raw_sy = norm_y * self.screen_height

        # EMA smoothing
        if not self._initialized:
            self._prev_sx = raw_sx
            self._prev_sy = raw_sy
            self._initialized = True

        alpha = self.smoothing_alpha
        sx = alpha * raw_sx + (1.0 - alpha) * self._prev_sx
        sy = alpha * raw_sy + (1.0 - alpha) * self._prev_sy

        # Dead zone — don't move if delta is tiny
        dx = abs(sx - self._prev_sx)
        dy = abs(sy - self._prev_sy)
        if dx < self.deadzone_px and dy < self.deadzone_px:
            return (int(self._prev_sx), int(self._prev_sy))

        self._prev_sx = sx
        self._prev_sy = sy

        # Clamp to screen bounds
        final_x = max(0, min(self.screen_width - 1, int(sx)))
        final_y = max(0, min(self.screen_height - 1, int(sy)))
        return (final_x, final_y)

    def reset(self) -> None:
        """Reset EMA state."""
        self._initialized = False
        self._prev_sx = self.screen_width / 2.0
        self._prev_sy = self.screen_height / 2.0


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
        pinch_dist = self._detector.pinch_distance(landmarks)
        is_scroll = self._detector.is_scroll_pose(landmarks)

        self._last_index_tip_px = index_tip
        self._last_pinch_dist = pinch_dist

        # Check pinch with hysteresis
        if self._is_pinching:
            pinching = pinch_dist < config.CONTROL_PINCH_RELEASE_THRESHOLD_PX
        else:
            pinching = pinch_dist < config.CONTROL_PINCH_THRESHOLD_PX

        # Debounce hand detection
        if self._state == GestureState.IDLE:
            if self._hand_detected_frames >= config.CONTROL_GESTURE_CONFIRM_FRAMES:
                self._state = GestureState.CURSOR
                self._mapper.reset()
            return GestureInfo(
                state=self._state,
                index_tip_px=index_tip,
                pinch_distance=pinch_dist,
                control_active=self._control_active,
            )

        # Map cursor position
        screen_pos = self._mapper.map(index_tip[0], index_tip[1])

        # State transitions
        if self._state == GestureState.CURSOR:
            if pinching:
                self._enter_click_down(screen_pos)
            elif is_scroll:
                self._scroll_confirm_frames += 1
                if self._scroll_confirm_frames >= config.CONTROL_GESTURE_CONFIRM_FRAMES:
                    self._enter_scrolling(landmarks)
            else:
                self._scroll_confirm_frames = 0
                self._move_cursor(screen_pos)

        elif self._state == GestureState.CLICK_DOWN:
            if not pinching:
                # Pinch released — fire click
                self._fire_click()
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
            if not pinching:
                self._release_drag()
                self._state = GestureState.CURSOR
            else:
                self._move_cursor(screen_pos)

        elif self._state == GestureState.SCROLLING:
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
            pinch_distance=pinch_dist,
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
