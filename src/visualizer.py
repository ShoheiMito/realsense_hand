"""Hand drawing and control overlay (Main Thread).

Provides :class:`HandVisualizer` for composing hand landmark overlays
and gesture control status onto BGR frames.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from src import config
from src.processor import HandResult

logger = logging.getLogger(__name__)


# MediaPipe Hand landmark connections (21 landmarks per hand)
HAND_CONNECTIONS: list[tuple[int, int]] = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index finger
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle finger
    (0, 9), (9, 10), (10, 11), (11, 12),
    # Ring finger
    (0, 13), (13, 14), (14, 15), (15, 16),
    # Pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
    # Palm
    (5, 9), (9, 13), (13, 17),
]

# Hand colour per handedness (BGR)
_HAND_COLORS: dict[str, tuple[int, int, int]] = {
    "Left": (255, 128, 0),   # 青系
    "Right": (0, 200, 128),  # 緑系
}

# Gesture state → overlay colour (BGR)
_GESTURE_COLORS: dict[str, tuple[int, int, int]] = {
    "idle": (128, 128, 128),      # gray
    "neutral": (0, 255, 255),     # yellow — 手は見えているがカーソル固定
    "cursor": (0, 255, 0),        # green
    "click_down": (0, 0, 255),    # red
    "dragging": (0, 165, 255),    # orange
    "scrolling": (255, 200, 0),   # cyan-ish
}


class HandVisualizer:
    """Hand landmark drawing and control overlay.

    Each public method accepts a BGR frame and returns a new BGR frame with
    the requested overlay applied, leaving the original unmodified.
    """

    def __init__(self) -> None:
        """Initialise drawing settings."""
        self._thickness: int = config.SKELETON_LINE_THICKNESS
        self._radius: int = config.SKELETON_CIRCLE_RADIUS

    def draw_hands(
        self,
        frame: np.ndarray,
        hands: list[HandResult],
    ) -> np.ndarray:
        """Draw hand joint connections for detected hands.

        Args:
            frame: Source BGR image (H, W, 3) uint8.
            hands: List of detected hand results.

        Returns:
            New BGR image with the hand overlay.
        """
        canvas = frame.copy()
        for hand in hands:
            color = _HAND_COLORS.get(hand.handedness, (200, 200, 200))
            lm = hand.landmarks_2d
            # 接続線
            for start, end in HAND_CONNECTIONS:
                if start < len(lm) and end < len(lm):
                    cv2.line(canvas, lm[start], lm[end], color, 2, cv2.LINE_AA)
            # 関節点
            for x, y in lm:
                cv2.circle(canvas, (x, y), 3, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(canvas, (x, y), 3, color, 1, cv2.LINE_AA)
        return canvas

    def draw_fps(
        self,
        frame: np.ndarray,
        fps: float,
    ) -> np.ndarray:
        """Overlay a green FPS counter in the top-left corner.

        Args:
            frame: Source BGR image (H, W, 3) uint8.
            fps: Frames per second value to display.

        Returns:
            New BGR image with the FPS overlay.
        """
        canvas = frame.copy()
        label = f"FPS: {fps:.1f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.7
        thick = 2
        cv2.putText(
            canvas, label, (11, 25), font, scale, (0, 0, 0), thick + 1, cv2.LINE_AA
        )
        cv2.putText(canvas, label, (10, 24), font, scale, (0, 255, 0), thick, cv2.LINE_AA)
        return canvas

    def draw_control_overlay(
        self,
        frame: np.ndarray,
        gesture_state: str,
        control_active: bool,
        index_tip_px: tuple[int, int] | None = None,
        pinch_distance: float = 0.0,
    ) -> np.ndarray:
        """Draw gesture control status overlay.

        Args:
            frame: Source BGR image (H, W, 3) uint8.
            gesture_state: Current gesture state name.
            control_active: Whether mouse control is active.
            index_tip_px: Index finger tip pixel position for indicator.
            pinch_distance: Current pinch distance in pixels.

        Returns:
            New BGR image with the control overlay.
        """
        canvas = frame.copy()
        h, w = canvas.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX

        # "CONTROL MODE" or "CONTROL OFF" banner at top-center
        if control_active:
            mode_label = "CONTROL MODE"
            mode_color = (0, 255, 0)
        else:
            mode_label = "CONTROL OFF"
            mode_color = (0, 0, 200)

        scale = 0.6
        thick = 2
        (tw, th), _ = cv2.getTextSize(mode_label, font, scale, thick)
        x = (w - tw) // 2
        y = th + 8
        cv2.putText(canvas, mode_label, (x + 1, y + 1), font, scale, (0, 0, 0), thick + 1, cv2.LINE_AA)
        cv2.putText(canvas, mode_label, (x, y), font, scale, mode_color, thick, cv2.LINE_AA)

        # ジェスチャー状態テキスト（右下）
        state_color = _GESTURE_COLORS.get(gesture_state, (200, 200, 200))
        state_label = gesture_state.upper()
        scale_s = 0.5
        thick_s = 1
        (tw_s, th_s), _ = cv2.getTextSize(state_label, font, scale_s, thick_s)
        sx = w - tw_s - 10
        sy = h - 10
        cv2.putText(canvas, state_label, (sx + 1, sy + 1), font, scale_s, (0, 0, 0), thick_s + 1, cv2.LINE_AA)
        cv2.putText(canvas, state_label, (sx, sy), font, scale_s, state_color, thick_s, cv2.LINE_AA)

        # 人差し指位置にカラーサークル
        if index_tip_px is not None and control_active:
            cv2.circle(canvas, index_tip_px, 10, state_color, 2, cv2.LINE_AA)
            cv2.circle(canvas, index_tip_px, 3, state_color, -1, cv2.LINE_AA)

        return canvas
