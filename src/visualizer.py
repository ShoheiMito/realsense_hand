"""Skeleton drawing and display (Thread 3 / Main Thread).

Provides :class:`PoseVisualizer` for composing OpenPose-style overlays onto
BGR frames, and :func:`run_visualizer` for the main-thread display loop.
"""

from __future__ import annotations

import colorsys
import logging
import queue
import threading

import cv2
import numpy as np

from src import config
from src.expression import ExpressionResult
from src.processor import FeatureFlags, HandResult, PoseKeypoint3D, ProcessingResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Skeleton topology
# ---------------------------------------------------------------------------

# MediaPipe Pose landmark connections (OpenPose-style grouping).
# Index pairs correspond to the 33 MediaPipe Pose landmark indices.
POSE_CONNECTIONS: list[tuple[int, int]] = [
    # Head (right side)
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 7),
    # Head (left side)
    (0, 4),
    (4, 5),
    (5, 6),
    (6, 8),
    # Mouth
    (9, 10),
    # Shoulders
    (11, 12),
    # Left arm
    (11, 13),
    (13, 15),
    (15, 17),
    (17, 19),
    (19, 15),
    # Right arm
    (12, 14),
    (14, 16),
    (16, 18),
    (18, 20),
    (20, 16),
    # Torso
    (11, 23),
    (12, 24),
    (23, 24),
    # Left leg
    (23, 25),
    (25, 27),
    (27, 29),
    (29, 31),
    (31, 27),
    # Right leg
    (24, 26),
    (26, 28),
    (28, 30),
    (30, 32),
    (32, 28),
]

# Emotion → overlay color (BGR)
_EMOTION_COLORS: dict[str, tuple[int, int, int]] = {
    "happy": (0, 220, 255),  # gold
    "surprise": (0, 140, 255),  # orange
    "angry": (0, 0, 220),  # red
    "sad": (180, 60, 60),  # blue-indigo
    "neutral": (200, 200, 200),  # light gray
}

# Landmark indices shown in the 3D debug overlay → short display names
_DEBUG_KEYPOINTS: dict[int, str] = {
    11: "L.Sho",
    12: "R.Sho",
    15: "L.Wri",
    16: "R.Wri",
    23: "L.Hip",
    24: "R.Hip",
}

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


# ---------------------------------------------------------------------------
# Landmark colour palette
# ---------------------------------------------------------------------------


def _build_landmark_colors(n: int = 33) -> list[tuple[int, int, int]]:
    """Generate N visually distinct BGR colors via golden-angle HSV distribution.

    Uses the golden angle (≈137.5°) to maximise perceptual distance between
    adjacent landmark indices.

    Args:
        n: Number of colors to generate (one per landmark).

    Returns:
        List of (B, G, R) tuples with values in [0, 255].
    """
    colors: list[tuple[int, int, int]] = []
    golden_angle = 137.508 / 360.0
    for i in range(n):
        hue = (i * golden_angle) % 1.0
        r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 1.0)
        colors.append((int(b * 255), int(g * 255), int(r * 255)))
    return colors


# Module-level palette (computed once at import time)
_LANDMARK_COLORS: list[tuple[int, int, int]] = _build_landmark_colors()


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------


def _lerp_color(
    c1: tuple[int, int, int],
    c2: tuple[int, int, int],
    t: float,
) -> tuple[int, int, int]:
    """Linearly interpolate two BGR colours.

    Args:
        c1: Start colour (B, G, R).
        c2: End colour (B, G, R).
        t: Blend factor in [0.0, 1.0].

    Returns:
        Interpolated (B, G, R) colour.
    """
    return (
        round(c1[0] + t * (c2[0] - c1[0])),
        round(c1[1] + t * (c2[1] - c1[1])),
        round(c1[2] + t * (c2[2] - c1[2])),
    )


def _draw_gradient_line(
    canvas: np.ndarray,
    pt1: tuple[int, int],
    pt2: tuple[int, int],
    color1: tuple[int, int, int],
    color2: tuple[int, int, int],
    thickness: int,
) -> None:
    """Draw an anti-aliased line with a smooth colour gradient.

    The number of segments is proportional to the pixel length of the line
    (clamped to [2, 20]) so that short segments are fast and long ones look
    smooth.

    Args:
        canvas: BGR image modified in place.
        pt1: Start point (x, y).
        pt2: End point (x, y).
        color1: Colour at pt1 (B, G, R).
        color2: Colour at pt2 (B, G, R).
        thickness: Line thickness in pixels.
    """
    x1, y1 = pt1
    x2, y2 = pt2

    pixel_dist = int(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)
    n = min(max(pixel_dist // 5, 2), 20)

    for i in range(n):
        t_start = i / n
        t_end = (i + 1) / n
        seg_pt1 = (round(x1 + t_start * (x2 - x1)), round(y1 + t_start * (y2 - y1)))
        seg_pt2 = (round(x1 + t_end * (x2 - x1)), round(y1 + t_end * (y2 - y1)))
        mid_t = (t_start + t_end) * 0.5
        color = _lerp_color(color1, color2, mid_t)
        cv2.line(canvas, seg_pt1, seg_pt2, color, thickness, lineType=cv2.LINE_AA)


# ---------------------------------------------------------------------------
# PoseVisualizer
# ---------------------------------------------------------------------------


class PoseVisualizer:
    """OpenPose-style skeleton drawing and HUD overlays.

    Each public method accepts a BGR frame and returns a new BGR frame with
    the requested overlay applied, leaving the original unmodified.
    """

    def __init__(self) -> None:
        """Initialise drawing settings from :mod:`src.config`."""
        self._thickness: int = config.SKELETON_LINE_THICKNESS
        self._radius: int = config.SKELETON_CIRCLE_RADIUS

    # ------------------------------------------------------------------
    # Public drawing methods
    # ------------------------------------------------------------------

    def draw_skeleton(
        self,
        frame: np.ndarray,
        keypoints_2d: list[tuple[int, int, float]],
        conf_thresh: float = 0.5,
    ) -> np.ndarray:
        """Draw an OpenPose-style colourful skeleton onto *frame*.

        Each limb connection is rendered as a gradient line whose colour
        transitions between the assigned landmark colours of its two endpoints.
        Keypoints are drawn as white-filled circles with a black outline.
        Connections and keypoints whose visibility falls below *conf_thresh*
        are silently skipped.

        Args:
            frame: Source BGR image (H, W, 3) uint8.
            keypoints_2d: Sequence of ``(x, y, visibility)`` for each of the
                33 MediaPipe Pose landmarks. *x* and *y* are pixel coordinates;
                *visibility* is a float in [0.0, 1.0].
            conf_thresh: Minimum visibility to include a keypoint or connection.

        Returns:
            New BGR image with the skeleton overlay.
        """
        canvas = frame.copy()
        n = len(keypoints_2d)

        # --- Limb connections with gradient colour ---
        for start_idx, end_idx in POSE_CONNECTIONS:
            if start_idx >= n or end_idx >= n:
                continue
            sx, sy, sc = keypoints_2d[start_idx]
            ex, ey, ec = keypoints_2d[end_idx]
            if sc < conf_thresh or ec < conf_thresh:
                continue
            color1 = _LANDMARK_COLORS[start_idx % len(_LANDMARK_COLORS)]
            color2 = _LANDMARK_COLORS[end_idx % len(_LANDMARK_COLORS)]
            _draw_gradient_line(
                canvas,
                (sx, sy),
                (ex, ey),
                color1,
                color2,
                self._thickness,
            )

        # --- Keypoint circles (white fill, black border) ---
        for x, y, conf in keypoints_2d:
            if conf < conf_thresh:
                continue
            cv2.circle(
                canvas, (x, y), self._radius, (255, 255, 255), -1, lineType=cv2.LINE_AA
            )
            cv2.circle(
                canvas, (x, y), self._radius, (0, 0, 0), 1, lineType=cv2.LINE_AA
            )

        return canvas

    def draw_expression(
        self,
        frame: np.ndarray,
        expression_result: ExpressionResult | None,
    ) -> np.ndarray:
        """Overlay the emotion label and confidence in the top-right corner.

        The text colour matches the detected emotion.  When *expression_result*
        is ``None`` a grey "No face" placeholder is shown instead.

        Args:
            frame: Source BGR image (H, W, 3) uint8.
            expression_result: Result from :class:`~src.expression.ExpressionRecognizer`,
                or ``None`` when no face is detected.

        Returns:
            New BGR image with the emotion overlay.
        """
        canvas = frame.copy()
        w = canvas.shape[1]

        if expression_result is None:
            label = "No face"
            color: tuple[int, int, int] = (160, 160, 160)
        else:
            emotion = expression_result.emotion
            conf = expression_result.confidence
            label = f"{emotion.upper()}  {conf:.0%}"
            color = _EMOTION_COLORS.get(emotion, (200, 200, 200))

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.8
        thick = 2
        (tw, th), _ = cv2.getTextSize(label, font, scale, thick)
        x = w - tw - 12
        y = th + 12

        # Drop shadow for readability
        cv2.putText(
            canvas, label, (x + 1, y + 1), font, scale, (0, 0, 0), thick + 1, cv2.LINE_AA
        )
        cv2.putText(canvas, label, (x, y), font, scale, color, thick, cv2.LINE_AA)

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
        # Drop shadow
        cv2.putText(
            canvas, label, (11, 25), font, scale, (0, 0, 0), thick + 1, cv2.LINE_AA
        )
        cv2.putText(canvas, label, (10, 24), font, scale, (0, 255, 0), thick, cv2.LINE_AA)
        return canvas

    def draw_3d_info(
        self,
        frame: np.ndarray,
        keypoints_3d: list[PoseKeypoint3D],
    ) -> np.ndarray:
        """Overlay 3D world coordinates for key landmarks (debug view).

        Displays ``(x, y, z)`` in metres for the shoulders, wrists, and hips
        in the bottom-left corner.  Landmarks with zero depth (depth
        unavailable) are shown as ``---``.

        Args:
            frame: Source BGR image (H, W, 3) uint8.
            keypoints_3d: One :class:`~src.processor.PoseKeypoint3D` per
                MediaPipe Pose landmark (33 entries).  Landmarks where depth
                was missing carry ``z == 0.0`` and low visibility.

        Returns:
            New BGR image with the 3D coordinate overlay.
        """
        canvas = frame.copy()
        h = canvas.shape[0]
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.45
        thick = 1
        line_h = 18
        n_lines = len(_DEBUG_KEYPOINTS)
        y_start = h - n_lines * line_h - 8

        for row, (idx, name) in enumerate(_DEBUG_KEYPOINTS.items()):
            y = y_start + row * line_h
            if idx >= len(keypoints_3d) or keypoints_3d[idx].z == 0.0:
                text = f"{name}: ---"
                text_color: tuple[int, int, int] = (120, 120, 120)
            else:
                kp = keypoints_3d[idx]
                text = f"{name}: ({kp.x:+.2f}, {kp.y:+.2f}, {kp.z:+.2f})m"
                text_color = (200, 230, 200)

            cv2.putText(
                canvas, text, (9, y + 1), font, scale, (0, 0, 0), thick + 1, cv2.LINE_AA
            )
            cv2.putText(canvas, text, (8, y), font, scale, text_color, thick, cv2.LINE_AA)

        return canvas

    def draw_hands(
        self,
        frame: np.ndarray,
        hands: list[HandResult],
    ) -> np.ndarray:
        """Draw hand joint connections for detected hands.

        Each hand is drawn in a distinct colour based on handedness
        (left = blue-ish, right = green-ish).

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

    def draw_feature_status(
        self,
        frame: np.ndarray,
        feature_flags: FeatureFlags,
    ) -> np.ndarray:
        """Draw feature toggle status bar at the bottom-right corner.

        Shows ``P:ON  H:OFF  F:ON`` style indicators for pose, hand,
        and face/expression features.

        Args:
            frame: Source BGR image (H, W, 3) uint8.
            feature_flags: Current feature toggle state.

        Returns:
            New BGR image with the status overlay.
        """
        canvas = frame.copy()
        h, w = canvas.shape[:2]
        labels = [
            ("P", feature_flags.pose.is_set()),
            ("H", feature_flags.hand.is_set()),
            ("F", feature_flags.expression.is_set()),
        ]
        text = "  ".join(f"{key}:{'ON' if on else 'OFF'}" for key, on in labels)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.5
        thick = 1
        (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
        x = w - tw - 10
        y = h - 10
        # Drop shadow
        cv2.putText(canvas, text, (x + 1, y + 1), font, scale, (0, 0, 0), thick + 1, cv2.LINE_AA)
        cv2.putText(canvas, text, (x, y), font, scale, (200, 200, 200), thick, cv2.LINE_AA)
        return canvas

    # ------------------------------------------------------------------
    # Composite convenience method
    # ------------------------------------------------------------------

    def draw(self, result: ProcessingResult) -> np.ndarray:
        """Compose all overlays from a :class:`~src.processor.ProcessingResult`.

        Applies skeleton → 3D info → expression → FPS in that order so that
        the FPS counter always appears on top.

        Args:
            result: Full processing result from the processor thread.

        Returns:
            Fully annotated BGR image ready for ``cv2.imshow``.
        """
        img = result.color_image.copy()

        if result.landmarks_2d is not None:
            # Attach per-landmark visibility from keypoints_3d when available.
            if result.keypoints_3d is not None:
                kpts: list[tuple[int, int, float]] = [
                    (x, y, kp.visibility)
                    for (x, y), kp in zip(result.landmarks_2d, result.keypoints_3d)
                ]
            else:
                kpts = [(x, y, 1.0) for x, y in result.landmarks_2d]
            img = self.draw_skeleton(img, kpts)

        if result.keypoints_3d is not None:
            img = self.draw_3d_info(img, result.keypoints_3d)

        img = self.draw_expression(img, result.expression)
        img = self.draw_fps(img, result.processing_fps)

        return img


# ---------------------------------------------------------------------------
# Main-thread visualisation loop
# ---------------------------------------------------------------------------


def run_visualizer(
    result_queue: queue.Queue[ProcessingResult],
    stop_event: threading.Event,
) -> None:
    """Main-thread visualisation loop.

    ``cv2.imshow`` and ``cv2.waitKey`` must be called from the main thread on
    most platforms, so this function is invoked directly from ``main.py``.

    Pressing **q** sets *stop_event* to trigger a graceful shutdown of all
    threads.

    Args:
        result_queue: Queue of :class:`~src.processor.ProcessingResult` items
            produced by the processor thread (``maxsize=2``).
        stop_event: Shared event; set externally to stop the loop, or set
            internally when the user presses **q**.
    """
    visualizer = PoseVisualizer()
    logger.info("Visualizer started. Press 'q' to quit.")

    while not stop_event.is_set():
        try:
            result = result_queue.get(timeout=0.1)
        except queue.Empty:
            # Keep checking stop_event while waiting for the next frame.
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                logger.info("'q' pressed — stopping.")
                stop_event.set()
                break
            continue

        img = visualizer.draw(result)
        cv2.imshow(config.WINDOW_NAME, img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            logger.info("'q' pressed — stopping.")
            stop_event.set()
            break

    cv2.destroyAllWindows()
    logger.info("Visualizer stopped.")
