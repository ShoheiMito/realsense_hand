"""Pose estimation, hand tracking, and expression processing (Thread 2).

Consumes FrameData from the camera queue, runs MediaPipe PoseLandmarker
and/or HandLandmarker, reprojects 2D keypoints to 3D world coordinates
using the RealSense depth stream, applies One Euro temporal smoothing,
and runs expression recognition every EXPRESSION_SKIP_FRAMES frames.

Each recognition feature (pose, hand, expression) can be independently
toggled on/off at runtime via FeatureFlags.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field

from typing import TYPE_CHECKING

import numpy as np

from src import config
from src.depth_utils import KeypointSmoother, batch_deproject
from src.expression import ExpressionRecognizer, ExpressionResult

if TYPE_CHECKING:
    from src.camera import FrameData

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    _MEDIAPIPE_AVAILABLE = True
except ImportError:
    mp = None  # type: ignore[assignment]
    mp_python = None  # type: ignore[assignment]
    mp_vision = None  # type: ignore[assignment]
    _MEDIAPIPE_AVAILABLE = False

logger = logging.getLogger(__name__)

# MediaPipe Pose landmark names — indices 0–32
LANDMARK_NAMES: list[str] = [
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
]

# MediaPipe Hand landmark names — indices 0–20
HAND_LANDMARK_NAMES: list[str] = [
    "wrist",
    "thumb_cmc",
    "thumb_mcp",
    "thumb_ip",
    "thumb_tip",
    "index_finger_mcp",
    "index_finger_pip",
    "index_finger_dip",
    "index_finger_tip",
    "middle_finger_mcp",
    "middle_finger_pip",
    "middle_finger_dip",
    "middle_finger_tip",
    "ring_finger_mcp",
    "ring_finger_pip",
    "ring_finger_dip",
    "ring_finger_tip",
    "pinky_mcp",
    "pinky_pip",
    "pinky_dip",
    "pinky_tip",
]


# ---------------------------------------------------------------------------
# Feature flags
# ---------------------------------------------------------------------------


def _make_set_event() -> threading.Event:
    """Create a threading.Event that starts in the set (enabled) state."""
    e = threading.Event()
    e.set()
    return e


@dataclass
class FeatureFlags:
    """Thread-safe feature toggle flags.

    Each flag is a threading.Event: is_set() = enabled, clear() = disabled.
    The main thread toggles via set()/clear(); the processor thread reads
    via is_set() each frame (lock-free on CPython).

    Attributes:
        pose: Enable/disable PoseLandmarker (body skeleton).
        hand: Enable/disable HandLandmarker (hand joints).
        expression: Enable/disable FaceLandmarker (expression recognition).
    """

    pose: threading.Event = field(default_factory=_make_set_event)
    hand: threading.Event = field(default_factory=_make_set_event)
    expression: threading.Event = field(default_factory=_make_set_event)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class PoseKeypoint3D:
    """A single 3D pose keypoint in world coordinates.

    Attributes:
        x: X coordinate in metres (positive = right).
        y: Y coordinate in metres (positive = down).
        z: Z coordinate in metres (positive = away from camera).
        visibility: MediaPipe visibility score in [0.0, 1.0].
        name: Landmark name (e.g. 'left_shoulder').
    """

    x: float
    y: float
    z: float
    visibility: float
    name: str


@dataclass
class HandResult:
    """Detection result for a single hand.

    Attributes:
        handedness: 'Left' or 'Right'.
        landmarks_2d: Pixel-space (x, y) for all 21 hand landmarks.
        keypoints_3d: 3D world coordinates for each landmark,
            or None if depth is unavailable.
        score: Detection confidence score.
    """

    handedness: str
    landmarks_2d: list[tuple[int, int]]
    keypoints_3d: list[PoseKeypoint3D] | None
    score: float


@dataclass
class ProcessingResult:
    """Data passed from the processor thread to the visualizer thread.

    Attributes:
        color_image: BGR image (H, W, 3) uint8 used for rendering.
        landmarks_2d: Pixel-space (x, y) for all 33 pose landmarks,
            or None if no pose was detected in this frame.
        keypoints_3d: 3D world coordinates for each landmark
            (zero coords with low visibility when depth is missing),
            or None if no pose detected.
        hands: List of detected hand results, or None if hand detection
            is disabled or no hands were found.
        expression: Latest expression result (may be cached from a prior
            frame when expression recognition was skipped).
        processing_fps: Frames-per-second measured by the processor thread.
        timestamp: time.monotonic() value of the originating camera frame.
    """

    color_image: np.ndarray
    landmarks_2d: list[tuple[int, int]] | None
    keypoints_3d: list[PoseKeypoint3D] | None
    hands: list[HandResult] | None
    expression: ExpressionResult | None
    processing_fps: float
    timestamp: float
    timings: dict[str, float] = field(default_factory=dict)  # per-step ms


# ---------------------------------------------------------------------------
# PoseProcessor
# ---------------------------------------------------------------------------


class PoseProcessor:
    """Integrates MediaPipe Pose, Hand, RealSense depth, and expression recognition.

    Designed to be instantiated once inside the processing thread and called
    once per frame via process_frame().  Each recognition feature is lazily
    initialised on first use and gated by FeatureFlags.
    """

    def __init__(self, feature_flags: FeatureFlags | None = None) -> None:
        """Initialize processor with optional feature flags.

        Models are lazily created on first use when their feature flag is
        enabled, rather than always at startup.

        Args:
            feature_flags: Runtime toggle flags.  Defaults to all-enabled.

        Raises:
            RuntimeError: If mediapipe is not installed.
        """
        if not _MEDIAPIPE_AVAILABLE:
            raise RuntimeError("mediapipe is not installed")

        self._flags = feature_flags or FeatureFlags()

        # Lazily initialised models (None until first use)
        self._landmarker: mp_vision.PoseLandmarker | None = None  # type: ignore[name-defined]
        self._hand_landmarker: mp_vision.HandLandmarker | None = None  # type: ignore[name-defined]
        self._expression_recognizer: ExpressionRecognizer | None = None

        # Pose smoother — 33 keypoints
        self._pose_smoother: KeypointSmoother | None = None

        # Hand smoothers — one per handedness
        self._hand_smoothers: dict[str, KeypointSmoother] = {}

        self._frame_count: int = 0
        self._last_ts_ms: int = 0  # last timestamp for VIDEO mode (strict monotonic)
        self._last_expression: ExpressionResult | None = None
        self._last_hands: list[HandResult] | None = None

        # FPS tracking
        self._processing_fps: float = 0.0
        self._last_fps_time: float = time.monotonic()
        self._fps_frame_count: int = 0

        # Timing accumulation for 100-frame reports
        self._timing_buf: list[dict[str, float]] = []
        self._REPORT_INTERVAL = 100

        # Eagerly init models that are enabled at startup
        if self._flags.pose.is_set():
            self._ensure_pose_landmarker()
        if self._flags.hand.is_set():
            self._ensure_hand_landmarker()
        if self._flags.expression.is_set():
            self._ensure_expression_recognizer()

    # ------------------------------------------------------------------
    # Lazy initialisation helpers
    # ------------------------------------------------------------------

    def _ensure_pose_landmarker(self) -> mp_vision.PoseLandmarker:  # type: ignore[name-defined]
        """Lazily create PoseLandmarker on first use."""
        if self._landmarker is None:
            options = mp_vision.PoseLandmarkerOptions(
                base_options=mp_python.BaseOptions(
                    model_asset_path=config.POSE_MODEL_PATH
                ),
                running_mode=mp_vision.RunningMode.VIDEO,
                num_poses=config.POSE_NUM_POSES,
                min_pose_detection_confidence=config.POSE_MIN_DETECTION_CONFIDENCE,
                min_tracking_confidence=config.POSE_MIN_TRACKING_CONFIDENCE,
            )
            self._landmarker = mp_vision.PoseLandmarker.create_from_options(options)
            self._pose_smoother = KeypointSmoother(num_keypoints=len(LANDMARK_NAMES))
            logger.info("PoseLandmarker initialized: %s", config.POSE_MODEL_PATH)
        return self._landmarker

    def _ensure_hand_landmarker(self) -> mp_vision.HandLandmarker:  # type: ignore[name-defined]
        """Lazily create HandLandmarker on first use."""
        if self._hand_landmarker is None:
            options = mp_vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(
                    model_asset_path=config.HAND_MODEL_PATH
                ),
                running_mode=mp_vision.RunningMode.VIDEO,
                num_hands=config.HAND_NUM_HANDS,
                min_hand_detection_confidence=config.HAND_MIN_DETECTION_CONFIDENCE,
                min_tracking_confidence=config.HAND_MIN_TRACKING_CONFIDENCE,
            )
            self._hand_landmarker = mp_vision.HandLandmarker.create_from_options(
                options
            )
            logger.info("HandLandmarker initialized: %s", config.HAND_MODEL_PATH)
        return self._hand_landmarker

    def _ensure_expression_recognizer(self) -> ExpressionRecognizer:
        """Lazily create ExpressionRecognizer on first use."""
        if self._expression_recognizer is None:
            self._expression_recognizer = ExpressionRecognizer(config.FACE_MODEL_PATH)
            logger.info("ExpressionRecognizer initialized: %s", config.FACE_MODEL_PATH)
        return self._expression_recognizer

    def _get_hand_smoother(self, handedness: str) -> KeypointSmoother:
        """Get or create a KeypointSmoother for a specific hand."""
        if handedness not in self._hand_smoothers:
            self._hand_smoothers[handedness] = KeypointSmoother(
                num_keypoints=len(HAND_LANDMARK_NAMES)
            )
        return self._hand_smoothers[handedness]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_frame(self, frame_data: FrameData) -> ProcessingResult:
        """Process one camera frame end-to-end.

        Each recognition block (pose, hand, expression) is gated by the
        corresponding FeatureFlags toggle and skipped when disabled.

        Args:
            frame_data: Frame data produced by the camera thread.

        Returns:
            ProcessingResult ready for the visualizer thread.
        """
        self._frame_count += 1

        # ---- Common: BGR → RGB + MediaPipe image -------------------------
        rgb_image: np.ndarray = frame_data.color_image[:, :, ::-1]
        rgb_image.flags.writeable = False
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        # Real-time ms, guaranteed strictly increasing for VIDEO mode
        ts_ms = max(int(frame_data.timestamp * 1000), self._last_ts_ms + 1)
        self._last_ts_ms = ts_ms

        rgb_image.flags.writeable = True

        h, w = frame_data.color_image.shape[:2]

        # ---- Pose (conditional) ------------------------------------------
        landmarks_2d: list[tuple[int, int]] | None = None
        keypoints_3d: list[PoseKeypoint3D] | None = None
        t_pose_ms = 0.0
        t_deproject_ms = 0.0
        t_smooth_ms = 0.0

        if self._flags.pose.is_set():
            landmarker = self._ensure_pose_landmarker()

            t_pose_start = time.perf_counter()
            pose_result = landmarker.detect_for_video(mp_image, ts_ms)
            t_pose_ms = (time.perf_counter() - t_pose_start) * 1000.0

            if pose_result.pose_landmarks:
                raw_lm = pose_result.pose_landmarks[0]

                landmarks_2d = [
                    (
                        max(0, min(w - 1, int(lm.x * w))),
                        max(0, min(h - 1, int(lm.y * h))),
                    )
                    for lm in raw_lm
                ]

                # Batch 2D → 3D deprojection with depth median fallback
                t_dep_start = time.perf_counter()
                points_raw = batch_deproject(
                    landmarks_2d,
                    frame_data.depth_frame,
                    frame_data.intrinsics,
                )
                t_deproject_ms = (time.perf_counter() - t_dep_start) * 1000.0

                # Temporal smoothing via One Euro Filter
                t_smooth_start = time.perf_counter()
                assert self._pose_smoother is not None
                points_smooth = self._pose_smoother.smooth(
                    points_raw, frame_data.timestamp
                )
                t_smooth_ms = (time.perf_counter() - t_smooth_start) * 1000.0

                keypoints_3d = []
                for i, (lm, pt) in enumerate(zip(raw_lm, points_smooth)):
                    if pt is not None:
                        x, y, z = pt
                    else:
                        x, y, z = 0.0, 0.0, 0.0
                    vis = (
                        float(lm.visibility)
                        if lm.visibility is not None
                        else 0.0
                    )
                    name = (
                        LANDMARK_NAMES[i]
                        if i < len(LANDMARK_NAMES)
                        else f"landmark_{i}"
                    )
                    keypoints_3d.append(
                        PoseKeypoint3D(x=x, y=y, z=z, visibility=vis, name=name)
                    )

        # ---- Hand (conditional, with skip-frames throttle) ---------------
        hands: list[HandResult] | None = None
        t_hand_ms = 0.0

        if self._flags.hand.is_set():
            if self._frame_count % config.HAND_SKIP_FRAMES == 0:
                hand_landmarker = self._ensure_hand_landmarker()

                t_hand_start = time.perf_counter()
                hand_result = hand_landmarker.detect_for_video(mp_image, ts_ms)
                t_hand_ms = (time.perf_counter() - t_hand_start) * 1000.0

                if hand_result.hand_landmarks:
                    hands = []
                    for hand_lms, handedness_list in zip(
                        hand_result.hand_landmarks, hand_result.handedness
                    ):
                        handedness = handedness_list[0].category_name
                        score = float(handedness_list[0].score)

                        lm_2d = [
                            (
                                max(0, min(w - 1, int(lm.x * w))),
                                max(0, min(h - 1, int(lm.y * h))),
                            )
                            for lm in hand_lms
                        ]

                        # 3D deprojection
                        pts_3d_raw = batch_deproject(
                            lm_2d,
                            frame_data.depth_frame,
                            frame_data.intrinsics,
                        )

                        # Temporal smoothing per hand
                        smoother = self._get_hand_smoother(handedness)
                        pts_smooth = smoother.smooth(
                            pts_3d_raw, frame_data.timestamp
                        )

                        kps_3d = []
                        for j, (lm, pt) in enumerate(zip(hand_lms, pts_smooth)):
                            if pt is not None:
                                x, y, z = pt
                            else:
                                x, y, z = 0.0, 0.0, 0.0
                            vis = (
                                float(lm.visibility)
                                if hasattr(lm, "visibility") and lm.visibility is not None
                                else 1.0
                            )
                            name = (
                                HAND_LANDMARK_NAMES[j]
                                if j < len(HAND_LANDMARK_NAMES)
                                else f"hand_{j}"
                            )
                            kps_3d.append(
                                PoseKeypoint3D(
                                    x=x, y=y, z=z, visibility=vis, name=name
                                )
                            )

                        hands.append(
                            HandResult(
                                handedness=handedness,
                                landmarks_2d=lm_2d,
                                keypoints_3d=kps_3d,
                                score=score,
                            )
                        )
                    self._last_hands = hands
                else:
                    self._last_hands = None
            else:
                # スキップフレーム: 前回の結果を再利用
                hands = self._last_hands

        # ---- Expression (conditional, with skip-frames throttle) ---------
        expression: ExpressionResult | None = None
        t_expr_ms = 0.0

        if self._flags.expression.is_set():
            expression = self._last_expression
            if self._frame_count % config.EXPRESSION_SKIP_FRAMES == 0:
                try:
                    recognizer = self._ensure_expression_recognizer()
                    t_expr_start = time.perf_counter()
                    new_expr = recognizer.analyze(rgb_image, timestamp_ms=ts_ms)
                    t_expr_ms = (time.perf_counter() - t_expr_start) * 1000.0
                    expression = new_expr
                    self._last_expression = new_expr
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Expression recognition error: %s", exc)

        # ---- FPS tracking ------------------------------------------------
        self._fps_frame_count += 1
        now = time.monotonic()
        elapsed = now - self._last_fps_time
        if elapsed >= 1.0:
            self._processing_fps = self._fps_frame_count / elapsed
            self._fps_frame_count = 0
            self._last_fps_time = now

        # ---- Timing accumulation and periodic report ---------------------
        timings: dict[str, float] = {
            **frame_data.timings,
            "pose_inference": t_pose_ms,
            "deproject": t_deproject_ms,
            "smoothing": t_smooth_ms,
            "hand_inference": t_hand_ms,
            "expression": t_expr_ms,
        }
        self._timing_buf.append(timings)
        if len(self._timing_buf) >= self._REPORT_INTERVAL:
            self._print_timing_report()
            self._timing_buf.clear()

        return ProcessingResult(
            color_image=frame_data.color_image,
            landmarks_2d=landmarks_2d,
            keypoints_3d=keypoints_3d,
            hands=hands,
            expression=expression,
            processing_fps=self._processing_fps,
            timestamp=frame_data.timestamp,
            timings=timings,
        )

    def _print_timing_report(self) -> None:
        """Print a per-step timing table (avg/min/max) for the last N frames."""
        buf = self._timing_buf
        n = len(buf)
        budget_ms = 1000.0 / 30.0  # 30 fps → 33.3 ms

        steps: list[tuple[str, str]] = [
            ("capture_align", "1. Capture + Align   "),
            ("depth_filter",  "2. Depth Filtering   "),
            ("pose_inference", "3. Pose Estimation   "),
            ("deproject",     "4. 3D Deprojection   "),
            ("smoothing",     "5. 1Euro Smoothing   "),
            ("hand_inference", "6. Hand Detection    "),
            ("expression",    "7. Expression Recog. "),
        ]

        print(  # noqa: T201
            f"\n{'=' * 62}\n"
            f"  Timing Report ({n} frames)  [ms]  budget={budget_ms:.1f}ms/frame\n"
            f"{'─' * 62}\n"
            f"  {'Step':<22} {'Avg':>6} {'Min':>6} {'Max':>6}  {'% Budget':>8}\n"
            f"{'─' * 62}"
        )
        proc_total_avg = 0.0
        for key, label in steps:
            vals = [f[key] for f in buf if key in f and f[key] > 0]
            if not vals:
                print(f"  {label} {'  n/a':>6} {'  n/a':>6} {'  n/a':>6}  {'  n/a':>8}")  # noqa: T201
                continue
            avg = sum(vals) / len(vals)
            mn = min(vals)
            mx = max(vals)
            pct = avg / budget_ms * 100.0
            proc_total_avg += avg
            print(f"  {label} {avg:6.1f} {mn:6.1f} {mx:6.1f}  {pct:7.1f}%")  # noqa: T201
        print(  # noqa: T201
            f"{'─' * 62}\n"
            f"  {'  Subtotal (camera+proc)':<22} {proc_total_avg:6.1f}"
            f"{'':>14}  {proc_total_avg / budget_ms * 100.0:7.1f}%\n"
            f"{'=' * 62}"
        )

    def close(self) -> None:
        """Release all resources held by the processor."""
        if self._landmarker is not None:
            try:
                self._landmarker.close()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Error closing PoseLandmarker: %s", exc)
        if self._hand_landmarker is not None:
            try:
                self._hand_landmarker.close()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Error closing HandLandmarker: %s", exc)
        if self._expression_recognizer is not None:
            try:
                self._expression_recognizer.close()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Error closing ExpressionRecognizer: %s", exc)
        logger.info("PoseProcessor closed.")


# ---------------------------------------------------------------------------
# Thread entry point
# ---------------------------------------------------------------------------


def processing_thread(
    frame_queue: queue.Queue[FrameData],
    result_queue: queue.Queue[ProcessingResult],
    stop_event: threading.Event,
    feature_flags: FeatureFlags | None = None,
) -> None:
    """Entry point for processing Thread 2.

    Reads FrameData from frame_queue, calls PoseProcessor.process_frame(),
    and writes ProcessingResult to result_queue.  Old results are discarded
    (maxsize=2 pattern identical to the camera thread) to prevent latency
    accumulation.

    Args:
        frame_queue: Incoming frames from the camera thread.
        result_queue: Outgoing results for the visualizer thread.
        stop_event: Set this event to signal a graceful shutdown.
        feature_flags: Runtime feature toggle flags.
    """
    try:
        processor = PoseProcessor(feature_flags=feature_flags)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to initialize PoseProcessor: %s", exc)
        stop_event.set()
        return

    try:
        while not stop_event.is_set():
            try:
                frame_data = frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                result = processor.process_frame(frame_data)
            except Exception as exc:  # noqa: BLE001
                logger.error("Frame processing error: %s", exc)
                continue

            # Discard stale result if the visualizer queue is full
            if result_queue.full():
                try:
                    result_queue.get_nowait()
                except queue.Empty:
                    pass

            try:
                result_queue.put_nowait(result)
            except queue.Full:
                pass  # Rare race condition — skip silently
    finally:
        processor.close()
