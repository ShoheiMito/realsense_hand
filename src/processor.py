"""Pose estimation and integration processing (Thread 2).

Consumes FrameData from the camera queue, runs MediaPipe PoseLandmarker,
reprojects 2D keypoints to 3D world coordinates using the RealSense depth
stream, applies One Euro temporal smoothing, and runs expression recognition
every EXPRESSION_SKIP_FRAMES frames.
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
class ProcessingResult:
    """Data passed from the processor thread to the visualizer thread.

    Attributes:
        color_image: BGR image (H, W, 3) uint8 used for rendering.
        landmarks_2d: Pixel-space (x, y) for all 33 pose landmarks,
            or None if no pose was detected in this frame.
        keypoints_3d: 3D world coordinates for each landmark
            (zero coords with low visibility when depth is missing),
            or None if no pose detected.
        expression: Latest expression result (may be cached from a prior
            frame when expression recognition was skipped).
        processing_fps: Frames-per-second measured by the processor thread.
        timestamp: time.monotonic() value of the originating camera frame.
    """

    color_image: np.ndarray
    landmarks_2d: list[tuple[int, int]] | None
    keypoints_3d: list[PoseKeypoint3D] | None
    expression: ExpressionResult | None
    processing_fps: float
    timestamp: float
    timings: dict[str, float] = field(default_factory=dict)  # per-step ms


# ---------------------------------------------------------------------------
# PoseProcessor
# ---------------------------------------------------------------------------


class PoseProcessor:
    """Integrates MediaPipe Pose, RealSense depth, and expression recognition.

    Designed to be instantiated once inside the processing thread and called
    once per frame via process_frame().
    """

    def __init__(self) -> None:
        """Initialize PoseLandmarker, ExpressionRecognizer, and KeypointSmoother.

        Raises:
            RuntimeError: If mediapipe is not installed.
        """
        if not _MEDIAPIPE_AVAILABLE:
            raise RuntimeError("mediapipe is not installed")

        # MediaPipe PoseLandmarker in VIDEO mode (inter-frame tracking)
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
        logger.info("PoseLandmarker initialized: %s", config.POSE_MODEL_PATH)

        self._expression_recognizer = ExpressionRecognizer(config.FACE_MODEL_PATH)

        # One Euro Filter smoother — 33 keypoints × 3 axes = 99 filter instances
        self._smoother = KeypointSmoother(num_keypoints=len(LANDMARK_NAMES))

        self._frame_count: int = 0
        self._last_ts_ms: int = 0  # last timestamp for VIDEO mode (strict monotonic)
        self._last_expression: ExpressionResult | None = None

        # FPS tracking
        self._processing_fps: float = 0.0
        self._last_fps_time: float = time.monotonic()
        self._fps_frame_count: int = 0

        # Timing accumulation for 100-frame reports
        self._timing_buf: list[dict[str, float]] = []
        self._REPORT_INTERVAL = 100

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_frame(self, frame_data: FrameData) -> ProcessingResult:
        """Process one camera frame end-to-end.

        Steps:
            1. BGR → RGB conversion.
            2. MediaPipe Pose inference (IMAGE mode, synchronous).
            3. Normalised landmarks → pixel coordinates.
            4. Batch 3D deprojection via RealSense depth + median fallback.
            5. Temporal smoothing with One Euro Filter.
            6. Expression recognition every EXPRESSION_SKIP_FRAMES frames
               (previous result reused on skipped frames).
            7. Return ProcessingResult.

        Args:
            frame_data: Frame data produced by the camera thread.

        Returns:
            ProcessingResult ready for the visualizer thread.
        """
        self._frame_count += 1

        # ---- Step 1: BGR → RGB -------------------------------------------
        rgb_image: np.ndarray = frame_data.color_image[:, :, ::-1]
        rgb_image.flags.writeable = False

        # ---- Step 2: MediaPipe Pose inference (VIDEO mode) ----------------
        t_pose_start = time.perf_counter()
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        # Real-time ms, guaranteed strictly increasing for VIDEO mode
        ts_ms = max(int(frame_data.timestamp * 1000), self._last_ts_ms + 1)
        self._last_ts_ms = ts_ms
        pose_result = self._landmarker.detect_for_video(mp_image, ts_ms)
        t_pose_end = time.perf_counter()

        rgb_image.flags.writeable = True

        # ---- Steps 3–5: landmarks → 2D pixels → 3D → smoothing ----------
        landmarks_2d: list[tuple[int, int]] | None = None
        keypoints_3d: list[PoseKeypoint3D] | None = None
        t_deproject_ms = 0.0
        t_smooth_ms = 0.0

        if pose_result.pose_landmarks:
            h, w = frame_data.color_image.shape[:2]
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
            t_dep_end = time.perf_counter()
            t_deproject_ms = (t_dep_end - t_dep_start) * 1000.0

            # Temporal smoothing via One Euro Filter
            t_smooth_start = time.perf_counter()
            points_smooth = self._smoother.smooth(points_raw, frame_data.timestamp)
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

        # ---- Step 6: Expression recognition (throttled) ------------------
        expression = self._last_expression
        t_expr_ms = 0.0
        if self._frame_count % config.EXPRESSION_SKIP_FRAMES == 0:
            try:
                t_expr_start = time.perf_counter()
                new_expr = self._expression_recognizer.analyze(
                    rgb_image, timestamp_ms=ts_ms
                )
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
            "pose_inference": (t_pose_end - t_pose_start) * 1000.0,
            "deproject": t_deproject_ms,
            "smoothing": t_smooth_ms,
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
            ("pose_inference","3. Pose Estimation   "),
            ("deproject",     "4. 3D Deprojection   "),
            ("smoothing",     "5. 1Euro Smoothing   "),
            ("expression",    "6. Expression Recog. "),
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
        try:
            self._landmarker.close()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Error closing PoseLandmarker: %s", exc)
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
    """
    try:
        processor = PoseProcessor()
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
