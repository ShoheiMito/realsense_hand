"""Hand tracking processing (Thread 2).

Consumes FrameData from the camera queue, runs MediaPipe HandLandmarker,
reprojects 2D keypoints to 3D world coordinates using the RealSense depth
stream, and applies One Euro temporal smoothing.
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
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class PoseKeypoint3D:
    """A single 3D keypoint in world coordinates.

    Attributes:
        x: X coordinate in metres (positive = right).
        y: Y coordinate in metres (positive = down).
        z: Z coordinate in metres (positive = away from camera).
        visibility: MediaPipe visibility score in [0.0, 1.0].
        name: Landmark name (e.g. 'index_finger_tip').
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
    """Data passed from the processor thread to the main thread.

    Attributes:
        color_image: BGR image (H, W, 3) uint8 used for rendering.
        hands: List of detected hand results, or None if no hands were found.
        processing_fps: Frames-per-second measured by the processor thread.
        timestamp: time.monotonic() value of the originating camera frame.
    """

    color_image: np.ndarray
    hands: list[HandResult] | None
    processing_fps: float
    timestamp: float
    timings: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# HandProcessor
# ---------------------------------------------------------------------------


class HandProcessor:
    """MediaPipe HandLandmarker + RealSense depth integration.

    Designed to be instantiated once inside the processing thread and called
    once per frame via process_frame().
    """

    def __init__(self) -> None:
        """Initialize processor.

        Raises:
            RuntimeError: If mediapipe is not installed.
        """
        if not _MEDIAPIPE_AVAILABLE:
            raise RuntimeError("mediapipe is not installed")

        self._hand_landmarker: mp_vision.HandLandmarker | None = None  # type: ignore[name-defined]
        self._hand_smoothers: dict[str, KeypointSmoother] = {}

        self._frame_count: int = 0
        self._last_ts_ms: int = 0
        self._last_hands: list[HandResult] | None = None

        # FPS tracking
        self._processing_fps: float = 0.0
        self._last_fps_time: float = time.monotonic()
        self._fps_frame_count: int = 0

        # Timing accumulation for periodic reports
        self._timing_buf: list[dict[str, float]] = []
        self._REPORT_INTERVAL = 100

        self._ensure_hand_landmarker()

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

    def _get_hand_smoother(self, handedness: str) -> KeypointSmoother:
        """Get or create a KeypointSmoother for a specific hand."""
        if handedness not in self._hand_smoothers:
            self._hand_smoothers[handedness] = KeypointSmoother(
                num_keypoints=len(HAND_LANDMARK_NAMES)
            )
        return self._hand_smoothers[handedness]

    def process_frame(self, frame_data: FrameData) -> ProcessingResult:
        """Process one camera frame end-to-end.

        Args:
            frame_data: Frame data produced by the camera thread.

        Returns:
            ProcessingResult ready for the main thread.
        """
        self._frame_count += 1

        # ---- Common: BGR → RGB + MediaPipe image -------------------------
        rgb_image: np.ndarray = frame_data.color_image[:, :, ::-1]
        rgb_image.flags.writeable = False
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        ts_ms = max(int(frame_data.timestamp * 1000), self._last_ts_ms + 1)
        self._last_ts_ms = ts_ms

        rgb_image.flags.writeable = True

        h, w = frame_data.color_image.shape[:2]

        # ---- Hand detection -----------------------------------------------
        hands: list[HandResult] | None = None
        t_hand_ms = 0.0

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
            hands = self._last_hands

        # ---- FPS tracking ------------------------------------------------
        self._fps_frame_count += 1
        now = time.monotonic()
        elapsed = now - self._last_fps_time
        if elapsed >= 1.0:
            self._processing_fps = self._fps_frame_count / elapsed
            self._fps_frame_count = 0
            self._last_fps_time = now

        # ---- Timing report -----------------------------------------------
        timings: dict[str, float] = {
            **frame_data.timings,
            "hand_inference": t_hand_ms,
        }
        self._timing_buf.append(timings)
        if len(self._timing_buf) >= self._REPORT_INTERVAL:
            self._print_timing_report()
            self._timing_buf.clear()

        return ProcessingResult(
            color_image=frame_data.color_image,
            hands=hands,
            processing_fps=self._processing_fps,
            timestamp=frame_data.timestamp,
            timings=timings,
        )

    def _print_timing_report(self) -> None:
        """Print a per-step timing table for the last N frames."""
        buf = self._timing_buf
        n = len(buf)
        budget_ms = 1000.0 / 30.0

        steps: list[tuple[str, str]] = [
            ("capture_align", "1. Capture + Align   "),
            ("depth_filter",  "2. Depth Filtering   "),
            ("hand_inference", "3. Hand Detection    "),
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
        if self._hand_landmarker is not None:
            try:
                self._hand_landmarker.close()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Error closing HandLandmarker: %s", exc)
        logger.info("HandProcessor closed.")


# ---------------------------------------------------------------------------
# Thread entry point
# ---------------------------------------------------------------------------


def processing_thread(
    frame_queue: queue.Queue[FrameData],
    result_queue: queue.Queue[ProcessingResult],
    stop_event: threading.Event,
) -> None:
    """Entry point for processing Thread 2.

    Args:
        frame_queue: Incoming frames from the camera thread.
        result_queue: Outgoing results for the main thread.
        stop_event: Set this event to signal a graceful shutdown.
    """
    try:
        processor = HandProcessor()
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to initialize HandProcessor: %s", exc)
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

            # Discard stale result if the queue is full
            if result_queue.full():
                try:
                    result_queue.get_nowait()
                except queue.Empty:
                    pass

            try:
                result_queue.put_nowait(result)
            except queue.Full:
                pass
    finally:
        processor.close()
