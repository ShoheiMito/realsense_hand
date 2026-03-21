"""RealSense L515 camera management (Thread 1)."""

import logging
import queue
import threading
import time
from dataclasses import dataclass, field

import numpy as np
import pyrealsense2 as rs

from src import config

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_WARMUP_FRAMES = 30
_FRAME_TIMEOUT_MS = 200


@dataclass
class FrameData:
    """Data passed from camera thread to processing thread."""

    color_image: np.ndarray  # BGR (H, W, 3) uint8
    depth_image: np.ndarray  # (H, W) uint16
    depth_frame: rs.depth_frame  # for get_distance()
    intrinsics: rs.intrinsics  # for deprojection
    timestamp: float  # time.monotonic()
    timings: dict[str, float] = field(default_factory=dict)  # per-step ms


class RealsenseCamera:
    """RealSense L515 camera management and frame acquisition."""

    def __init__(self) -> None:
        """Initialize pipeline, filters, and alignment."""
        self._pipeline = rs.pipeline()
        self._rs_config = rs.config()
        self._align = rs.align(rs.stream.color)

        # Configure streams — color and depth at same resolution (L515 note)
        self._rs_config.enable_stream(
            rs.stream.color,
            config.CAMERA_WIDTH,
            config.CAMERA_HEIGHT,
            rs.format.bgr8,
            config.CAMERA_FPS,
        )
        self._rs_config.enable_stream(
            rs.stream.depth,
            config.CAMERA_WIDTH,
            config.CAMERA_HEIGHT,
            rs.format.z16,
            config.CAMERA_FPS,
        )

        # Create filter objects once and reuse (L515: no disparity conversion needed)
        self._spatial_filter = rs.spatial_filter()
        self._spatial_filter.set_option(
            rs.option.filter_magnitude, config.SPATIAL_FILTER_MAGNITUDE
        )
        self._spatial_filter.set_option(
            rs.option.filter_smooth_alpha, config.SPATIAL_FILTER_ALPHA
        )
        self._spatial_filter.set_option(
            rs.option.filter_smooth_delta, config.SPATIAL_FILTER_DELTA
        )

        self._temporal_filter = rs.temporal_filter()
        self._temporal_filter.set_option(
            rs.option.filter_smooth_alpha, config.TEMPORAL_FILTER_ALPHA
        )
        self._temporal_filter.set_option(
            rs.option.filter_smooth_delta, config.TEMPORAL_FILTER_DELTA
        )

        self._hole_filling_filter = rs.hole_filling_filter()
        self._hole_filling_filter.set_option(
            rs.option.holes_fill, config.HOLE_FILLING_MODE
        )

        self._profile: rs.pipeline_profile | None = None
        self._intrinsics: rs.intrinsics | None = None

    def _check_usb_speed(self) -> None:
        """Warn if connected via USB 2.0 (depth limited to 320x240)."""
        if self._profile is None:
            return
        device = self._profile.get_device()
        usb_type = device.get_info(rs.camera_info.usb_type_descriptor)
        if usb_type.startswith("2."):
            logger.warning(
                "USB 2.0 connection detected (%s). "
                "Depth stream may be limited to 320x240. "
                "Use USB 3.x for full resolution.",
                usb_type,
            )

    def start(self) -> None:
        """Start streaming and discard warmup frames.

        Raises:
            RuntimeError: If the pipeline fails to start.
        """
        try:
            self._profile = self._pipeline.start(self._rs_config)
        except RuntimeError as e:
            logger.error("Failed to start RealSense pipeline: %s", e)
            raise

        self._check_usb_speed()

        # Cache intrinsics from aligned depth stream profile
        depth_stream = self._profile.get_stream(rs.stream.depth)
        self._intrinsics = (
            depth_stream.as_video_stream_profile().get_intrinsics()
        )

        # Discard warmup frames to let auto-exposure/white-balance stabilize
        logger.info("Warming up camera (%d frames)...", _WARMUP_FRAMES)
        for _ in range(_WARMUP_FRAMES):
            try:
                self._pipeline.wait_for_frames(timeout_ms=_FRAME_TIMEOUT_MS)
            except RuntimeError:
                pass
        logger.info("Camera ready.")

    def get_frame(self) -> FrameData | None:
        """Acquire one aligned and filtered frame.

        Returns:
            FrameData on success, None on failure.
        """
        if self._intrinsics is None:
            logger.error("Camera not started. Call start() first.")
            return None

        for attempt in range(_MAX_RETRIES):
            try:
                t0 = time.perf_counter()
                frames = self._pipeline.wait_for_frames(
                    timeout_ms=_FRAME_TIMEOUT_MS
                )
            except RuntimeError as e:
                logger.warning(
                    "Frame timeout (attempt %d/%d): %s",
                    attempt + 1,
                    _MAX_RETRIES,
                    e,
                )
                continue

            aligned = self._align.process(frames)
            t1 = time.perf_counter()  # end of capture + align
            depth_frame = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()

            if not depth_frame or not color_frame:
                logger.warning(
                    "Invalid frame on attempt %d/%d", attempt + 1, _MAX_RETRIES
                )
                continue

            # Apply filter chain: spatial → temporal → hole_filling
            depth_frame = self._spatial_filter.process(depth_frame).as_depth_frame()
            depth_frame = self._temporal_filter.process(depth_frame).as_depth_frame()
            depth_frame = self._hole_filling_filter.process(depth_frame).as_depth_frame()
            t2 = time.perf_counter()  # end of depth filtering

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            return FrameData(
                color_image=color_image,
                depth_image=depth_image,
                depth_frame=depth_frame,
                intrinsics=self._intrinsics,
                timestamp=time.monotonic(),
                timings={
                    "capture_align": (t1 - t0) * 1000.0,
                    "depth_filter": (t2 - t1) * 1000.0,
                },
            )

        logger.error("Failed to acquire frame after %d attempts.", _MAX_RETRIES)
        return None

    def stop(self) -> None:
        """Stop the pipeline."""
        try:
            self._pipeline.stop()
            logger.info("RealSense pipeline stopped.")
        except RuntimeError as e:
            logger.warning("Error stopping pipeline: %s", e)


def camera_thread(
    frame_queue: queue.Queue[FrameData],
    stop_event: threading.Event,
) -> None:
    """Entry point for camera Thread 1.

    Args:
        frame_queue: Queue for FrameData (maxsize=2).
        stop_event: Set this event to signal shutdown.
    """
    camera = RealsenseCamera()
    try:
        camera.start()
    except RuntimeError:
        logger.error("Camera failed to start. Signalling stop.")
        stop_event.set()
        return

    retry_count = 0

    while not stop_event.is_set():
        frame_data = camera.get_frame()

        if frame_data is None:
            retry_count += 1
            if retry_count >= _MAX_RETRIES:
                logger.error(
                    "Camera thread: %d consecutive failures. Stopping.",
                    retry_count,
                )
                stop_event.set()
                break
            continue

        retry_count = 0

        # Discard oldest frame if queue is full to prevent latency accumulation
        if frame_queue.full():
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                pass

        try:
            frame_queue.put_nowait(frame_data)
        except queue.Full:
            pass  # Rare race condition — skip

    camera.stop()
