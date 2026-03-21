"""Entry point for RealSense L515 3D Pose Estimation + Expression Recognition system.

Usage:
    python -m src.main [options]

Options:
    --no-pose         Disable pose estimation.
    --no-hand         Disable hand detection.
    --no-expression   Disable expression recognition (reduces CPU load).
    --no-3d           Disable 3D coordinate HUD overlay.
    --resolution WxH  Override camera resolution (e.g. 1280x720).
    --record          Save annotated output to a timestamped .avi file.

Runtime keyboard controls:
    p   Toggle pose estimation on/off
    h   Toggle hand detection on/off
    f   Toggle face/expression on/off
    q   Quit
"""

from __future__ import annotations

import argparse
import logging
import queue
import signal
import sys
import threading
import time

import cv2

from src import config
from src.camera import FrameData, camera_thread
from src.processor import FeatureFlags, ProcessingResult, processing_thread
from src.visualizer import PoseVisualizer


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RealSense L515 real-time 3D pose estimation and expression recognition."
    )
    parser.add_argument(
        "--no-pose",
        action="store_true",
        help="Disable pose estimation.",
    )
    parser.add_argument(
        "--no-hand",
        action="store_true",
        help="Disable hand detection.",
    )
    parser.add_argument(
        "--no-expression",
        action="store_true",
        help="Disable expression recognition.",
    )
    parser.add_argument(
        "--no-3d",
        action="store_true",
        help="Disable 3D coordinate HUD overlay.",
    )
    parser.add_argument(
        "--resolution",
        metavar="WxH",
        help="Camera resolution override, e.g. 1280x720.",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Save annotated output video to a timestamped .avi file.",
    )
    return parser.parse_args()


def _apply_resolution(res_str: str) -> None:
    """Parse and apply a WxH resolution string to config.

    Args:
        res_str: Resolution string in 'WxH' format (case-insensitive).

    Raises:
        ValueError: If the format is invalid or values are non-positive.
    """
    parts = res_str.lower().split("x")
    if len(parts) != 2:
        raise ValueError(
            f"Invalid resolution {res_str!r}. Use WxH format, e.g. 640x480."
        )
    try:
        w, h = int(parts[0]), int(parts[1])
    except ValueError:
        raise ValueError(
            f"Resolution values must be integers, got {res_str!r}."
        ) from None
    if w <= 0 or h <= 0:
        raise ValueError(f"Resolution must be positive, got {w}x{h}.")
    config.CAMERA_WIDTH = w
    config.CAMERA_HEIGHT = h


def _print_display_report(buf: list[dict[str, float]]) -> None:
    """Print display-side timing table for the last N frames."""
    budget_ms = 1000.0 / 30.0
    steps: list[tuple[str, str]] = [
        ("draw_skeleton", "7. Draw Skeleton     "),
        ("draw_hands",    "8. Draw Hands        "),
        ("draw_3d",       "9. Draw 3D Info      "),
        ("draw_expr",     "10. Draw Expression   "),
        ("draw_fps",      "11. Draw FPS          "),
        ("imshow",        "12. cv2.imshow         "),
    ]
    print(  # noqa: T201
        f"\n{'─' * 62}\n"
        f"  Display-thread timing ({len(buf)} frames)\n"
        f"{'─' * 62}\n"
        f"  {'Step':<22} {'Avg':>6} {'Min':>6} {'Max':>6}  {'% Budget':>8}\n"
        f"{'─' * 62}"
    )
    display_total = 0.0
    for key, label in steps:
        vals = [f[key] for f in buf if key in f]
        if not vals:
            continue
        avg = sum(vals) / len(vals)
        mn = min(vals)
        mx = max(vals)
        pct = avg / budget_ms * 100.0
        display_total += avg
        print(f"  {label} {avg:6.1f} {mn:6.1f} {mx:6.1f}  {pct:7.1f}%")  # noqa: T201
    print(  # noqa: T201
        f"{'─' * 62}\n"
        f"  {'  Subtotal (display)':<22} {display_total:6.1f}"
        f"{'':>14}  {display_total / budget_ms * 100.0:7.1f}%\n"
        f"{'─' * 62}"
    )


def main() -> None:
    """Application entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    args = _parse_args()

    # --- Resolution override (must happen before camera thread starts) ------
    if args.resolution:
        try:
            _apply_resolution(args.resolution)
            logger.info(
                "Camera resolution set to %dx%d.", config.CAMERA_WIDTH, config.CAMERA_HEIGHT
            )
        except ValueError as exc:
            logger.error("%s", exc)
            sys.exit(1)

    # --- Feature flags (thread-safe toggles) --------------------------------
    feature_flags = FeatureFlags()
    if args.no_pose:
        feature_flags.pose.clear()
        logger.info("Pose estimation disabled.")
    if args.no_hand:
        feature_flags.hand.clear()
        logger.info("Hand detection disabled.")
    if args.no_expression:
        feature_flags.expression.clear()
        logger.info("Expression recognition disabled.")

    # --- Shared queues and stop event ----------------------------------------
    frame_queue: queue.Queue[FrameData] = queue.Queue(maxsize=config.FRAME_QUEUE_SIZE)
    result_queue: queue.Queue[ProcessingResult] = queue.Queue(
        maxsize=config.RESULT_QUEUE_SIZE
    )
    stop_event = threading.Event()

    # --- SIGINT / Ctrl+C handler ---------------------------------------------
    def _on_sigint(signum: int, frame: object) -> None:  # noqa: ARG001
        logger.info("Ctrl+C received — shutting down.")
        stop_event.set()

    signal.signal(signal.SIGINT, _on_sigint)

    # --- Background threads --------------------------------------------------
    cam_thread = threading.Thread(
        target=camera_thread,
        args=(frame_queue, stop_event),
        name="CameraThread",
        daemon=True,
    )
    proc_thread = threading.Thread(
        target=processing_thread,
        args=(frame_queue, result_queue, stop_event, feature_flags),
        name="ProcessingThread",
        daemon=True,
    )
    cam_thread.start()
    proc_thread.start()
    logger.info("Camera and processing threads started.")

    # --- Optional video recorder ---------------------------------------------
    video_writer: cv2.VideoWriter | None = None
    if args.record:
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        output_path = f"output_{timestamp_str}.avi"
        fourcc = cv2.VideoWriter_fourcc(*"XVID")  # type: ignore[attr-defined]
        video_writer = cv2.VideoWriter(
            output_path,
            fourcc,
            float(config.CAMERA_FPS),
            (config.CAMERA_WIDTH, config.CAMERA_HEIGHT),
        )
        if not video_writer.isOpened():
            logger.error("Failed to open VideoWriter for path: %s", output_path)
            video_writer = None
        else:
            logger.info("Recording output to: %s", output_path)

    # --- Main-thread display loop --------------------------------------------
    visualizer = PoseVisualizer()
    show_3d = not args.no_3d

    display_fps: float = 0.0
    display_fps_count: int = 0
    display_fps_time: float = time.monotonic()

    # Display-side timing accumulation
    _disp_buf: list[dict[str, float]] = []
    _DISP_REPORT_INTERVAL = 100

    logger.info(
        "Display loop started. Press 'q' to quit, 'p'/'h'/'f' to toggle features."
    )

    try:
        while not stop_event.is_set():
            # Poll the result queue with a short timeout so 'q' stays responsive.
            result: ProcessingResult | None = None
            try:
                result = result_queue.get(timeout=0.05)
            except queue.Empty:
                pass

            if result is not None:
                # Build annotated image layer by layer
                img = result.color_image.copy()
                _dt: dict[str, float] = {}

                if result.landmarks_2d is not None:
                    if result.keypoints_3d is not None:
                        kpts = [
                            (x, y, kp.visibility)
                            for (x, y), kp in zip(
                                result.landmarks_2d, result.keypoints_3d
                            )
                        ]
                    else:
                        kpts = [(x, y, 1.0) for x, y in result.landmarks_2d]
                    _t0 = time.perf_counter()
                    img = visualizer.draw_skeleton(img, kpts)
                    _dt["draw_skeleton"] = (time.perf_counter() - _t0) * 1000.0

                if result.hands is not None:
                    _t0 = time.perf_counter()
                    img = visualizer.draw_hands(img, result.hands)
                    _dt["draw_hands"] = (time.perf_counter() - _t0) * 1000.0

                if show_3d and result.keypoints_3d is not None:
                    _t0 = time.perf_counter()
                    img = visualizer.draw_3d_info(img, result.keypoints_3d)
                    _dt["draw_3d"] = (time.perf_counter() - _t0) * 1000.0

                if feature_flags.expression.is_set():
                    _t0 = time.perf_counter()
                    img = visualizer.draw_expression(img, result.expression)
                    _dt["draw_expr"] = (time.perf_counter() - _t0) * 1000.0

                _t0 = time.perf_counter()
                img = visualizer.draw_fps(img, display_fps)
                _dt["draw_fps"] = (time.perf_counter() - _t0) * 1000.0

                # Feature status HUD
                img = visualizer.draw_feature_status(img, feature_flags)

                _t0 = time.perf_counter()
                cv2.imshow(config.WINDOW_NAME, img)
                _dt["imshow"] = (time.perf_counter() - _t0) * 1000.0

                if video_writer is not None:
                    video_writer.write(img)

                # Accumulate display timings
                _disp_buf.append(_dt)
                if len(_disp_buf) >= _DISP_REPORT_INTERVAL:
                    _print_display_report(_disp_buf)
                    _disp_buf.clear()

                # Update display FPS counter
                display_fps_count += 1
                now = time.monotonic()
                elapsed = now - display_fps_time
                if elapsed >= 1.0:
                    display_fps = display_fps_count / elapsed
                    display_fps_count = 0
                    display_fps_time = now

            # Check keyboard — keep waitKey outside the 'if result' block so it
            # polls even on empty frames (required to keep the OpenCV window alive).
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                logger.info("'q' pressed — stopping.")
                stop_event.set()
                break
            elif key == ord("p"):
                if feature_flags.pose.is_set():
                    feature_flags.pose.clear()
                else:
                    feature_flags.pose.set()
                logger.info("Pose: %s", "ON" if feature_flags.pose.is_set() else "OFF")
            elif key == ord("h"):
                if feature_flags.hand.is_set():
                    feature_flags.hand.clear()
                else:
                    feature_flags.hand.set()
                logger.info("Hand: %s", "ON" if feature_flags.hand.is_set() else "OFF")
            elif key == ord("f"):
                if feature_flags.expression.is_set():
                    feature_flags.expression.clear()
                else:
                    feature_flags.expression.set()
                logger.info(
                    "Expression: %s",
                    "ON" if feature_flags.expression.is_set() else "OFF",
                )

    except Exception as exc:  # noqa: BLE001
        logger.error("Unexpected error in display loop: %s", exc)
        stop_event.set()

    finally:
        cv2.destroyAllWindows()
        if video_writer is not None:
            video_writer.release()
            logger.info("Video file saved.")

        logger.info("Waiting for threads to finish...")
        cam_thread.join(timeout=5.0)
        proc_thread.join(timeout=5.0)

        if cam_thread.is_alive():
            logger.warning("CameraThread did not stop within timeout.")
        if proc_thread.is_alive():
            logger.warning("ProcessingThread did not stop within timeout.")

        logger.info("Shutdown complete.")


if __name__ == "__main__":
    main()
