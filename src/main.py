"""Entry point for RealSense Hand Control system.

Usage:
    python -m src.main [options]

Options:
    --no-control      Disable mouse control (hand tracking + display only).
    --resolution WxH  Override camera resolution (e.g. 1280x720).
    --record          Save annotated output to a timestamped .avi file.

Runtime keyboard controls:
    c   Toggle mouse control on/off
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
from src.hand_controller import HandController
from src.processor import ProcessingResult, processing_thread
from src.visualizer import HandVisualizer


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RealSense L515 hand gesture control for PC screen."
    )
    parser.add_argument(
        "--no-control",
        action="store_true",
        help="Disable mouse control (display hand tracking only).",
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
        args=(frame_queue, result_queue, stop_event),
        name="ProcessingThread",
        daemon=True,
    )
    cam_thread.start()
    proc_thread.start()
    logger.info("Camera and processing threads started.")

    # --- Hand controller -----------------------------------------------------
    controller: HandController | None = None
    if not args.no_control:
        try:
            controller = HandController()
            logger.info("Hand controller initialized. Press 'c' to toggle control.")
        except RuntimeError as exc:
            logger.warning("Hand controller disabled: %s", exc)

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
    visualizer = HandVisualizer()

    display_fps: float = 0.0
    display_fps_count: int = 0
    display_fps_time: float = time.monotonic()

    logger.info(
        "Display loop started. Press 'q' to quit, 'c' to toggle control."
    )

    try:
        while not stop_event.is_set():
            result: ProcessingResult | None = None
            try:
                result = result_queue.get(timeout=0.05)
            except queue.Empty:
                pass

            if result is not None:
                img = result.color_image.copy()

                # Hand landmark drawing
                if result.hands is not None:
                    img = visualizer.draw_hands(img, result.hands)

                # Hand controller update
                gesture_state = "idle"
                control_active = controller is not None and controller.control_active
                index_tip_px: tuple[int, int] | None = None
                pinch_dist = 0.0

                if controller is not None:
                    gesture_info = controller.update(result.hands)
                    gesture_state = gesture_info.state.value
                    control_active = gesture_info.control_active
                    index_tip_px = gesture_info.index_tip_px
                    pinch_dist = gesture_info.pinch_distance

                # Control overlay
                img = visualizer.draw_control_overlay(
                    img,
                    gesture_state=gesture_state,
                    control_active=control_active,
                    index_tip_px=index_tip_px,
                    pinch_distance=pinch_dist,
                )

                # FPS overlay
                img = visualizer.draw_fps(img, display_fps)

                cv2.imshow(config.WINDOW_NAME, img)

                if video_writer is not None:
                    video_writer.write(img)

                # Update display FPS counter
                display_fps_count += 1
                now = time.monotonic()
                elapsed = now - display_fps_time
                if elapsed >= 1.0:
                    display_fps = display_fps_count / elapsed
                    display_fps_count = 0
                    display_fps_time = now

            # Keyboard handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                logger.info("'q' pressed — stopping.")
                stop_event.set()
                break
            elif key == ord("c"):
                if controller is not None:
                    controller.toggle_control()
            elif key == ord("r"):
                if video_writer is not None:
                    video_writer.release()
                    video_writer = None
                    logger.info("Recording stopped.")
                else:
                    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
                    output_path = f"output_{timestamp_str}.avi"
                    fourcc = cv2.VideoWriter_fourcc(*"XVID")  # type: ignore[attr-defined]
                    video_writer = cv2.VideoWriter(
                        output_path,
                        fourcc,
                        float(config.CAMERA_FPS),
                        (config.CAMERA_WIDTH, config.CAMERA_HEIGHT),
                    )
                    if video_writer.isOpened():
                        logger.info("Recording started: %s", output_path)
                    else:
                        logger.error("Failed to start recording.")
                        video_writer = None

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
