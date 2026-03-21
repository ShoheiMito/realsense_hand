"""Tests for camera module."""

import queue
import threading
import time
from dataclasses import fields
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers — build a minimal pyrealsense2 mock
# ---------------------------------------------------------------------------

def _make_rs_mock() -> MagicMock:
    """Return a minimal mock for the pyrealsense2 namespace."""
    rs = MagicMock(name="pyrealsense2")

    # stream / format constants
    rs.stream.color = "color"
    rs.stream.depth = "depth"
    rs.format.bgr8 = "bgr8"
    rs.format.z16 = "z16"

    # option constants
    rs.option.filter_magnitude = "filter_magnitude"
    rs.option.filter_smooth_alpha = "filter_smooth_alpha"
    rs.option.filter_smooth_delta = "filter_smooth_delta"
    rs.option.holes_fill = "holes_fill"

    # camera_info
    rs.camera_info.usb_type_descriptor = "usb_type_descriptor"

    return rs



# ---------------------------------------------------------------------------
# FrameData dataclass
# ---------------------------------------------------------------------------

class TestFrameData:
    def test_fields_exist(self) -> None:
        """FrameData must have the five expected fields."""
        with patch.dict("sys.modules", {"pyrealsense2": _make_rs_mock()}):
            from src.camera import FrameData  # noqa: PLC0415

            field_names = {f.name for f in fields(FrameData)}
            assert field_names == {
                "color_image",
                "depth_image",
                "depth_frame",
                "intrinsics",
                "timestamp",
            }

    def test_construction(self) -> None:
        with patch.dict("sys.modules", {"pyrealsense2": _make_rs_mock()}):
            from src.camera import FrameData  # noqa: PLC0415

            color = np.zeros((480, 640, 3), dtype=np.uint8)
            depth = np.zeros((480, 640), dtype=np.uint16)
            fd = FrameData(
                color_image=color,
                depth_image=depth,
                depth_frame=MagicMock(),
                intrinsics=MagicMock(),
                timestamp=1.0,
            )
            assert fd.timestamp == 1.0
            assert fd.color_image.shape == (480, 640, 3)


# ---------------------------------------------------------------------------
# RealsenseCamera
# ---------------------------------------------------------------------------

class TestRealsenseCameraInit:
    def test_init_creates_pipeline_and_filters(self) -> None:
        rs_mock = _make_rs_mock()
        with patch.dict("sys.modules", {"pyrealsense2": rs_mock}):
            import importlib

            import src.camera as cam_module  # noqa: PLC0415

            importlib.reload(cam_module)
            cam_module.RealsenseCamera()

            rs_mock.pipeline.assert_called_once()
            rs_mock.config.assert_called_once()
            rs_mock.align.assert_called_once_with("color")
            rs_mock.spatial_filter.assert_called_once()
            rs_mock.temporal_filter.assert_called_once()
            rs_mock.hole_filling_filter.assert_called_once()


class TestRealsenseCameraStart:
    def _make_camera(self, rs_mock: MagicMock):
        import importlib

        import src.camera as cam_module  # noqa: PLC0415

        importlib.reload(cam_module)
        return cam_module.RealsenseCamera(), cam_module

    def test_start_raises_on_pipeline_failure(self) -> None:
        rs_mock = _make_rs_mock()
        rs_mock.pipeline.return_value.start.side_effect = RuntimeError("no device")

        with patch.dict("sys.modules", {"pyrealsense2": rs_mock}):
            cam, _ = self._make_camera(rs_mock)
            with pytest.raises(RuntimeError, match="no device"):
                cam.start()

    def test_start_success_warms_up(self) -> None:
        rs_mock = _make_rs_mock()

        profile = MagicMock()
        intrinsics = MagicMock()
        video_profile = MagicMock()
        video_profile.get_intrinsics.return_value = intrinsics
        depth_stream = MagicMock()
        depth_stream.as_video_stream_profile.return_value = video_profile
        profile.get_stream.return_value = depth_stream
        device = MagicMock()
        device.get_info.return_value = "3.2"
        profile.get_device.return_value = device

        pipeline = MagicMock()
        pipeline.start.return_value = profile
        rs_mock.pipeline.return_value = pipeline

        with patch.dict("sys.modules", {"pyrealsense2": rs_mock}):
            cam, cam_module = self._make_camera(rs_mock)
            with patch.object(cam_module, "_WARMUP_FRAMES", 3):
                cam.start()

            assert pipeline.wait_for_frames.call_count == 3

    def test_start_warns_on_usb2(self, caplog: pytest.LogCaptureFixture) -> None:
        import logging

        rs_mock = _make_rs_mock()

        profile = MagicMock()
        intrinsics = MagicMock()
        video_profile = MagicMock()
        video_profile.get_intrinsics.return_value = intrinsics
        depth_stream = MagicMock()
        depth_stream.as_video_stream_profile.return_value = video_profile
        profile.get_stream.return_value = depth_stream
        device = MagicMock()
        device.get_info.return_value = "2.1"  # USB 2.x
        profile.get_device.return_value = device

        pipeline = MagicMock()
        pipeline.start.return_value = profile
        rs_mock.pipeline.return_value = pipeline

        with patch.dict("sys.modules", {"pyrealsense2": rs_mock}):
            import importlib

            import src.camera as cam_module  # noqa: PLC0415

            importlib.reload(cam_module)
            cam = cam_module.RealsenseCamera()
            with patch.object(cam_module, "_WARMUP_FRAMES", 0):
                with caplog.at_level(logging.WARNING, logger="src.camera"):
                    cam.start()

        assert any("USB 2" in r.message for r in caplog.records)


class TestRealsenseCameraGetFrame:
    def _setup(self):
        rs_mock = _make_rs_mock()

        color_data = np.zeros((480, 640, 3), dtype=np.uint8)
        depth_data = np.ones((480, 640), dtype=np.uint16) * 500

        depth_frame = MagicMock()
        depth_frame.get_data.return_value = depth_data
        depth_frame.as_depth_frame.return_value = depth_frame

        filtered = MagicMock()
        filtered.as_depth_frame.return_value = depth_frame

        color_frame = MagicMock()
        color_frame.get_data.return_value = color_data

        aligned = MagicMock()
        aligned.get_depth_frame.return_value = depth_frame
        aligned.get_color_frame.return_value = color_frame

        intrinsics = MagicMock()
        video_profile = MagicMock()
        video_profile.get_intrinsics.return_value = intrinsics
        depth_stream = MagicMock()
        depth_stream.as_video_stream_profile.return_value = video_profile
        profile = MagicMock()
        profile.get_stream.return_value = depth_stream
        device = MagicMock()
        device.get_info.return_value = "3.2"
        profile.get_device.return_value = device

        pipeline = MagicMock()
        pipeline.start.return_value = profile
        pipeline.wait_for_frames.return_value = MagicMock()

        rs_mock.pipeline.return_value = pipeline

        align = MagicMock()
        align.process.return_value = aligned
        rs_mock.align.return_value = align

        # Filters: process() returns something whose as_depth_frame() returns depth_frame
        rs_mock.spatial_filter.return_value.process.return_value = filtered
        rs_mock.temporal_filter.return_value.process.return_value = filtered
        rs_mock.hole_filling_filter.return_value.process.return_value = filtered

        return rs_mock, depth_frame, color_data, depth_data, intrinsics

    def test_get_frame_returns_frame_data(self) -> None:
        rs_mock, depth_frame, color_data, depth_data, intrinsics = self._setup()

        with patch.dict("sys.modules", {"pyrealsense2": rs_mock}):
            import importlib

            import src.camera as cam_module  # noqa: PLC0415

            importlib.reload(cam_module)
            cam = cam_module.RealsenseCamera()
            with patch.object(cam_module, "_WARMUP_FRAMES", 0):
                cam.start()

            result = cam.get_frame()

        assert result is not None
        assert result.color_image.shape == (480, 640, 3)
        assert result.depth_image.shape == (480, 640)
        assert isinstance(result.timestamp, float)

    def test_get_frame_returns_none_before_start(self) -> None:
        rs_mock = _make_rs_mock()
        with patch.dict("sys.modules", {"pyrealsense2": rs_mock}):
            import importlib

            import src.camera as cam_module  # noqa: PLC0415

            importlib.reload(cam_module)
            cam = cam_module.RealsenseCamera()
            result = cam.get_frame()

        assert result is None

    def test_get_frame_retries_on_timeout(self) -> None:
        rs_mock, depth_frame, color_data, depth_data, intrinsics = self._setup()
        # First call raises, second succeeds
        rs_mock.pipeline.return_value.wait_for_frames.side_effect = [
            RuntimeError("timeout"),
            MagicMock(),
        ]

        with patch.dict("sys.modules", {"pyrealsense2": rs_mock}):
            import importlib

            import src.camera as cam_module  # noqa: PLC0415

            importlib.reload(cam_module)
            cam = cam_module.RealsenseCamera()
            with patch.object(cam_module, "_WARMUP_FRAMES", 0):
                cam.start()
            # Reset side_effect after start() exhausted it
            rs_mock.pipeline.return_value.wait_for_frames.side_effect = [
                RuntimeError("timeout"),
                MagicMock(),
            ]
            result = cam.get_frame()

        assert result is not None

    def test_get_frame_returns_none_after_max_retries(self) -> None:
        rs_mock, *_ = self._setup()
        rs_mock.pipeline.return_value.wait_for_frames.side_effect = RuntimeError(
            "timeout"
        )

        with patch.dict("sys.modules", {"pyrealsense2": rs_mock}):
            import importlib

            import src.camera as cam_module  # noqa: PLC0415

            importlib.reload(cam_module)
            cam = cam_module.RealsenseCamera()
            with patch.object(cam_module, "_WARMUP_FRAMES", 0):
                cam.start()
            result = cam.get_frame()

        assert result is None


# ---------------------------------------------------------------------------
# camera_thread
# ---------------------------------------------------------------------------

class TestCameraThread:
    def test_thread_stops_on_stop_event(self) -> None:
        rs_mock = _make_rs_mock()

        color_data = np.zeros((480, 640, 3), dtype=np.uint8)
        depth_data = np.ones((480, 640), dtype=np.uint16) * 500

        depth_frame = MagicMock()
        depth_frame.get_data.return_value = depth_data
        depth_frame.as_depth_frame.return_value = depth_frame

        filtered = MagicMock()
        filtered.as_depth_frame.return_value = depth_frame

        color_frame = MagicMock()
        color_frame.get_data.return_value = color_data

        aligned = MagicMock()
        aligned.get_depth_frame.return_value = depth_frame
        aligned.get_color_frame.return_value = color_frame

        intrinsics = MagicMock()
        video_profile = MagicMock()
        video_profile.get_intrinsics.return_value = intrinsics
        depth_stream = MagicMock()
        depth_stream.as_video_stream_profile.return_value = video_profile
        profile = MagicMock()
        profile.get_stream.return_value = depth_stream
        device = MagicMock()
        device.get_info.return_value = "3.2"
        profile.get_device.return_value = device

        pipeline = MagicMock()
        pipeline.start.return_value = profile
        rs_mock.pipeline.return_value = pipeline

        align = MagicMock()
        align.process.return_value = aligned
        rs_mock.align.return_value = align

        rs_mock.spatial_filter.return_value.process.return_value = filtered
        rs_mock.temporal_filter.return_value.process.return_value = filtered
        rs_mock.hole_filling_filter.return_value.process.return_value = filtered

        with patch.dict("sys.modules", {"pyrealsense2": rs_mock}):
            import importlib

            import src.camera as cam_module  # noqa: PLC0415

            importlib.reload(cam_module)
            with patch.object(cam_module, "_WARMUP_FRAMES", 0):
                fq: queue.Queue = queue.Queue(maxsize=2)
                stop_event = threading.Event()

                t = threading.Thread(
                    target=cam_module.camera_thread,
                    args=(fq, stop_event),
                    daemon=True,
                )
                t.start()
                time.sleep(0.05)
                stop_event.set()
                t.join(timeout=2.0)

        assert not t.is_alive()

    def test_thread_discards_old_frames_when_queue_full(self) -> None:
        """Queue must never exceed maxsize=2."""
        rs_mock = _make_rs_mock()

        color_data = np.zeros((480, 640, 3), dtype=np.uint8)
        depth_data = np.ones((480, 640), dtype=np.uint16) * 500

        depth_frame = MagicMock()
        depth_frame.get_data.return_value = depth_data
        depth_frame.as_depth_frame.return_value = depth_frame

        filtered = MagicMock()
        filtered.as_depth_frame.return_value = depth_frame

        color_frame = MagicMock()
        color_frame.get_data.return_value = color_data

        aligned = MagicMock()
        aligned.get_depth_frame.return_value = depth_frame
        aligned.get_color_frame.return_value = color_frame

        intrinsics = MagicMock()
        video_profile = MagicMock()
        video_profile.get_intrinsics.return_value = intrinsics
        depth_stream = MagicMock()
        depth_stream.as_video_stream_profile.return_value = video_profile
        profile = MagicMock()
        profile.get_stream.return_value = depth_stream
        device = MagicMock()
        device.get_info.return_value = "3.2"
        profile.get_device.return_value = device

        pipeline = MagicMock()
        pipeline.start.return_value = profile
        rs_mock.pipeline.return_value = pipeline

        align = MagicMock()
        align.process.return_value = aligned
        rs_mock.align.return_value = align

        rs_mock.spatial_filter.return_value.process.return_value = filtered
        rs_mock.temporal_filter.return_value.process.return_value = filtered
        rs_mock.hole_filling_filter.return_value.process.return_value = filtered

        with patch.dict("sys.modules", {"pyrealsense2": rs_mock}):
            import importlib

            import src.camera as cam_module  # noqa: PLC0415

            importlib.reload(cam_module)
            with patch.object(cam_module, "_WARMUP_FRAMES", 0):
                fq: queue.Queue = queue.Queue(maxsize=2)
                stop_event = threading.Event()

                t = threading.Thread(
                    target=cam_module.camera_thread,
                    args=(fq, stop_event),
                    daemon=True,
                )
                t.start()
                time.sleep(0.1)
                stop_event.set()
                t.join(timeout=2.0)

        assert fq.qsize() <= 2

    def test_thread_sets_stop_event_on_camera_start_failure(self) -> None:
        rs_mock = _make_rs_mock()
        rs_mock.pipeline.return_value.start.side_effect = RuntimeError("no device")

        with patch.dict("sys.modules", {"pyrealsense2": rs_mock}):
            import importlib

            import src.camera as cam_module  # noqa: PLC0415

            importlib.reload(cam_module)
            with patch.object(cam_module, "_WARMUP_FRAMES", 0):
                fq: queue.Queue = queue.Queue(maxsize=2)
                stop_event = threading.Event()

                t = threading.Thread(
                    target=cam_module.camera_thread,
                    args=(fq, stop_event),
                    daemon=True,
                )
                t.start()
                t.join(timeout=2.0)

        assert stop_event.is_set()
