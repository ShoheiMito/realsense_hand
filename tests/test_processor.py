"""Tests for processor module.

All tests run without RealSense hardware, MediaPipe model files, or a GPU.
External dependencies (PoseLandmarker, ExpressionRecognizer) are replaced with
MagicMock instances so every test is fully deterministic and offline.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import fields
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.depth_utils import KeypointSmoother
from src.expression import ExpressionResult
from src.processor import (
    HAND_LANDMARK_NAMES,
    LANDMARK_NAMES,
    FeatureFlags,
    HandResult,
    PoseKeypoint3D,
    PoseProcessor,
    ProcessingResult,
    processing_thread,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_frame_data() -> MagicMock:
    """Return a minimal mock that satisfies FrameData's interface."""
    fd = MagicMock()
    fd.color_image = np.zeros((480, 640, 3), dtype=np.uint8)
    fd.depth_image = np.zeros((480, 640), dtype=np.uint16)
    fd.timestamp = 1.0
    fd.timings = {}

    # depth_frame mock: get_distance always returns 1.0 m
    depth_frame = MagicMock()
    depth_frame.get_distance.return_value = 1.0
    depth_frame.get_units.return_value = 0.001
    depth_frame.get_data.return_value = fd.depth_image
    fd.depth_frame = depth_frame

    # intrinsics mock (simple pinhole)
    from types import SimpleNamespace

    fd.intrinsics = SimpleNamespace(fx=600.0, fy=600.0, ppx=320.0, ppy=240.0)
    return fd


def _make_processor(feature_flags: FeatureFlags | None = None) -> PoseProcessor:
    """Instantiate PoseProcessor bypassing model file I/O."""
    proc: PoseProcessor = PoseProcessor.__new__(PoseProcessor)
    proc._flags = feature_flags or FeatureFlags()
    proc._landmarker = MagicMock()
    proc._hand_landmarker = MagicMock()
    proc._expression_recognizer = MagicMock()
    proc._expression_recognizer.analyze.return_value = None
    proc._pose_smoother = KeypointSmoother(num_keypoints=33)
    proc._hand_smoothers = {}
    proc._frame_count = 0
    proc._last_ts_ms = 0
    proc._last_expression = None
    proc._last_hands = None
    proc._processing_fps = 0.0
    proc._last_fps_time = time.monotonic()
    proc._fps_frame_count = 0
    proc._timing_buf = []
    proc._REPORT_INTERVAL = 100
    return proc


def _make_pose_landmark(
    x: float = 0.5,
    y: float = 0.5,
    z: float = 0.0,
    visibility: float = 0.9,
) -> MagicMock:
    lm = MagicMock()
    lm.x = x
    lm.y = y
    lm.z = z
    lm.visibility = visibility
    return lm


def _mock_pose_result(landmarks: list[MagicMock]) -> MagicMock:
    result = MagicMock()
    result.pose_landmarks = [landmarks]
    return result


def _no_pose_result() -> MagicMock:
    result = MagicMock()
    result.pose_landmarks = []
    return result


def _mock_hand_result(
    n_hands: int = 1,
) -> MagicMock:
    """Return a mock HandLandmarker result with n_hands detected."""
    result = MagicMock()
    if n_hands == 0:
        result.hand_landmarks = []
        result.handedness = []
        return result

    hand_landmarks_list = []
    handedness_list = []
    names = ["Left", "Right"]
    for i in range(n_hands):
        lms = [_make_pose_landmark(x=0.5, y=0.5) for _ in range(21)]
        hand_landmarks_list.append(lms)
        h_cat = MagicMock()
        h_cat.category_name = names[i % 2]
        h_cat.score = 0.95
        handedness_list.append([h_cat])
    result.hand_landmarks = hand_landmarks_list
    result.handedness = handedness_list
    return result


# ---------------------------------------------------------------------------
# FeatureFlags
# ---------------------------------------------------------------------------


class TestFeatureFlags:
    def test_default_all_enabled(self) -> None:
        flags = FeatureFlags()
        assert flags.pose.is_set()
        assert flags.hand.is_set()
        assert flags.expression.is_set()

    def test_toggle_pose(self) -> None:
        flags = FeatureFlags()
        flags.pose.clear()
        assert not flags.pose.is_set()
        flags.pose.set()
        assert flags.pose.is_set()

    def test_independent_flags(self) -> None:
        flags = FeatureFlags()
        flags.hand.clear()
        assert flags.pose.is_set()
        assert not flags.hand.is_set()
        assert flags.expression.is_set()


# ---------------------------------------------------------------------------
# PoseKeypoint3D
# ---------------------------------------------------------------------------


class TestPoseKeypoint3D:
    def test_fields_exist(self) -> None:
        field_names = {f.name for f in fields(PoseKeypoint3D)}
        assert field_names == {"x", "y", "z", "visibility", "name"}

    def test_construction(self) -> None:
        kp = PoseKeypoint3D(x=1.0, y=2.0, z=3.0, visibility=0.9, name="nose")
        assert kp.x == 1.0
        assert kp.y == 2.0
        assert kp.z == 3.0
        assert kp.visibility == 0.9
        assert kp.name == "nose"

    def test_zero_coords(self) -> None:
        kp = PoseKeypoint3D(x=0.0, y=0.0, z=0.0, visibility=0.0, name="test")
        assert kp.x == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# HandResult
# ---------------------------------------------------------------------------


class TestHandResult:
    def test_fields_exist(self) -> None:
        field_names = {f.name for f in fields(HandResult)}
        assert field_names == {"handedness", "landmarks_2d", "keypoints_3d", "score"}

    def test_construction(self) -> None:
        hr = HandResult(
            handedness="Left",
            landmarks_2d=[(100, 200)],
            keypoints_3d=None,
            score=0.95,
        )
        assert hr.handedness == "Left"
        assert hr.score == pytest.approx(0.95)


# ---------------------------------------------------------------------------
# ProcessingResult
# ---------------------------------------------------------------------------


class TestProcessingResult:
    def test_fields_exist(self) -> None:
        field_names = {f.name for f in fields(ProcessingResult)}
        assert field_names == {
            "color_image",
            "landmarks_2d",
            "keypoints_3d",
            "hands",
            "expression",
            "processing_fps",
            "timestamp",
            "timings",
        }

    def test_construction_all_none(self) -> None:
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        result = ProcessingResult(
            color_image=img,
            landmarks_2d=None,
            keypoints_3d=None,
            hands=None,
            expression=None,
            processing_fps=0.0,
            timestamp=0.0,
        )
        assert result.landmarks_2d is None
        assert result.keypoints_3d is None
        assert result.hands is None
        assert result.expression is None

    def test_construction_with_data(self) -> None:
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        kp = PoseKeypoint3D(x=1.0, y=0.0, z=2.0, visibility=0.8, name="nose")
        expr = ExpressionResult(
            emotion="happy", confidence=0.9, blendshapes={}
        )
        result = ProcessingResult(
            color_image=img,
            landmarks_2d=[(320, 240)],
            keypoints_3d=[kp],
            hands=None,
            expression=expr,
            processing_fps=29.5,
            timestamp=1.0,
        )
        assert result.landmarks_2d == [(320, 240)]
        assert len(result.keypoints_3d) == 1  # type: ignore[arg-type]
        assert result.expression is not None
        assert result.expression.emotion == "happy"
        assert result.processing_fps == pytest.approx(29.5)


# ---------------------------------------------------------------------------
# LANDMARK_NAMES
# ---------------------------------------------------------------------------


class TestLandmarkNames:
    def test_count_is_33(self) -> None:
        assert len(LANDMARK_NAMES) == 33

    def test_all_strings(self) -> None:
        for name in LANDMARK_NAMES:
            assert isinstance(name, str)
            assert len(name) > 0

    def test_no_duplicates(self) -> None:
        assert len(LANDMARK_NAMES) == len(set(LANDMARK_NAMES))

    def test_known_names_present(self) -> None:
        assert "nose" in LANDMARK_NAMES
        assert "left_shoulder" in LANDMARK_NAMES
        assert "right_hip" in LANDMARK_NAMES
        assert "left_ankle" in LANDMARK_NAMES


class TestHandLandmarkNames:
    def test_count_is_21(self) -> None:
        assert len(HAND_LANDMARK_NAMES) == 21

    def test_no_duplicates(self) -> None:
        assert len(HAND_LANDMARK_NAMES) == len(set(HAND_LANDMARK_NAMES))

    def test_known_names_present(self) -> None:
        assert "wrist" in HAND_LANDMARK_NAMES
        assert "thumb_tip" in HAND_LANDMARK_NAMES
        assert "index_finger_tip" in HAND_LANDMARK_NAMES


# ---------------------------------------------------------------------------
# Shared mp fixture for process_frame tests
# ---------------------------------------------------------------------------


def _make_mp_mock() -> MagicMock:
    """Return a minimal mediapipe namespace mock for process_frame calls."""
    mp_mock = MagicMock()
    mp_mock.ImageFormat.SRGB = "SRGB"
    return mp_mock


# ---------------------------------------------------------------------------
# PoseProcessor.process_frame — no pose detected
# ---------------------------------------------------------------------------


class TestProcessFrameNoPose:
    @pytest.fixture(autouse=True)
    def _patch_mp(self) -> None:  # type: ignore[return]
        with patch("src.processor.mp", _make_mp_mock()):
            yield

    @pytest.fixture
    def proc(self) -> PoseProcessor:
        return _make_processor()

    def test_returns_processing_result(self, proc: PoseProcessor) -> None:
        proc._landmarker.detect_for_video.return_value = _no_pose_result()
        proc._hand_landmarker.detect_for_video.return_value = _mock_hand_result(0)
        fd = _make_frame_data()
        result = proc.process_frame(fd)
        assert isinstance(result, ProcessingResult)

    def test_landmarks_2d_is_none(self, proc: PoseProcessor) -> None:
        proc._landmarker.detect_for_video.return_value = _no_pose_result()
        proc._hand_landmarker.detect_for_video.return_value = _mock_hand_result(0)
        result = proc.process_frame(_make_frame_data())
        assert result.landmarks_2d is None

    def test_keypoints_3d_is_none(self, proc: PoseProcessor) -> None:
        proc._landmarker.detect_for_video.return_value = _no_pose_result()
        proc._hand_landmarker.detect_for_video.return_value = _mock_hand_result(0)
        result = proc.process_frame(_make_frame_data())
        assert result.keypoints_3d is None

    def test_color_image_preserved(self, proc: PoseProcessor) -> None:
        proc._landmarker.detect_for_video.return_value = _no_pose_result()
        proc._hand_landmarker.detect_for_video.return_value = _mock_hand_result(0)
        fd = _make_frame_data()
        fd.color_image[0, 0] = [255, 128, 0]
        result = proc.process_frame(fd)
        assert result.color_image[0, 0, 0] == 255

    def test_timestamp_preserved(self, proc: PoseProcessor) -> None:
        proc._landmarker.detect_for_video.return_value = _no_pose_result()
        proc._hand_landmarker.detect_for_video.return_value = _mock_hand_result(0)
        fd = _make_frame_data()
        fd.timestamp = 42.0
        result = proc.process_frame(fd)
        assert result.timestamp == pytest.approx(42.0)


# ---------------------------------------------------------------------------
# PoseProcessor.process_frame — pose detected
# ---------------------------------------------------------------------------


class TestProcessFrameWithPose:
    @pytest.fixture(autouse=True)
    def _patch_mp(self) -> None:  # type: ignore[return]
        with patch("src.processor.mp", _make_mp_mock()):
            yield

    @pytest.fixture
    def proc(self) -> PoseProcessor:
        return _make_processor()

    def _landmarks_33(self) -> list[MagicMock]:
        return [_make_pose_landmark(x=0.5, y=0.5) for _ in range(33)]

    def test_landmarks_2d_length_is_33(self, proc: PoseProcessor) -> None:
        proc._landmarker.detect_for_video.return_value = _mock_pose_result(
            self._landmarks_33()
        )
        proc._hand_landmarker.detect_for_video.return_value = _mock_hand_result(0)
        result = proc.process_frame(_make_frame_data())
        assert result.landmarks_2d is not None
        assert len(result.landmarks_2d) == 33

    def test_landmarks_2d_are_tuples_of_ints(self, proc: PoseProcessor) -> None:
        proc._landmarker.detect_for_video.return_value = _mock_pose_result(
            self._landmarks_33()
        )
        proc._hand_landmarker.detect_for_video.return_value = _mock_hand_result(0)
        result = proc.process_frame(_make_frame_data())
        assert result.landmarks_2d is not None
        for pt in result.landmarks_2d:
            assert isinstance(pt[0], int)
            assert isinstance(pt[1], int)

    def test_landmarks_2d_within_image_bounds(self, proc: PoseProcessor) -> None:
        proc._landmarker.detect_for_video.return_value = _mock_pose_result(
            self._landmarks_33()
        )
        proc._hand_landmarker.detect_for_video.return_value = _mock_hand_result(0)
        result = proc.process_frame(_make_frame_data())
        assert result.landmarks_2d is not None
        for x, y in result.landmarks_2d:
            assert 0 <= x < 640
            assert 0 <= y < 480

    def test_keypoints_3d_length_is_33(self, proc: PoseProcessor) -> None:
        proc._landmarker.detect_for_video.return_value = _mock_pose_result(
            self._landmarks_33()
        )
        proc._hand_landmarker.detect_for_video.return_value = _mock_hand_result(0)
        result = proc.process_frame(_make_frame_data())
        assert result.keypoints_3d is not None
        assert len(result.keypoints_3d) == 33

    def test_keypoints_3d_type(self, proc: PoseProcessor) -> None:
        proc._landmarker.detect_for_video.return_value = _mock_pose_result(
            self._landmarks_33()
        )
        proc._hand_landmarker.detect_for_video.return_value = _mock_hand_result(0)
        result = proc.process_frame(_make_frame_data())
        assert result.keypoints_3d is not None
        for kp in result.keypoints_3d:
            assert isinstance(kp, PoseKeypoint3D)

    def test_keypoints_3d_names_match_landmark_names(
        self, proc: PoseProcessor
    ) -> None:
        proc._landmarker.detect_for_video.return_value = _mock_pose_result(
            self._landmarks_33()
        )
        proc._hand_landmarker.detect_for_video.return_value = _mock_hand_result(0)
        result = proc.process_frame(_make_frame_data())
        assert result.keypoints_3d is not None
        for kp, expected_name in zip(result.keypoints_3d, LANDMARK_NAMES):
            assert kp.name == expected_name

    def test_keypoints_visibility_propagated(self, proc: PoseProcessor) -> None:
        lms = [_make_pose_landmark(visibility=0.75) for _ in range(33)]
        proc._landmarker.detect_for_video.return_value = _mock_pose_result(lms)
        proc._hand_landmarker.detect_for_video.return_value = _mock_hand_result(0)
        result = proc.process_frame(_make_frame_data())
        assert result.keypoints_3d is not None
        for kp in result.keypoints_3d:
            assert kp.visibility == pytest.approx(0.75)

    def test_keypoints_visibility_none_becomes_zero(
        self, proc: PoseProcessor
    ) -> None:
        lms = [_make_pose_landmark(visibility=None) for _ in range(33)]  # type: ignore[arg-type]
        proc._landmarker.detect_for_video.return_value = _mock_pose_result(lms)
        proc._hand_landmarker.detect_for_video.return_value = _mock_hand_result(0)
        result = proc.process_frame(_make_frame_data())
        assert result.keypoints_3d is not None
        for kp in result.keypoints_3d:
            assert kp.visibility == pytest.approx(0.0)

    def test_out_of_bounds_landmarks_clamped(self, proc: PoseProcessor) -> None:
        """Normalised coords > 1.0 or < 0.0 must be clamped to image bounds."""
        lms = [_make_pose_landmark(x=-0.5, y=2.0) for _ in range(33)]
        proc._landmarker.detect_for_video.return_value = _mock_pose_result(lms)
        proc._hand_landmarker.detect_for_video.return_value = _mock_hand_result(0)
        result = proc.process_frame(_make_frame_data())
        assert result.landmarks_2d is not None
        for x, y in result.landmarks_2d:
            assert x == 0
            assert y == 479


# ---------------------------------------------------------------------------
# PoseProcessor.process_frame — hand detection
# ---------------------------------------------------------------------------


class TestProcessFrameWithHands:
    @pytest.fixture(autouse=True)
    def _patch_mp(self) -> None:  # type: ignore[return]
        with patch("src.processor.mp", _make_mp_mock()):
            yield

    @pytest.fixture
    def proc(self) -> PoseProcessor:
        p = _make_processor()
        p._landmarker.detect_for_video.return_value = _no_pose_result()
        return p

    def test_hands_detected(self, proc: PoseProcessor) -> None:
        proc._hand_landmarker.detect_for_video.return_value = _mock_hand_result(2)
        result = proc.process_frame(_make_frame_data())
        assert result.hands is not None
        assert len(result.hands) == 2

    def test_hand_result_type(self, proc: PoseProcessor) -> None:
        proc._hand_landmarker.detect_for_video.return_value = _mock_hand_result(1)
        result = proc.process_frame(_make_frame_data())
        assert result.hands is not None
        assert isinstance(result.hands[0], HandResult)

    def test_hand_landmarks_count(self, proc: PoseProcessor) -> None:
        proc._hand_landmarker.detect_for_video.return_value = _mock_hand_result(1)
        result = proc.process_frame(_make_frame_data())
        assert result.hands is not None
        assert len(result.hands[0].landmarks_2d) == 21

    def test_hand_handedness(self, proc: PoseProcessor) -> None:
        proc._hand_landmarker.detect_for_video.return_value = _mock_hand_result(2)
        result = proc.process_frame(_make_frame_data())
        assert result.hands is not None
        names = {h.handedness for h in result.hands}
        assert names == {"Left", "Right"}

    def test_no_hands_returns_none(self, proc: PoseProcessor) -> None:
        proc._hand_landmarker.detect_for_video.return_value = _mock_hand_result(0)
        result = proc.process_frame(_make_frame_data())
        assert result.hands is None

    def test_hand_disabled_returns_none(self, proc: PoseProcessor) -> None:
        proc._flags.hand.clear()
        proc._hand_landmarker.detect_for_video.return_value = _mock_hand_result(2)
        result = proc.process_frame(_make_frame_data())
        assert result.hands is None


# ---------------------------------------------------------------------------
# Feature flags in process_frame
# ---------------------------------------------------------------------------


class TestFeatureFlagsInProcessFrame:
    @pytest.fixture(autouse=True)
    def _patch_mp(self) -> None:  # type: ignore[return]
        with patch("src.processor.mp", _make_mp_mock()):
            yield

    def test_pose_disabled_returns_none(self) -> None:
        flags = FeatureFlags()
        flags.pose.clear()
        proc = _make_processor(feature_flags=flags)
        proc._hand_landmarker.detect_for_video.return_value = _mock_hand_result(0)
        result = proc.process_frame(_make_frame_data())
        assert result.landmarks_2d is None
        assert result.keypoints_3d is None
        proc._landmarker.detect_for_video.assert_not_called()

    def test_expression_disabled_returns_none(self) -> None:
        flags = FeatureFlags()
        flags.expression.clear()
        proc = _make_processor(feature_flags=flags)
        proc._landmarker.detect_for_video.return_value = _no_pose_result()
        proc._hand_landmarker.detect_for_video.return_value = _mock_hand_result(0)
        result = proc.process_frame(_make_frame_data())
        assert result.expression is None
        proc._expression_recognizer.analyze.assert_not_called()


# ---------------------------------------------------------------------------
# PoseProcessor.process_frame — expression recognition interval
# ---------------------------------------------------------------------------


class TestExpressionInterval:
    @pytest.fixture(autouse=True)
    def _patch_mp(self) -> None:  # type: ignore[return]
        with patch("src.processor.mp", _make_mp_mock()):
            yield

    @pytest.fixture
    def proc(self) -> PoseProcessor:
        p = _make_processor()
        p._landmarker.detect_for_video.return_value = _no_pose_result()
        p._hand_landmarker.detect_for_video.return_value = _mock_hand_result(0)
        return p

    def test_expression_called_on_first_frame(self, proc: PoseProcessor) -> None:
        from src import config

        # Force frame_count to SKIP-1 so next call triggers expression
        proc._frame_count = config.EXPRESSION_SKIP_FRAMES - 1
        proc.process_frame(_make_frame_data())
        proc._expression_recognizer.analyze.assert_called_once()

    def test_expression_not_called_on_skipped_frame(
        self, proc: PoseProcessor
    ) -> None:
        from src import config

        proc._frame_count = 0  # next call: frame_count=1, skip if SKIP>1
        proc.process_frame(_make_frame_data())
        if config.EXPRESSION_SKIP_FRAMES > 1:
            proc._expression_recognizer.analyze.assert_not_called()

    def test_cached_expression_returned_on_skipped_frames(
        self, proc: PoseProcessor
    ) -> None:
        from src import config

        cached = ExpressionResult(emotion="happy", confidence=0.8, blendshapes={})
        proc._last_expression = cached
        proc._frame_count = 1  # next call is frame 2; skip if SKIP > 2

        result = proc.process_frame(_make_frame_data())
        if config.EXPRESSION_SKIP_FRAMES > 2:
            assert result.expression is cached
        else:
            assert result.expression is not None

    def test_expression_called_every_skip_frames(
        self, proc: PoseProcessor
    ) -> None:
        from src import config

        n_frames = config.EXPRESSION_SKIP_FRAMES * 4
        proc._frame_count = 0
        for _ in range(n_frames):
            proc.process_frame(_make_frame_data())

        expected_calls = n_frames // config.EXPRESSION_SKIP_FRAMES
        assert proc._expression_recognizer.analyze.call_count == expected_calls

    def test_expression_result_cached_after_recognition(
        self, proc: PoseProcessor
    ) -> None:
        from src import config

        expr = ExpressionResult(emotion="sad", confidence=0.6, blendshapes={})
        proc._expression_recognizer.analyze.return_value = expr
        proc._frame_count = config.EXPRESSION_SKIP_FRAMES - 1

        result = proc.process_frame(_make_frame_data())
        assert result.expression is expr
        assert proc._last_expression is expr


# ---------------------------------------------------------------------------
# PoseProcessor.close
# ---------------------------------------------------------------------------


class TestPoseProcessorClose:
    def test_close_calls_landmarker_close(self) -> None:
        proc = _make_processor()
        proc.close()
        proc._landmarker.close.assert_called_once()

    def test_close_calls_hand_landmarker_close(self) -> None:
        proc = _make_processor()
        proc.close()
        proc._hand_landmarker.close.assert_called_once()

    def test_close_calls_expression_recognizer_close(self) -> None:
        proc = _make_processor()
        proc.close()
        proc._expression_recognizer.close.assert_called_once()

    def test_close_tolerates_landmarker_error(self) -> None:
        proc = _make_processor()
        proc._landmarker.close.side_effect = RuntimeError("oops")
        proc.close()  # must not raise
        proc._expression_recognizer.close.assert_called_once()

    def test_close_handles_none_landmarkers(self) -> None:
        proc = _make_processor()
        proc._landmarker = None
        proc._hand_landmarker = None
        proc._expression_recognizer = None
        proc.close()  # must not raise


# ---------------------------------------------------------------------------
# FPS tracking
# ---------------------------------------------------------------------------


class TestFpsTracking:
    @pytest.fixture(autouse=True)
    def _patch_mp(self) -> None:  # type: ignore[return]
        with patch("src.processor.mp", _make_mp_mock()):
            yield

    def test_fps_is_zero_initially(self) -> None:
        proc = _make_processor()
        proc._landmarker.detect_for_video.return_value = _no_pose_result()
        proc._hand_landmarker.detect_for_video.return_value = _mock_hand_result(0)
        result = proc.process_frame(_make_frame_data())
        # Only one frame processed, elapsed < 1 s → fps still 0.0
        assert result.processing_fps == pytest.approx(0.0)

    def test_fps_updates_after_one_second(self) -> None:
        proc = _make_processor()
        proc._landmarker.detect_for_video.return_value = _no_pose_result()
        proc._hand_landmarker.detect_for_video.return_value = _mock_hand_result(0)

        # Simulate 30 frames in 1.1 s by back-dating the fps timer
        proc._last_fps_time = time.monotonic() - 1.1
        proc._fps_frame_count = 29  # 29 already counted; next call makes 30

        result = proc.process_frame(_make_frame_data())
        assert result.processing_fps == pytest.approx(30 / 1.1, rel=0.05)


# ---------------------------------------------------------------------------
# processing_thread
# ---------------------------------------------------------------------------


class TestProcessingThread:
    def _make_processor_mock(self) -> tuple[MagicMock, MagicMock]:
        """Return (PoseProcessor mock class, PoseProcessor instance mock)."""
        instance = MagicMock(spec=PoseProcessor)
        instance.process_frame.return_value = ProcessingResult(
            color_image=np.zeros((480, 640, 3), dtype=np.uint8),
            landmarks_2d=None,
            keypoints_3d=None,
            hands=None,
            expression=None,
            processing_fps=30.0,
            timestamp=1.0,
        )
        cls_mock = MagicMock(return_value=instance)
        return cls_mock, instance

    def test_thread_stops_on_stop_event(self) -> None:
        cls_mock, _ = self._make_processor_mock()
        fq: queue.Queue = queue.Queue(maxsize=2)
        rq: queue.Queue = queue.Queue(maxsize=2)
        stop_event = threading.Event()

        with patch("src.processor.PoseProcessor", cls_mock):
            t = threading.Thread(
                target=processing_thread,
                args=(fq, rq, stop_event),
                daemon=True,
            )
            t.start()
            time.sleep(0.05)
            stop_event.set()
            t.join(timeout=2.0)

        assert not t.is_alive()

    def test_thread_calls_process_frame(self) -> None:
        cls_mock, instance = self._make_processor_mock()
        fd = _make_frame_data()
        fq: queue.Queue = queue.Queue(maxsize=2)
        rq: queue.Queue = queue.Queue(maxsize=2)
        stop_event = threading.Event()

        fq.put(fd)

        with patch("src.processor.PoseProcessor", cls_mock):
            t = threading.Thread(
                target=processing_thread,
                args=(fq, rq, stop_event),
                daemon=True,
            )
            t.start()
            # Wait for the result to appear
            try:
                rq.get(timeout=1.0)
            except queue.Empty:
                pass
            stop_event.set()
            t.join(timeout=2.0)

        instance.process_frame.assert_called()

    def test_thread_calls_close_on_exit(self) -> None:
        cls_mock, instance = self._make_processor_mock()
        fq: queue.Queue = queue.Queue(maxsize=2)
        rq: queue.Queue = queue.Queue(maxsize=2)
        stop_event = threading.Event()

        with patch("src.processor.PoseProcessor", cls_mock):
            stop_event.set()  # stop immediately
            t = threading.Thread(
                target=processing_thread,
                args=(fq, rq, stop_event),
                daemon=True,
            )
            t.start()
            t.join(timeout=2.0)

        instance.close.assert_called_once()

    def test_thread_sets_stop_event_on_init_failure(self) -> None:
        cls_mock = MagicMock(side_effect=RuntimeError("model not found"))
        fq: queue.Queue = queue.Queue(maxsize=2)
        rq: queue.Queue = queue.Queue(maxsize=2)
        stop_event = threading.Event()

        with patch("src.processor.PoseProcessor", cls_mock):
            t = threading.Thread(
                target=processing_thread,
                args=(fq, rq, stop_event),
                daemon=True,
            )
            t.start()
            t.join(timeout=2.0)

        assert stop_event.is_set()

    def test_result_queue_does_not_exceed_maxsize(self) -> None:
        cls_mock, _ = self._make_processor_mock()
        fq: queue.Queue = queue.Queue(maxsize=2)
        rq: queue.Queue = queue.Queue(maxsize=2)
        stop_event = threading.Event()

        # Flood the frame queue
        for _ in range(10):
            fd = _make_frame_data()
            try:
                fq.put_nowait(fd)
            except queue.Full:
                break

        with patch("src.processor.PoseProcessor", cls_mock):
            t = threading.Thread(
                target=processing_thread,
                args=(fq, rq, stop_event),
                daemon=True,
            )
            t.start()
            time.sleep(0.1)
            stop_event.set()
            t.join(timeout=2.0)

        assert rq.qsize() <= 2

    def test_thread_tolerates_process_frame_exception(self) -> None:
        """A single bad frame must not kill the thread."""
        cls_mock, instance = self._make_processor_mock()
        good_result = ProcessingResult(
            color_image=np.zeros((480, 640, 3), dtype=np.uint8),
            landmarks_2d=None,
            keypoints_3d=None,
            hands=None,
            expression=None,
            processing_fps=0.0,
            timestamp=0.0,
        )
        instance.process_frame.side_effect = [
            RuntimeError("bad frame"),
            good_result,
        ]

        fq: queue.Queue = queue.Queue(maxsize=4)
        rq: queue.Queue = queue.Queue(maxsize=4)
        stop_event = threading.Event()

        fq.put(_make_frame_data())
        fq.put(_make_frame_data())

        with patch("src.processor.PoseProcessor", cls_mock):
            t = threading.Thread(
                target=processing_thread,
                args=(fq, rq, stop_event),
                daemon=True,
            )
            t.start()
            # Give thread time to process both frames
            time.sleep(0.1)
            stop_event.set()
            t.join(timeout=2.0)

        # The good result should have made it through
        assert rq.qsize() >= 1
