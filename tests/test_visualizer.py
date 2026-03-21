"""Tests for visualizer module."""

from __future__ import annotations

import numpy as np
import pytest

from src.expression import ExpressionResult
from src.processor import FeatureFlags, HandResult, PoseKeypoint3D, ProcessingResult
from src.visualizer import (
    HAND_CONNECTIONS,
    POSE_CONNECTIONS,
    PoseVisualizer,
    _build_landmark_colors,
    _draw_gradient_line,
    _lerp_color,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def blank_frame() -> np.ndarray:
    """Return a black 640×480 BGR frame."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


def _make_keypoints_2d(
    n: int = 33,
    visibility: float = 1.0,
) -> list[tuple[int, int, float]]:
    """Return *n* dummy (x, y, visibility) keypoints spread across the frame."""
    return [(10 + i * 10, 100 + i * 5, visibility) for i in range(n)]


def _make_keypoints_3d(
    n: int = 33,
    z: float = 1.5,
    visibility: float = 1.0,
) -> list[PoseKeypoint3D]:
    """Return *n* PoseKeypoint3D objects with dummy values."""
    return [
        PoseKeypoint3D(x=0.1 * i, y=0.05 * i, z=z, visibility=visibility, name=f"KP_{i}")
        for i in range(n)
    ]


def _make_expression(emotion: str = "happy", confidence: float = 0.9) -> ExpressionResult:
    return ExpressionResult(emotion=emotion, confidence=confidence, blendshapes={})


# ---------------------------------------------------------------------------
# _lerp_color
# ---------------------------------------------------------------------------


class TestLerpColor:
    def test_at_t0_returns_c1(self) -> None:
        c1 = (10, 20, 30)
        c2 = (110, 120, 130)
        assert _lerp_color(c1, c2, 0.0) == c1

    def test_at_t1_returns_c2(self) -> None:
        c1 = (10, 20, 30)
        c2 = (110, 120, 130)
        assert _lerp_color(c1, c2, 1.0) == c2

    def test_midpoint(self) -> None:
        c1 = (0, 0, 0)
        c2 = (100, 200, 100)
        result = _lerp_color(c1, c2, 0.5)
        assert result == (50, 100, 50)


# ---------------------------------------------------------------------------
# _build_landmark_colors
# ---------------------------------------------------------------------------


class TestBuildLandmarkColors:
    def test_returns_33_colors(self) -> None:
        colors = _build_landmark_colors(33)
        assert len(colors) == 33

    def test_all_values_in_range(self) -> None:
        for b, g, r in _build_landmark_colors(33):
            assert 0 <= b <= 255
            assert 0 <= g <= 255
            assert 0 <= r <= 255

    def test_colors_are_distinct(self) -> None:
        colors = _build_landmark_colors(33)
        # At least 20 distinct colours expected from the golden-angle distribution.
        assert len(set(colors)) >= 20

    def test_custom_n(self) -> None:
        assert len(_build_landmark_colors(10)) == 10


# ---------------------------------------------------------------------------
# _draw_gradient_line
# ---------------------------------------------------------------------------


class TestDrawGradientLine:
    def test_does_not_raise(self, blank_frame: np.ndarray) -> None:
        canvas = blank_frame.copy()
        _draw_gradient_line(canvas, (50, 100), (200, 300), (0, 0, 255), (255, 0, 0), 2)

    def test_modifies_canvas(self, blank_frame: np.ndarray) -> None:
        canvas = blank_frame.copy()
        _draw_gradient_line(canvas, (10, 10), (200, 200), (255, 255, 255), (0, 0, 255), 3)
        assert not np.array_equal(canvas, blank_frame), "Canvas should be modified"

    def test_same_point_does_not_crash(self, blank_frame: np.ndarray) -> None:
        canvas = blank_frame.copy()
        _draw_gradient_line(canvas, (100, 100), (100, 100), (255, 0, 0), (0, 255, 0), 2)


# ---------------------------------------------------------------------------
# PoseVisualizer.draw_skeleton
# ---------------------------------------------------------------------------


class TestDrawSkeleton:
    def test_returns_ndarray(self, blank_frame: np.ndarray) -> None:
        vis = PoseVisualizer()
        kpts = _make_keypoints_2d(33, visibility=1.0)
        result = vis.draw_skeleton(blank_frame, kpts)
        assert isinstance(result, np.ndarray)

    def test_same_shape_as_input(self, blank_frame: np.ndarray) -> None:
        vis = PoseVisualizer()
        kpts = _make_keypoints_2d(33)
        result = vis.draw_skeleton(blank_frame, kpts)
        assert result.shape == blank_frame.shape

    def test_does_not_modify_original(self, blank_frame: np.ndarray) -> None:
        vis = PoseVisualizer()
        original = blank_frame.copy()
        kpts = _make_keypoints_2d(33)
        vis.draw_skeleton(blank_frame, kpts)
        np.testing.assert_array_equal(blank_frame, original)

    def test_draws_on_canvas(self, blank_frame: np.ndarray) -> None:
        vis = PoseVisualizer()
        kpts = _make_keypoints_2d(33, visibility=1.0)
        result = vis.draw_skeleton(blank_frame, kpts)
        assert not np.array_equal(result, blank_frame), "Skeleton should be visible"

    def test_low_confidence_keypoints_skipped(self, blank_frame: np.ndarray) -> None:
        vis = PoseVisualizer()
        # All keypoints below threshold — nothing should be drawn.
        kpts = _make_keypoints_2d(33, visibility=0.0)
        result = vis.draw_skeleton(blank_frame, kpts, conf_thresh=0.5)
        np.testing.assert_array_equal(
            result, blank_frame, err_msg="Low-confidence keypoints must not be drawn"
        )

    def test_partial_confidence_filters_connections(self, blank_frame: np.ndarray) -> None:
        vis = PoseVisualizer()
        # Only first keypoint has high confidence — all connections are skipped
        # because the other endpoint has zero visibility.
        kpts = [(100, 100, 0.9 if i == 0 else 0.0) for i in range(33)]
        result = vis.draw_skeleton(blank_frame, kpts, conf_thresh=0.5)
        # No connections drawn, but the single visible keypoint circle is rendered.
        # Verify that result differs from blank (circle was drawn).
        assert not np.array_equal(result, blank_frame)
        # Verify that ONLY the region near keypoint 0 was modified (no limb lines).
        # Clear the circle region and compare — rest should be black.
        check = result.copy()
        pad = vis._radius + 2
        check[100 - pad : 100 + pad + 1, 100 - pad : 100 + pad + 1] = 0
        np.testing.assert_array_equal(
            check, blank_frame,
            err_msg="Only keypoint circle should be drawn, no connection lines",
        )

    def test_empty_keypoints_returns_copy(self, blank_frame: np.ndarray) -> None:
        vis = PoseVisualizer()
        result = vis.draw_skeleton(blank_frame, [])
        np.testing.assert_array_equal(result, blank_frame)

    def test_fewer_than_33_keypoints(self, blank_frame: np.ndarray) -> None:
        """Connections referencing missing landmark indices are skipped gracefully."""
        vis = PoseVisualizer()
        kpts = _make_keypoints_2d(5, visibility=1.0)
        result = vis.draw_skeleton(blank_frame, kpts)
        assert result.shape == blank_frame.shape


# ---------------------------------------------------------------------------
# PoseVisualizer.draw_expression
# ---------------------------------------------------------------------------


class TestDrawExpression:
    def test_returns_ndarray(self, blank_frame: np.ndarray) -> None:
        vis = PoseVisualizer()
        result = vis.draw_expression(blank_frame, None)
        assert isinstance(result, np.ndarray)

    def test_same_shape(self, blank_frame: np.ndarray) -> None:
        vis = PoseVisualizer()
        result = vis.draw_expression(blank_frame, _make_expression())
        assert result.shape == blank_frame.shape

    def test_does_not_modify_original(self, blank_frame: np.ndarray) -> None:
        vis = PoseVisualizer()
        original = blank_frame.copy()
        vis.draw_expression(blank_frame, _make_expression())
        np.testing.assert_array_equal(blank_frame, original)

    def test_none_draws_something(self, blank_frame: np.ndarray) -> None:
        vis = PoseVisualizer()
        result = vis.draw_expression(blank_frame, None)
        assert not np.array_equal(result, blank_frame)

    @pytest.mark.parametrize(
        "emotion",
        ["happy", "surprise", "angry", "sad", "neutral", "unknown_emotion"],
    )
    def test_all_emotions_render(self, blank_frame: np.ndarray, emotion: str) -> None:
        vis = PoseVisualizer()
        expr = _make_expression(emotion=emotion, confidence=0.75)
        result = vis.draw_expression(blank_frame, expr)
        assert result.shape == blank_frame.shape
        assert not np.array_equal(result, blank_frame)


# ---------------------------------------------------------------------------
# PoseVisualizer.draw_fps
# ---------------------------------------------------------------------------


class TestDrawFps:
    def test_returns_ndarray(self, blank_frame: np.ndarray) -> None:
        vis = PoseVisualizer()
        result = vis.draw_fps(blank_frame, 30.0)
        assert isinstance(result, np.ndarray)

    def test_same_shape(self, blank_frame: np.ndarray) -> None:
        vis = PoseVisualizer()
        result = vis.draw_fps(blank_frame, 30.0)
        assert result.shape == blank_frame.shape

    def test_does_not_modify_original(self, blank_frame: np.ndarray) -> None:
        vis = PoseVisualizer()
        original = blank_frame.copy()
        vis.draw_fps(blank_frame, 30.0)
        np.testing.assert_array_equal(blank_frame, original)

    def test_text_rendered(self, blank_frame: np.ndarray) -> None:
        vis = PoseVisualizer()
        result = vis.draw_fps(blank_frame, 29.5)
        assert not np.array_equal(result, blank_frame)

    @pytest.mark.parametrize("fps", [0.0, 15.0, 30.0, 60.0, 120.5])
    def test_various_fps_values(self, blank_frame: np.ndarray, fps: float) -> None:
        vis = PoseVisualizer()
        result = vis.draw_fps(blank_frame, fps)
        assert result.shape == blank_frame.shape


# ---------------------------------------------------------------------------
# PoseVisualizer.draw_3d_info
# ---------------------------------------------------------------------------


class TestDraw3dInfo:
    def test_returns_ndarray(self, blank_frame: np.ndarray) -> None:
        vis = PoseVisualizer()
        kps = _make_keypoints_3d()
        result = vis.draw_3d_info(blank_frame, kps)
        assert isinstance(result, np.ndarray)

    def test_same_shape(self, blank_frame: np.ndarray) -> None:
        vis = PoseVisualizer()
        result = vis.draw_3d_info(blank_frame, _make_keypoints_3d())
        assert result.shape == blank_frame.shape

    def test_does_not_modify_original(self, blank_frame: np.ndarray) -> None:
        vis = PoseVisualizer()
        original = blank_frame.copy()
        vis.draw_3d_info(blank_frame, _make_keypoints_3d())
        np.testing.assert_array_equal(blank_frame, original)

    def test_renders_text(self, blank_frame: np.ndarray) -> None:
        vis = PoseVisualizer()
        result = vis.draw_3d_info(blank_frame, _make_keypoints_3d())
        assert not np.array_equal(result, blank_frame)

    def test_zero_depth_shows_dashes(self, blank_frame: np.ndarray) -> None:
        """Keypoints with z == 0.0 (missing depth) must render as dashes."""
        vis = PoseVisualizer()
        kps = _make_keypoints_3d(z=0.0)
        result = vis.draw_3d_info(blank_frame, kps)
        assert result.shape == blank_frame.shape
        assert not np.array_equal(result, blank_frame)

    def test_mixed_zero_and_valid_depth(self, blank_frame: np.ndarray) -> None:
        vis = PoseVisualizer()
        kps = _make_keypoints_3d()
        # Simulate missing depth for shoulder landmarks
        kps[11] = PoseKeypoint3D(x=0.0, y=0.0, z=0.0, visibility=0.0, name="left_shoulder")
        kps[12] = PoseKeypoint3D(x=0.0, y=0.0, z=0.0, visibility=0.0, name="right_shoulder")
        result = vis.draw_3d_info(blank_frame, kps)
        assert result.shape == blank_frame.shape

    def test_empty_list_handled(self, blank_frame: np.ndarray) -> None:
        vis = PoseVisualizer()
        result = vis.draw_3d_info(blank_frame, [])
        assert result.shape == blank_frame.shape


# ---------------------------------------------------------------------------
# PoseVisualizer.draw (composite)
# ---------------------------------------------------------------------------


class TestDraw:
    def _make_result(
        self,
        frame: np.ndarray,
        with_pose: bool = True,
        with_3d: bool = True,
        with_expr: bool = True,
    ) -> ProcessingResult:
        landmarks_2d: list[tuple[int, int]] | None = None
        keypoints_3d: list[PoseKeypoint3D] | None = None
        expression: ExpressionResult | None = None

        if with_pose:
            landmarks_2d = [(10 + i * 10, 100) for i in range(33)]
        if with_3d:
            keypoints_3d = _make_keypoints_3d()
        if with_expr:
            expression = _make_expression()

        return ProcessingResult(
            color_image=frame,
            landmarks_2d=landmarks_2d,
            keypoints_3d=keypoints_3d,
            hands=None,
            expression=expression,
            processing_fps=30.0,
            timestamp=0.0,
        )

    def test_returns_ndarray(self, blank_frame: np.ndarray) -> None:
        vis = PoseVisualizer()
        result = vis.draw(self._make_result(blank_frame))
        assert isinstance(result, np.ndarray)

    def test_same_shape(self, blank_frame: np.ndarray) -> None:
        vis = PoseVisualizer()
        result = vis.draw(self._make_result(blank_frame))
        assert result.shape == blank_frame.shape

    def test_does_not_modify_original_frame(self, blank_frame: np.ndarray) -> None:
        vis = PoseVisualizer()
        original = blank_frame.copy()
        result_obj = self._make_result(blank_frame)
        vis.draw(result_obj)
        np.testing.assert_array_equal(blank_frame, original)

    def test_no_pose_skips_skeleton(self, blank_frame: np.ndarray) -> None:
        vis = PoseVisualizer()
        result_obj = self._make_result(blank_frame, with_pose=False, with_3d=False)
        result = vis.draw(result_obj)
        # FPS and expression should still render
        assert not np.array_equal(result, blank_frame)

    def test_no_expression_no_crash(self, blank_frame: np.ndarray) -> None:
        vis = PoseVisualizer()
        result_obj = self._make_result(blank_frame, with_expr=False)
        result = vis.draw(result_obj)
        assert result.shape == blank_frame.shape

    def test_all_none_renders_fps_at_minimum(self, blank_frame: np.ndarray) -> None:
        """Even with no pose/expression data, FPS must still be drawn."""
        vis = PoseVisualizer()
        result_obj = ProcessingResult(
            color_image=blank_frame,
            landmarks_2d=None,
            keypoints_3d=None,
            hands=None,
            expression=None,
            processing_fps=25.0,
            timestamp=0.0,
        )
        result = vis.draw(result_obj)
        assert not np.array_equal(result, blank_frame)


# ---------------------------------------------------------------------------
# POSE_CONNECTIONS sanity checks
# ---------------------------------------------------------------------------


class TestPoseConnections:
    def test_all_indices_in_range(self) -> None:
        for start, end in POSE_CONNECTIONS:
            assert 0 <= start < 33, f"Invalid start index {start}"
            assert 0 <= end < 33, f"Invalid end index {end}"

    def test_no_self_loops(self) -> None:
        for start, end in POSE_CONNECTIONS:
            assert start != end, f"Self-loop at index {start}"

    def test_expected_connection_count(self) -> None:
        # 4+4 head + 1 mouth + 1 shoulders + 5+5 arms + 3 torso + 5+5 legs = 33
        assert len(POSE_CONNECTIONS) == 33


# ---------------------------------------------------------------------------
# HAND_CONNECTIONS sanity checks
# ---------------------------------------------------------------------------


class TestHandConnections:
    def test_all_indices_in_range(self) -> None:
        for start, end in HAND_CONNECTIONS:
            assert 0 <= start < 21, f"Invalid start index {start}"
            assert 0 <= end < 21, f"Invalid end index {end}"

    def test_no_self_loops(self) -> None:
        for start, end in HAND_CONNECTIONS:
            assert start != end, f"Self-loop at index {start}"


# ---------------------------------------------------------------------------
# PoseVisualizer.draw_hands
# ---------------------------------------------------------------------------


def _make_hand_result(
    handedness: str = "Left",
    n_landmarks: int = 21,
) -> HandResult:
    """Return a dummy HandResult for testing."""
    lm_2d = [(100 + i * 5, 200 + i * 3) for i in range(n_landmarks)]
    return HandResult(
        handedness=handedness,
        landmarks_2d=lm_2d,
        keypoints_3d=None,
        score=0.95,
    )


class TestDrawHands:
    def test_returns_ndarray(self, blank_frame: np.ndarray) -> None:
        vis = PoseVisualizer()
        hands = [_make_hand_result("Left")]
        result = vis.draw_hands(blank_frame, hands)
        assert isinstance(result, np.ndarray)

    def test_same_shape(self, blank_frame: np.ndarray) -> None:
        vis = PoseVisualizer()
        hands = [_make_hand_result("Left")]
        result = vis.draw_hands(blank_frame, hands)
        assert result.shape == blank_frame.shape

    def test_does_not_modify_original(self, blank_frame: np.ndarray) -> None:
        vis = PoseVisualizer()
        original = blank_frame.copy()
        vis.draw_hands(blank_frame, [_make_hand_result()])
        np.testing.assert_array_equal(blank_frame, original)

    def test_draws_on_canvas(self, blank_frame: np.ndarray) -> None:
        vis = PoseVisualizer()
        result = vis.draw_hands(blank_frame, [_make_hand_result()])
        assert not np.array_equal(result, blank_frame)

    def test_two_hands(self, blank_frame: np.ndarray) -> None:
        vis = PoseVisualizer()
        hands = [_make_hand_result("Left"), _make_hand_result("Right")]
        result = vis.draw_hands(blank_frame, hands)
        assert not np.array_equal(result, blank_frame)

    def test_empty_hands_returns_copy(self, blank_frame: np.ndarray) -> None:
        vis = PoseVisualizer()
        result = vis.draw_hands(blank_frame, [])
        np.testing.assert_array_equal(result, blank_frame)


# ---------------------------------------------------------------------------
# PoseVisualizer.draw_feature_status
# ---------------------------------------------------------------------------


class TestDrawFeatureStatus:
    def test_returns_ndarray(self, blank_frame: np.ndarray) -> None:
        vis = PoseVisualizer()
        flags = FeatureFlags()
        result = vis.draw_feature_status(blank_frame, flags)
        assert isinstance(result, np.ndarray)

    def test_same_shape(self, blank_frame: np.ndarray) -> None:
        vis = PoseVisualizer()
        flags = FeatureFlags()
        result = vis.draw_feature_status(blank_frame, flags)
        assert result.shape == blank_frame.shape

    def test_draws_text(self, blank_frame: np.ndarray) -> None:
        vis = PoseVisualizer()
        flags = FeatureFlags()
        result = vis.draw_feature_status(blank_frame, flags)
        assert not np.array_equal(result, blank_frame)

    def test_with_features_disabled(self, blank_frame: np.ndarray) -> None:
        vis = PoseVisualizer()
        flags = FeatureFlags()
        flags.pose.clear()
        flags.hand.clear()
        result = vis.draw_feature_status(blank_frame, flags)
        assert not np.array_equal(result, blank_frame)
