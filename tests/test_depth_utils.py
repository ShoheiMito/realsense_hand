"""Tests for depth_utils module."""

from __future__ import annotations

import math
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.config import OneEuroFilterParams
from src.depth_utils import (
    KeypointSmoother,
    OneEuroFilter,
    batch_deproject,
    deproject_landmarks,
    deproject_pixel_to_point,
    deproject_to_3d,
    get_depth_at_point,
    get_median_depth,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_intrinsics(fx: float = 600.0, fy: float = 600.0,
                     ppx: float = 320.0, ppy: float = 240.0) -> SimpleNamespace:
    """Return a minimal pinhole-compatible intrinsics object."""
    return SimpleNamespace(fx=fx, fy=fy, ppx=ppx, ppy=ppy)


def _make_depth_frame(image: np.ndarray, scale: float = 0.001) -> MagicMock:
    """Return a mock depth frame backed by the given uint16 image."""
    frame = MagicMock()
    frame.get_data.return_value = image
    frame.get_units.return_value = scale

    def get_distance(x: int, y: int) -> float:
        raw = int(image[y, x])
        return raw * scale

    frame.get_distance.side_effect = get_distance
    return frame


# ---------------------------------------------------------------------------
# get_median_depth
# ---------------------------------------------------------------------------


class TestGetMedianDepth:
    def _make_image(self, h: int = 480, w: int = 640,
                    fill: int = 1000) -> np.ndarray:
        return np.full((h, w), fill, dtype=np.uint16)

    def test_uniform_image_returns_fill_value(self) -> None:
        img = self._make_image(fill=2000)
        result = get_median_depth(img, x=320, y=240, kernel_size=5)
        assert result == pytest.approx(2000.0)

    def test_all_zeros_returns_zero(self) -> None:
        img = np.zeros((480, 640), dtype=np.uint16)
        result = get_median_depth(img, x=320, y=240, kernel_size=5)
        assert result == pytest.approx(0.0)

    def test_mixed_patch_ignores_zeros(self) -> None:
        img = np.zeros((480, 640), dtype=np.uint16)
        # Place non-zero values in a 5×5 patch around (100, 100)
        img[98:103, 98:103] = 3000
        img[100, 100] = 0  # hole in the centre
        result = get_median_depth(img, x=100, y=100, kernel_size=5)
        assert result == pytest.approx(3000.0)

    def test_near_border_does_not_raise(self) -> None:
        img = self._make_image(fill=500)
        # Top-left corner
        result = get_median_depth(img, x=0, y=0, kernel_size=5)
        assert result == pytest.approx(500.0)
        # Bottom-right corner
        result = get_median_depth(img, x=639, y=479, kernel_size=5)
        assert result == pytest.approx(500.0)

    def test_kernel_size_1_returns_single_pixel(self) -> None:
        img = np.zeros((480, 640), dtype=np.uint16)
        img[50, 50] = 1234
        result = get_median_depth(img, x=50, y=50, kernel_size=1)
        assert result == pytest.approx(1234.0)

    def test_returns_median_not_mean(self) -> None:
        img = np.zeros((480, 640), dtype=np.uint16)
        # 5×5 patch: 24 pixels = 100, 1 pixel = 9000 → median = 100
        img[98:103, 98:103] = 100
        img[100, 100] = 9000
        result = get_median_depth(img, x=100, y=100, kernel_size=5)
        assert result == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# get_depth_at_point
# ---------------------------------------------------------------------------


class TestGetDepthAtPoint:
    def test_valid_depth_returned_directly(self) -> None:
        img = np.full((480, 640), 2000, dtype=np.uint16)
        frame = _make_depth_frame(img, scale=0.001)
        result = get_depth_at_point(frame, px=100, py=100)
        assert result == pytest.approx(2.0)

    def test_zero_depth_falls_back_to_median(self) -> None:
        img = np.full((480, 640), 1500, dtype=np.uint16)
        img[100, 100] = 0  # hole
        frame = _make_depth_frame(img, scale=0.001)
        result = get_depth_at_point(frame, px=100, py=100, radius=5)
        # Fallback median of surrounding pixels (= 1500 raw → 1.5 m)
        assert result == pytest.approx(1.5)

    def test_all_zeros_returns_zero(self) -> None:
        img = np.zeros((480, 640), dtype=np.uint16)
        frame = _make_depth_frame(img, scale=0.001)
        result = get_depth_at_point(frame, px=50, py=50, radius=3)
        assert result == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# deproject_to_3d
# ---------------------------------------------------------------------------


class TestDeprojectTo3D:
    def test_centre_pixel_gives_zero_xy(self) -> None:
        intr = _make_intrinsics(fx=600.0, fy=600.0, ppx=320.0, ppy=240.0)
        result = deproject_to_3d(320, 240, 2.0, intr)
        assert result is not None
        x, y, z = result
        assert x == pytest.approx(0.0, abs=1e-9)
        assert y == pytest.approx(0.0, abs=1e-9)
        assert z == pytest.approx(2.0)

    def test_off_centre_pixel(self) -> None:
        intr = _make_intrinsics(fx=600.0, fy=600.0, ppx=320.0, ppy=240.0)
        # px=920 → (920-320)/600 * 2.0 = 2.0 m in X
        result = deproject_to_3d(920, 240, 2.0, intr)
        assert result is not None
        x, y, z = result
        assert x == pytest.approx(2.0)
        assert y == pytest.approx(0.0, abs=1e-9)
        assert z == pytest.approx(2.0)

    def test_zero_depth_returns_none(self) -> None:
        intr = _make_intrinsics()
        assert deproject_to_3d(100, 100, 0.0, intr) is None

    def test_negative_depth_returns_none(self) -> None:
        intr = _make_intrinsics()
        assert deproject_to_3d(100, 100, -1.0, intr) is None

    def test_float_pixel_coords(self) -> None:
        intr = _make_intrinsics(fx=500.0, fy=500.0, ppx=0.0, ppy=0.0)
        result = deproject_to_3d(1.0, 2.0, 5.0, intr)
        assert result is not None
        x, y, z = result
        assert x == pytest.approx(0.01)   # 1/500 * 5
        assert y == pytest.approx(0.02)   # 2/500 * 5
        assert z == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# deproject_pixel_to_point
# ---------------------------------------------------------------------------


class TestDeprojectPixelToPoint:
    def test_valid_depth(self) -> None:
        img = np.full((480, 640), 3000, dtype=np.uint16)
        frame = _make_depth_frame(img, scale=0.001)
        intr = _make_intrinsics(fx=600.0, fy=600.0, ppx=320.0, ppy=240.0)
        result = deproject_pixel_to_point(intr, (320, 240), frame, img)
        assert result is not None
        x, y, z = result
        assert x == pytest.approx(0.0, abs=1e-9)
        assert y == pytest.approx(0.0, abs=1e-9)
        assert z == pytest.approx(3.0)

    def test_missing_depth_uses_fallback(self) -> None:
        img = np.full((480, 640), 2000, dtype=np.uint16)
        img[100, 100] = 0  # hole
        frame = _make_depth_frame(img, scale=0.001)
        intr = _make_intrinsics(fx=600.0, fy=600.0, ppx=320.0, ppy=240.0)
        result = deproject_pixel_to_point(intr, (100, 100), frame, img)
        assert result is not None
        assert result[2] == pytest.approx(2.0)

    def test_entirely_missing_returns_none(self) -> None:
        img = np.zeros((480, 640), dtype=np.uint16)
        frame = _make_depth_frame(img, scale=0.001)
        intr = _make_intrinsics()
        result = deproject_pixel_to_point(intr, (320, 240), frame, img)
        assert result is None


# ---------------------------------------------------------------------------
# batch_deproject
# ---------------------------------------------------------------------------


class TestBatchDeproject:
    def test_all_valid(self) -> None:
        img = np.full((480, 640), 1000, dtype=np.uint16)
        frame = _make_depth_frame(img, scale=0.001)
        intr = _make_intrinsics(fx=600.0, fy=600.0, ppx=320.0, ppy=240.0)
        keypoints = [(320, 240), (320, 240), (320, 240)]
        results = batch_deproject(keypoints, frame, intr)
        assert len(results) == 3
        for pt in results:
            assert pt is not None
            assert pt[2] == pytest.approx(1.0)

    def test_missing_depth_gives_none(self) -> None:
        img = np.zeros((480, 640), dtype=np.uint16)
        frame = _make_depth_frame(img, scale=0.001)
        intr = _make_intrinsics()
        results = batch_deproject([(10, 10)], frame, intr)
        assert results == [None]

    def test_empty_keypoints(self) -> None:
        img = np.zeros((480, 640), dtype=np.uint16)
        frame = _make_depth_frame(img, scale=0.001)
        intr = _make_intrinsics()
        results = batch_deproject([], frame, intr)
        assert results == []

    def test_result_length_matches_input(self) -> None:
        img = np.full((480, 640), 500, dtype=np.uint16)
        frame = _make_depth_frame(img, scale=0.001)
        intr = _make_intrinsics()
        kps = [(i * 10, i * 10) for i in range(10)]
        results = batch_deproject(kps, frame, intr)
        assert len(results) == 10


# ---------------------------------------------------------------------------
# deproject_landmarks
# ---------------------------------------------------------------------------


class TestDeprojectLandmarks:
    def test_normalized_coords_converted_correctly(self) -> None:
        img = np.full((480, 640), 2000, dtype=np.uint16)
        frame = _make_depth_frame(img, scale=0.001)
        intr = _make_intrinsics(fx=600.0, fy=600.0, ppx=320.0, ppy=240.0)
        # Normalised (0.5, 0.5) → pixel (320, 240) → centre → x=0, y=0
        results = deproject_landmarks(
            intr, [(0.5, 0.5)], frame, img,
            image_width=640, image_height=480,
        )
        assert len(results) == 1
        assert results[0] is not None
        x, y, z = results[0]
        assert x == pytest.approx(0.0, abs=1e-6)
        assert y == pytest.approx(0.0, abs=1e-6)
        assert z == pytest.approx(2.0)

    def test_clamped_at_border(self) -> None:
        img = np.full((480, 640), 1000, dtype=np.uint16)
        frame = _make_depth_frame(img, scale=0.001)
        intr = _make_intrinsics()
        # Values outside [0,1] should be clamped, not raise
        results = deproject_landmarks(
            intr, [(2.0, 2.0)], frame, img,
            image_width=640, image_height=480,
        )
        assert len(results) == 1  # no exception

    def test_empty_landmarks(self) -> None:
        img = np.zeros((480, 640), dtype=np.uint16)
        frame = _make_depth_frame(img, scale=0.001)
        intr = _make_intrinsics()
        results = deproject_landmarks(intr, [], frame, img, 640, 480)
        assert results == []


# ---------------------------------------------------------------------------
# OneEuroFilter
# ---------------------------------------------------------------------------


class TestOneEuroFilter:
    def test_first_call_returns_input(self) -> None:
        f = OneEuroFilter()
        assert f(1.0, 0.0) == pytest.approx(1.0)

    def test_stable_signal_converges(self) -> None:
        f = OneEuroFilter()
        t = 0.0
        val = 0.0
        for _ in range(200):
            t += 1 / 30.0
            val = f(1.0, t)
        assert val == pytest.approx(1.0, abs=1e-3)

    def test_smoothing_reduces_noise(self) -> None:
        rng = np.random.default_rng(42)
        f = OneEuroFilter()
        t = 0.0
        raw_vals: list[float] = []
        filtered_vals: list[float] = []
        for _ in range(100):
            t += 1 / 30.0
            raw = 1.0 + float(rng.normal(0, 0.2))
            raw_vals.append(raw)
            filtered_vals.append(f(raw, t))
        # Variance of filtered signal should be less than raw
        assert float(np.var(filtered_vals)) < float(np.var(raw_vals))

    def test_tracks_slow_ramp(self) -> None:
        f = OneEuroFilter(OneEuroFilterParams(min_cutoff=1.0, beta=0.0, d_cutoff=1.0))
        t = 0.0
        for i in range(300):
            t += 1 / 30.0
            out = f(float(i) / 300.0, t)
        # After 300 steps the output should be close to the final ramp value (≈1.0)
        assert out == pytest.approx(1.0, abs=0.05)

    def test_same_timestamp_returns_previous(self) -> None:
        f = OneEuroFilter()
        first = f(5.0, 1.0)
        # dt=0 → should return previous value without dividing by zero
        second = f(10.0, 1.0)
        assert second == pytest.approx(first)

    def test_custom_params(self) -> None:
        params = OneEuroFilterParams(min_cutoff=0.1, beta=0.5, d_cutoff=2.0)
        f = OneEuroFilter(params)
        assert f(0.0, 0.0) == pytest.approx(0.0)

    def test_alpha_property(self) -> None:
        # alpha(cutoff=1/(2π), dt=1) should be 0.5
        alpha = OneEuroFilter._alpha(1.0 / (2.0 * math.pi), 1.0)
        assert alpha == pytest.approx(0.5, rel=1e-6)

    def test_default_params_used_when_none(self) -> None:
        f = OneEuroFilter(None)
        assert f(3.14, 0.0) == pytest.approx(3.14)


# ---------------------------------------------------------------------------
# KeypointSmoother
# ---------------------------------------------------------------------------


class TestKeypointSmoother:
    def test_output_length_matches_input(self) -> None:
        smoother = KeypointSmoother(num_keypoints=33)
        kps: list[tuple[float, float, float] | None] = [(0.1, 0.2, 1.0)] * 33
        result = smoother.smooth(kps, timestamp=0.0)
        assert len(result) == 33

    def test_none_entries_pass_through(self) -> None:
        smoother = KeypointSmoother(num_keypoints=5)
        kps: list[tuple[float, float, float] | None] = [None] * 5
        result = smoother.smooth(kps, timestamp=0.0)
        assert all(r is None for r in result)

    def test_mixed_none_and_valid(self) -> None:
        smoother = KeypointSmoother(num_keypoints=3)
        kps: list[tuple[float, float, float] | None] = [
            (1.0, 2.0, 3.0), None, (4.0, 5.0, 6.0)
        ]
        result = smoother.smooth(kps, timestamp=0.0)
        assert result[0] is not None
        assert result[1] is None
        assert result[2] is not None

    def test_first_call_returns_input_unchanged(self) -> None:
        smoother = KeypointSmoother(num_keypoints=2)
        kps: list[tuple[float, float, float] | None] = [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)]
        result = smoother.smooth(kps, timestamp=0.0)
        assert result[0] == pytest.approx((1.0, 2.0, 3.0))
        assert result[1] == pytest.approx((4.0, 5.0, 6.0))

    def test_smoothing_over_multiple_frames(self) -> None:
        smoother = KeypointSmoother(num_keypoints=1)
        t = 0.0
        for _ in range(50):
            t += 1 / 30.0
            result = smoother.smooth([(1.0, 1.0, 1.0)], timestamp=t)
        assert result[0] is not None
        x, y, z = result[0]
        assert x == pytest.approx(1.0, abs=0.01)
        assert y == pytest.approx(1.0, abs=0.01)
        assert z == pytest.approx(1.0, abs=0.01)

    def test_independent_filters_per_keypoint(self) -> None:
        """Each keypoint has its own filter history."""
        smoother = KeypointSmoother(num_keypoints=2)
        t = 0.0
        for _ in range(10):
            t += 1 / 30.0
            smoother.smooth([(0.0, 0.0, 0.0), (10.0, 10.0, 10.0)], timestamp=t)
        result = smoother.smooth([(0.0, 0.0, 0.0), (10.0, 10.0, 10.0)], timestamp=t + 1 / 30.0)
        # The two keypoints should converge to their own target values
        assert result[0] is not None
        assert result[1] is not None
        assert result[0][0] < result[1][0]
