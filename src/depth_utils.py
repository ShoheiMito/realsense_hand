"""Depth processing utilities for RealSense pose estimation.

Provides depth filter setup, 2D→3D deprojection, median fallback,
and One Euro Filter for temporal smoothing of 3D keypoints.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

try:
    import pyrealsense2 as rs  # type: ignore[import-untyped]

    RS_AVAILABLE = True
except ImportError:
    rs = None  # type: ignore[assignment]
    RS_AVAILABLE = False

from . import config
from .config import DEPTH_FALLBACK_KERNEL_SIZE, OneEuroFilterParams


# ---------------------------------------------------------------------------
# Filter chain
# ---------------------------------------------------------------------------


@dataclass
class DepthFilters:
    """Container for the RealSense depth filter chain."""

    spatial: Any
    temporal: Any
    hole_filling: Any


def setup_depth_filters() -> DepthFilters:
    """Initialize the depth filter chain: spatial → temporal → hole_filling.

    Returns:
        DepthFilters: Initialized filter objects ready for reuse each frame.

    Raises:
        RuntimeError: If pyrealsense2 is not installed.
    """
    if not RS_AVAILABLE:
        raise RuntimeError("pyrealsense2 is not installed")

    spatial = rs.spatial_filter()
    spatial.set_option(rs.option.filter_magnitude, config.SPATIAL_FILTER_MAGNITUDE)
    spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
    spatial.set_option(rs.option.filter_smooth_delta, 20)

    temporal = rs.temporal_filter()
    temporal.set_option(rs.option.filter_smooth_alpha, 0.4)
    temporal.set_option(rs.option.filter_smooth_delta, 20)

    hole_filling = rs.hole_filling_filter()
    hole_filling.set_option(rs.option.holes_fill, 1)

    return DepthFilters(
        spatial=spatial,
        temporal=temporal,
        hole_filling=hole_filling,
    )


def filter_depth_frame(depth_frame: Any, filters: DepthFilters) -> Any:
    """Apply the filter chain (spatial → temporal → hole_filling) to a depth frame.

    Args:
        depth_frame: Raw rs.depth_frame from the RealSense pipeline.
        filters: DepthFilters returned by setup_depth_filters().

    Returns:
        Filtered rs.depth_frame.
    """
    filtered = filters.spatial.process(depth_frame)
    filtered = filters.temporal.process(filtered)
    filtered = filters.hole_filling.process(filtered)
    return filtered


# ---------------------------------------------------------------------------
# Depth sampling
# ---------------------------------------------------------------------------


def get_median_depth(
    depth_image: np.ndarray,
    x: int,
    y: int,
    kernel_size: int = DEPTH_FALLBACK_KERNEL_SIZE,
) -> float:
    """Return the median depth (in raw uint16 units) around a pixel.

    Args:
        depth_image: Depth image array with shape (H, W), dtype uint16.
        x: Column index (pixel x-coordinate).
        y: Row index (pixel y-coordinate).
        kernel_size: Side length of the neighbourhood square (should be odd).

    Returns:
        Median of non-zero values in the neighbourhood, or 0.0 if none exist.
    """
    h, w = depth_image.shape
    half = kernel_size // 2
    y0 = max(0, y - half)
    y1 = min(h, y + half + 1)
    x0 = max(0, x - half)
    x1 = min(w, x + half + 1)

    patch = depth_image[y0:y1, x0:x1]
    valid = patch[patch > 0]
    if valid.size == 0:
        return 0.0
    return float(np.median(valid))


def get_depth_at_point(
    depth_frame: Any,
    px: int,
    py: int,
    radius: int = 5,
) -> float:
    """Return the depth in metres at (px, py), falling back to neighbourhood median.

    Tries rs.depth_frame.get_distance() first.  If the value is 0 (missing),
    computes the median of a (2*radius+1) × (2*radius+1) neighbourhood from the
    underlying numpy array.

    Args:
        depth_frame: rs.depth_frame (or any object with get_distance() and
            a numpy-representable backing array).
        px: Pixel x-coordinate (column).
        py: Pixel y-coordinate (row).
        radius: Half-width of the fallback neighbourhood window.

    Returns:
        Depth in metres, or 0.0 if no valid depth is available.
    """
    depth_m: float = depth_frame.get_distance(px, py)
    if depth_m > 0.0:
        return depth_m

    # Fallback: median of neighbourhood in raw uint16 array
    depth_image: np.ndarray = np.asanyarray(depth_frame.get_data())
    depth_scale: float = 1.0
    if hasattr(depth_frame, "get_units"):
        depth_scale = depth_frame.get_units()

    kernel_size = 2 * radius + 1
    raw_median = get_median_depth(depth_image, px, py, kernel_size)
    if raw_median == 0.0:
        return 0.0
    # Raw values are in depth_scale metres (typically 0.001 for millimetres)
    return raw_median * depth_scale


# ---------------------------------------------------------------------------
# Deprojection
# ---------------------------------------------------------------------------


def deproject_to_3d(
    px: int | float,
    py: int | float,
    depth_m: float,
    intrinsics: Any,
) -> tuple[float, float, float] | None:
    """Convert a 2D pixel + depth to a 3D world coordinate.

    Uses the pinhole camera model (or the distortion model embedded in
    rs.intrinsics when pyrealsense2 is available).

    Args:
        px: Pixel x-coordinate (column).
        py: Pixel y-coordinate (row).
        depth_m: Depth in metres.
        intrinsics: rs.intrinsics object, or any object with attributes
            ``fx``, ``fy``, ``ppx``, ``ppy`` for the pure-Python fallback.

    Returns:
        (x, y, z) in metres, or None if depth_m is 0.
    """
    if depth_m <= 0.0:
        return None

    if RS_AVAILABLE and isinstance(intrinsics, rs.intrinsics):
        point = rs.rs2_deproject_pixel_to_point(intrinsics, [px, py], depth_m)
        return (float(point[0]), float(point[1]), float(point[2]))

    # Pure-Python pinhole fallback (no distortion correction)
    x = (px - intrinsics.ppx) / intrinsics.fx * depth_m
    y = (py - intrinsics.ppy) / intrinsics.fy * depth_m
    return (float(x), float(y), float(depth_m))


def batch_deproject(
    keypoints_2d: list[tuple[int, int]],
    depth_frame: Any,
    intrinsics: Any,
    radius: int = 5,
) -> list[tuple[float, float, float] | None]:
    """Convert all 2D keypoints to 3D world coordinates in one call.

    For each keypoint the depth is sampled with get_depth_at_point() which
    applies the median-fallback automatically.

    Args:
        keypoints_2d: List of (px, py) pixel coordinates.
        depth_frame: rs.depth_frame used for depth sampling.
        intrinsics: Camera intrinsics (rs.intrinsics or pinhole-compatible).
        radius: Fallback neighbourhood radius passed to get_depth_at_point().

    Returns:
        List of (x, y, z) tuples in metres; None where depth is unavailable.
    """
    results: list[tuple[float, float, float] | None] = []
    for px, py in keypoints_2d:
        depth_m = get_depth_at_point(depth_frame, px, py, radius)
        results.append(deproject_to_3d(px, py, depth_m, intrinsics))
    return results


def deproject_pixel_to_point(
    intrinsics: Any,
    pixel: tuple[int, int],
    depth_frame: Any,
    depth_image: np.ndarray,
    fallback_kernel: int = DEPTH_FALLBACK_KERNEL_SIZE,
) -> tuple[float, float, float] | None:
    """Convert a pixel coordinate to a 3D world coordinate.

    If the depth at the pixel is 0, a fallback_kernel × fallback_kernel
    neighbourhood median is used.

    Args:
        intrinsics: rs.intrinsics or pinhole-compatible object.
        pixel: (x, y) pixel coordinate.
        depth_frame: rs.depth_frame for get_distance().
        depth_image: Depth image as numpy array (H, W) uint16.
        fallback_kernel: Neighbourhood size for the median fallback.

    Returns:
        (x, y, z) in metres, or None if no valid depth is found.
    """
    px, py = pixel
    depth_m: float = depth_frame.get_distance(px, py)

    if depth_m <= 0.0:
        depth_scale: float = 1.0
        if hasattr(depth_frame, "get_units"):
            depth_scale = depth_frame.get_units()
        raw = get_median_depth(depth_image, px, py, fallback_kernel)
        depth_m = raw * depth_scale

    return deproject_to_3d(px, py, depth_m, intrinsics)


def deproject_landmarks(
    intrinsics: Any,
    landmarks: list[tuple[float, float]],
    depth_frame: Any,
    depth_image: np.ndarray,
    image_width: int,
    image_height: int,
) -> list[tuple[float, float, float] | None]:
    """Convert normalized landmark coordinates to 3D world coordinates.

    Args:
        intrinsics: Camera intrinsics.
        landmarks: Normalized coordinates [(nx, ny), ...] in [0, 1].
        depth_frame: rs.depth_frame.
        depth_image: Depth image (H, W) uint16.
        image_width: Image width in pixels.
        image_height: Image height in pixels.

    Returns:
        List of (x, y, z) or None for each landmark.
    """
    results: list[tuple[float, float, float] | None] = []
    for nx, ny in landmarks:
        px = int(nx * image_width)
        py = int(ny * image_height)
        px = max(0, min(image_width - 1, px))
        py = max(0, min(image_height - 1, py))
        results.append(
            deproject_pixel_to_point(intrinsics, (px, py), depth_frame, depth_image)
        )
    return results


# ---------------------------------------------------------------------------
# One Euro Filter
# ---------------------------------------------------------------------------


class OneEuroFilter:
    """1D One Euro Filter for temporal smoothing of noisy signal streams.

    Reference: https://cristal.univ-lille.fr/~casiez/1euro/
    """

    def __init__(self, params: OneEuroFilterParams | None = None) -> None:
        """Initialize the filter with the given parameters.

        Args:
            params: OneEuroFilterParams; uses defaults if None.
        """
        self._params = params if params is not None else OneEuroFilterParams()
        self._prev_x: float | None = None
        self._prev_dx: float = 0.0
        self._prev_t: float | None = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        """Compute the smoothing factor for the low-pass filter step.

        Args:
            cutoff: Cutoff frequency in Hz.
            dt: Time step in seconds.

        Returns:
            Smoothing factor alpha in (0, 1].
        """
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x: float, timestamp: float) -> float:
        """Apply the filter and return the smoothed value.

        Args:
            x: Raw input value.
            timestamp: Current time in seconds (monotonically increasing).

        Returns:
            Smoothed value.
        """
        if self._prev_t is None:
            self._prev_t = timestamp
            self._prev_x = x
            return x

        dt = timestamp - self._prev_t
        if dt <= 0.0:
            return self._prev_x if self._prev_x is not None else x

        prev_x = self._prev_x if self._prev_x is not None else x

        # Estimate derivative
        dx = (x - prev_x) / dt

        # Low-pass filter the derivative
        alpha_d = self._alpha(self._params.d_cutoff, dt)
        dx_filtered = alpha_d * dx + (1.0 - alpha_d) * self._prev_dx

        # Adaptive cutoff: higher speed → higher cutoff → less lag
        cutoff = self._params.min_cutoff + self._params.beta * abs(dx_filtered)

        # Low-pass filter the signal
        alpha = self._alpha(cutoff, dt)
        x_filtered = alpha * x + (1.0 - alpha) * prev_x

        self._prev_x = x_filtered
        self._prev_dx = dx_filtered
        self._prev_t = timestamp

        return x_filtered


# ---------------------------------------------------------------------------
# Keypoint smoother
# ---------------------------------------------------------------------------


@dataclass
class KeypointSmoother:
    """Apply One Euro Filters to all axes of a set of 3D keypoints.

    Maintains 33 × 3 = 99 independent filter instances (one per axis per joint).
    """

    num_keypoints: int = 33
    _filters: list[list[OneEuroFilter]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        params = OneEuroFilterParams()
        self._filters = [
            [OneEuroFilter(params) for _ in range(3)]
            for _ in range(self.num_keypoints)
        ]

    def smooth(
        self,
        keypoints_3d: list[tuple[float, float, float] | None],
        timestamp: float,
    ) -> list[tuple[float, float, float] | None]:
        """Apply smoothing to every keypoint.

        Args:
            keypoints_3d: Raw 3D coordinates; None entries are passed through.
            timestamp: Current time in seconds.

        Returns:
            Smoothed 3D coordinates.  None entries remain None.
        """
        result: list[tuple[float, float, float] | None] = []
        for i, kp in enumerate(keypoints_3d):
            if kp is None:
                result.append(None)
                continue
            filters = self._filters[i]
            sx = filters[0](kp[0], timestamp)
            sy = filters[1](kp[1], timestamp)
            sz = filters[2](kp[2], timestamp)
            result.append((sx, sy, sz))
        return result
