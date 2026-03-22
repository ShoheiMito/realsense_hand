"""Tests for hand gesture recognition and mouse control."""

from __future__ import annotations

import sys
import time
import types
from unittest.mock import MagicMock, patch

import pytest

# Mock pynput before importing hand_controller (pynput may not be installed)
_mock_pynput = types.ModuleType("pynput")
_mock_pynput_mouse = types.ModuleType("pynput.mouse")
_mock_pynput_mouse.Button = MagicMock()  # type: ignore[attr-defined]
_mock_pynput_mouse.Button.left = "left"  # type: ignore[attr-defined]
_mock_pynput_mouse.Controller = MagicMock  # type: ignore[attr-defined]
_mock_pynput.mouse = _mock_pynput_mouse  # type: ignore[attr-defined]

if "pynput" not in sys.modules:
    sys.modules["pynput"] = _mock_pynput
    sys.modules["pynput.mouse"] = _mock_pynput_mouse

from src.hand_controller import (  # noqa: E402
    CoordinateMapper,
    GestureDetector,
    GestureState,
    HandController,
)
from src.processor import HandResult, PoseKeypoint3D  # noqa: E402


# ---------------------------------------------------------------------------
# Test fixtures — realistic hand landmark data
# ---------------------------------------------------------------------------


def _make_open_hand(cx: int = 320, cy: int = 240) -> list[tuple[int, int]]:
    """Generate 21 landmarks for an open hand (all fingers extended).

    Wrist at (cx, cy+80), fingers spread above.
    """
    return [
        (cx, cy + 80),       # 0: wrist
        (cx - 40, cy + 50),  # 1: thumb_cmc
        (cx - 60, cy + 20),  # 2: thumb_mcp
        (cx - 75, cy - 10),  # 3: thumb_ip
        (cx - 85, cy - 30),  # 4: thumb_tip (遠めに配置)
        (cx - 15, cy + 20),  # 5: index_mcp
        (cx - 15, cy - 10),  # 6: index_pip
        (cx - 15, cy - 30),  # 7: index_dip
        (cx - 15, cy - 50),  # 8: index_tip
        (cx, cy + 15),       # 9: middle_mcp
        (cx, cy - 15),       # 10: middle_pip
        (cx, cy - 35),       # 11: middle_dip
        (cx, cy - 55),       # 12: middle_tip
        (cx + 15, cy + 20),  # 13: ring_mcp
        (cx + 15, cy - 5),   # 14: ring_pip
        (cx + 15, cy - 25),  # 15: ring_dip
        (cx + 15, cy - 40),  # 16: ring_tip
        (cx + 30, cy + 25),  # 17: pinky_mcp
        (cx + 30, cy + 5),   # 18: pinky_pip
        (cx + 30, cy - 10),  # 19: pinky_dip
        (cx + 30, cy - 20),  # 20: pinky_tip
    ]


def _make_pinch_hand(cx: int = 320, cy: int = 240) -> list[tuple[int, int]]:
    """Generate landmarks with thumb and index finger pinching."""
    lm = _make_open_hand(cx, cy)
    # Move thumb tip close to index tip
    lm[4] = (lm[8][0] + 5, lm[8][1] + 5)
    return lm


def _make_scroll_hand(cx: int = 320, cy: int = 240) -> list[tuple[int, int]]:
    """Generate landmarks for scroll gesture (index + middle up, others down)."""
    lm = _make_open_hand(cx, cy)
    # Fold ring and pinky (tips below pip)
    lm[16] = (lm[14][0], lm[14][1] + 30)  # ring tip below pip
    lm[20] = (lm[18][0], lm[18][1] + 30)  # pinky tip below pip
    return lm


def _make_pointing_hand(cx: int = 320, cy: int = 240) -> list[tuple[int, int]]:
    """Generate landmarks for pointing pose (index only extended)."""
    lm = _make_open_hand(cx, cy)
    # Fold middle, ring, pinky (tips below pip)
    lm[12] = (lm[10][0], lm[10][1] + 30)  # middle tip below pip
    lm[16] = (lm[14][0], lm[14][1] + 30)  # ring tip below pip
    lm[20] = (lm[18][0], lm[18][1] + 30)  # pinky tip below pip
    return lm


def _make_3d_keypoints(
    thumb_tip: tuple[float, float, float] = (0.05, 0.0, 0.5),
    index_tip: tuple[float, float, float] = (-0.02, -0.08, 0.5),
) -> list[PoseKeypoint3D]:
    """Generate 21 dummy 3D keypoints with configurable thumb/index tips."""
    kps = [
        PoseKeypoint3D(x=0.0, y=0.0, z=0.5, visibility=1.0, name=f"lm_{i}")
        for i in range(21)
    ]
    kps[4] = PoseKeypoint3D(
        x=thumb_tip[0], y=thumb_tip[1], z=thumb_tip[2],
        visibility=1.0, name="thumb_tip",
    )
    kps[8] = PoseKeypoint3D(
        x=index_tip[0], y=index_tip[1], z=index_tip[2],
        visibility=1.0, name="index_finger_tip",
    )
    return kps


def _make_hand_result(
    landmarks: list[tuple[int, int]] | None = None,
    handedness: str = "Right",
    score: float = 0.9,
    keypoints_3d: list[PoseKeypoint3D] | None = None,
) -> HandResult:
    if landmarks is None:
        landmarks = _make_open_hand()
    return HandResult(
        handedness=handedness,
        landmarks_2d=landmarks,
        keypoints_3d=keypoints_3d,
        score=score,
    )


# ===========================================================================
# GestureDetector tests
# ===========================================================================


class TestGestureDetector:
    """Tests for stateless gesture feature extraction."""

    def test_pinch_detected_when_close(self) -> None:
        lm = _make_pinch_hand()
        dist = GestureDetector.pinch_distance(lm)
        assert dist < 30  # Within pinch threshold

    def test_pinch_not_detected_when_far(self) -> None:
        lm = _make_open_hand()
        dist = GestureDetector.pinch_distance(lm)
        assert dist > 30  # Beyond pinch threshold

    def test_finger_extended(self) -> None:
        lm = _make_open_hand()
        # Index tip (8) should be above pip (6)
        assert GestureDetector.is_finger_extended(lm, 8, 6) is True

    def test_finger_not_extended(self) -> None:
        lm = _make_open_hand()
        # Fold index: tip below pip
        lm[8] = (lm[6][0], lm[6][1] + 30)
        assert GestureDetector.is_finger_extended(lm, 8, 6) is False

    def test_scroll_pose_detected(self) -> None:
        lm = _make_scroll_hand()
        assert GestureDetector.is_scroll_pose(lm) is True

    def test_scroll_pose_not_detected_all_fingers_extended(self) -> None:
        lm = _make_open_hand()
        # All fingers extended — ring is also up, so not scroll
        assert GestureDetector.is_scroll_pose(lm) is False

    def test_get_index_tip(self) -> None:
        lm = _make_open_hand(300, 200)
        tip = GestureDetector.get_index_tip(lm)
        assert tip == lm[8]

    def test_get_scroll_center(self) -> None:
        lm = _make_open_hand()
        center = GestureDetector.get_scroll_center(lm)
        expected_x = (lm[8][0] + lm[12][0]) // 2
        expected_y = (lm[8][1] + lm[12][1]) // 2
        assert center == (expected_x, expected_y)

    def test_pointing_pose_detected(self) -> None:
        lm = _make_pointing_hand()
        assert GestureDetector.is_pointing_pose(lm) is True

    def test_pointing_pose_not_detected_open_hand(self) -> None:
        lm = _make_open_hand()
        assert GestureDetector.is_pointing_pose(lm) is False

    def test_open_hand_detected(self) -> None:
        lm = _make_open_hand()
        assert GestureDetector.is_open_hand(lm) is True

    def test_open_hand_not_detected_pointing(self) -> None:
        lm = _make_pointing_hand()
        assert GestureDetector.is_open_hand(lm) is False

    def test_pinch_distance_3d(self) -> None:
        kps = _make_3d_keypoints(
            thumb_tip=(0.0, 0.0, 0.5),
            index_tip=(0.01, 0.01, 0.5),
        )
        dist = GestureDetector.pinch_distance_3d(kps)
        assert dist is not None
        assert dist < 0.025  # Within 25mm threshold

    def test_pinch_distance_3d_far(self) -> None:
        kps = _make_3d_keypoints(
            thumb_tip=(0.05, 0.0, 0.5),
            index_tip=(-0.02, -0.08, 0.5),
        )
        dist = GestureDetector.pinch_distance_3d(kps)
        assert dist is not None
        assert dist > 0.035  # Beyond release threshold

    def test_pinch_distance_3d_returns_none_without_data(self) -> None:
        assert GestureDetector.pinch_distance_3d(None) is None


# ===========================================================================
# CoordinateMapper tests
# ===========================================================================


class TestCoordinateMapper:
    """Tests for camera-to-screen coordinate mapping."""

    def _make_mapper(self, **kwargs: object) -> CoordinateMapper:
        """Create a mapper with mocked screen resolution."""
        with patch("src.hand_controller.ctypes") as mock_ctypes:
            user32 = MagicMock()
            user32.GetSystemMetrics.side_effect = lambda x: 1920 if x == 0 else 1080
            mock_ctypes.windll.user32 = user32
            defaults = {
                "camera_width": 640,
                "camera_height": 480,
                "active_region": 0.7,
                "deadzone_px": 0,
                "mirror_x": False,
            }
            defaults.update(kwargs)
            return CoordinateMapper(**defaults)  # type: ignore[arg-type]

    def test_center_maps_to_screen_center(self) -> None:
        mapper = self._make_mapper()
        sx, sy = mapper.map(320, 240)
        # Center of camera → center of screen (approximately)
        assert abs(sx - 960) < 50
        assert abs(sy - 540) < 50

    def test_active_region_boundary_maps_to_edge(self) -> None:
        mapper = self._make_mapper()
        # Left boundary of active region (0.15 * 640 = 96)
        sx, _ = mapper.map(96, 240)
        assert sx < 50  # Near left edge

    def test_clamp_outside_active_region(self) -> None:
        mapper = self._make_mapper()
        # Far outside active region — should clamp
        sx1, _ = mapper.map(0, 240)
        sx2, _ = mapper.map(96, 240)
        assert sx1 == sx2  # Both clamp to the same boundary

    def test_mirror_x(self) -> None:
        mapper_no_mirror = self._make_mapper(mirror_x=False)
        mapper_mirror = self._make_mapper(mirror_x=True)

        sx_no, _ = mapper_no_mirror.map(200, 240)
        sx_yes, _ = mapper_mirror.map(200, 240)

        # Mirrored should be on opposite side
        assert sx_no < 960  # Left side without mirror
        assert sx_yes > 960  # Right side with mirror

    def test_deadzone_prevents_micro_movement(self) -> None:
        mapper = self._make_mapper(deadzone_px=10)
        pos1 = mapper.map(320, 240)
        pos2 = mapper.map(321, 240)  # Very small movement
        assert pos1 == pos2  # Should not change due to deadzone

    def test_one_euro_smoothing_reduces_jitter(self) -> None:
        mapper = self._make_mapper(deadzone_px=0)
        t = 0.0
        # Initial position
        mapper.map(320, 240, timestamp=t)
        t += 0.033  # ~30fps
        # Jump to a far position — One Euro should lag behind
        sx, _ = mapper.map(500, 240, timestamp=t)
        # With One Euro, smoothed position should not reach target immediately
        mapper2 = self._make_mapper(deadzone_px=0)
        # Feed same position repeatedly to converge
        for i in range(30):
            target_sx, _ = mapper2.map(500, 240, timestamp=i * 0.033)
        # First frame after jump should be less than converged value
        assert sx < target_sx

    def test_reset_clears_state(self) -> None:
        mapper = self._make_mapper(deadzone_px=0)
        mapper.map(100, 100)
        mapper.reset()
        # After reset, should act as fresh initialization
        assert mapper._initialized is False


# ===========================================================================
# HandController state machine tests
# ===========================================================================


class TestHandController:
    """Tests for the gesture state machine + mouse control."""

    @pytest.fixture()
    def controller(self) -> HandController:
        """Create a HandController with mocked pynput and ctypes."""
        with patch("src.hand_controller.ctypes") as mock_ctypes:
            user32 = MagicMock()
            user32.GetSystemMetrics.side_effect = lambda x: 1920 if x == 0 else 1080
            mock_ctypes.windll.user32 = user32

            ctrl = HandController(preferred_hand="Right")
            # Replace the mouse controller with a fresh mock
            ctrl._mouse = MagicMock()
            return ctrl

    def _enter_cursor(self, controller: HandController) -> None:
        """Helper: advance controller from IDLE → NEUTRAL → CURSOR."""
        hand_open = _make_hand_result(_make_open_hand())
        hand_point = _make_hand_result(_make_pointing_hand())
        # IDLE → NEUTRAL (debounce)
        for _ in range(3):
            controller.update([hand_open])
        # NEUTRAL → CURSOR (pointing)
        controller.update([hand_point])

    def test_idle_to_neutral_on_hand_detect(self, controller: HandController) -> None:
        """After GESTURE_CONFIRM_FRAMES with hand, transition IDLE→NEUTRAL."""
        hand = _make_hand_result()
        for _ in range(3):
            info = controller.update([hand])
        assert info.state == GestureState.NEUTRAL

    def test_neutral_to_cursor_on_pointing(self, controller: HandController) -> None:
        """Pointing pose in NEUTRAL should transition to CURSOR."""
        hand_open = _make_hand_result(_make_open_hand())
        hand_point = _make_hand_result(_make_pointing_hand())
        # IDLE → NEUTRAL
        for _ in range(3):
            controller.update([hand_open])
        # NEUTRAL → CURSOR
        info = controller.update([hand_point])
        assert info.state == GestureState.CURSOR

    def test_cursor_to_neutral_on_open_hand(self, controller: HandController) -> None:
        """Opening hand in CURSOR for GESTURE_CONFIRM_FRAMES should go to NEUTRAL."""
        self._enter_cursor(controller)
        hand_open = _make_hand_result(_make_open_hand())
        # デバウンス: GESTURE_CONFIRM_FRAMES(3)フレーム必要
        for _ in range(3):
            info = controller.update([hand_open])
        assert info.state == GestureState.NEUTRAL

    def test_cursor_to_idle_on_hand_lost(self, controller: HandController) -> None:
        """After HAND_LOST_FRAMES without hand, transition to IDLE."""
        self._enter_cursor(controller)
        # Then lose hand
        for _ in range(6):
            info = controller.update(None)
        assert info.state == GestureState.IDLE

    def test_click_fires_on_quick_pinch(self, controller: HandController) -> None:
        """Quick pinch and release should fire mouse.click()."""
        self._enter_cursor(controller)
        hand_pinch = _make_hand_result(_make_pinch_hand())
        hand_point = _make_hand_result(_make_pointing_hand())

        # Pinch
        controller.update([hand_pinch])
        assert controller._state == GestureState.CLICK_DOWN

        # Quick release — clutch mode returns to NEUTRAL
        controller.update([hand_point])
        assert controller._state == GestureState.NEUTRAL
        controller._mouse.click.assert_called()

    def test_click_from_neutral_via_pinch(self, controller: HandController) -> None:
        """Pinch directly from NEUTRAL should fire click on release."""
        hand_open = _make_hand_result(_make_open_hand())
        hand_pinch = _make_hand_result(_make_pinch_hand())

        # IDLE → NEUTRAL
        for _ in range(3):
            controller.update([hand_open])
        assert controller._state == GestureState.NEUTRAL

        # Pinch from NEUTRAL → CLICK_DOWN
        controller.update([hand_pinch])
        assert controller._state == GestureState.CLICK_DOWN

        # Release → NEUTRAL
        controller.update([hand_open])
        assert controller._state == GestureState.NEUTRAL
        controller._mouse.click.assert_called()

    def test_drag_on_sustained_pinch_with_movement(self, controller: HandController) -> None:
        """Sustained pinch with movement should transition to DRAGGING."""
        self._enter_cursor(controller)

        # Pinch at one position
        pinch1 = _make_pinch_hand(320, 240)
        controller.update([_make_hand_result(pinch1)])

        # Hold pinch long enough to exceed DRAG_MIN_HOLD_S
        time.sleep(0.3)
        pinch2 = _make_pinch_hand(320, 240)
        controller.update([_make_hand_result(pinch2)])

        assert controller._state == GestureState.DRAGGING
        controller._mouse.press.assert_called()

    def test_drag_release(self, controller: HandController) -> None:
        """Releasing pinch during drag should call mouse.release()."""
        self._enter_cursor(controller)
        pinch1 = _make_pinch_hand(320, 240)
        controller.update([_make_hand_result(pinch1)])

        # Hold pinch to enter drag
        time.sleep(0.3)
        pinch2 = _make_pinch_hand(320, 240)
        controller.update([_make_hand_result(pinch2)])
        assert controller._state == GestureState.DRAGGING

        # Release — clutch mode returns to NEUTRAL
        hand_open = _make_hand_result(_make_open_hand())
        controller.update([hand_open])
        assert controller._state == GestureState.NEUTRAL
        controller._mouse.release.assert_called()

    def test_scroll_on_two_finger_gesture(self, controller: HandController) -> None:
        """Scroll pose with vertical movement should call mouse.scroll()."""
        self._enter_cursor(controller)

        # Scroll pose for confirm frames
        for _ in range(4):
            scroll_lm = _make_scroll_hand(320, 240)
            controller.update([_make_hand_result(scroll_lm)])

        assert controller._state == GestureState.SCROLLING

        # Move up (large delta)
        scroll_up = _make_scroll_hand(320, 200)
        controller.update([_make_hand_result(scroll_up)])
        controller._mouse.scroll.assert_called()

    def test_3d_pinch_detection(self, controller: HandController) -> None:
        """3D pinch detection should use metre-based threshold."""
        self._enter_cursor(controller)

        # Pinch with 3D data: thumb and index very close in 3D
        kps_close = _make_3d_keypoints(
            thumb_tip=(0.0, 0.0, 0.5),
            index_tip=(0.01, 0.01, 0.5),
        )
        pinch_lm = _make_pinch_hand()
        hand = _make_hand_result(pinch_lm, keypoints_3d=kps_close)
        controller.update([hand])
        assert controller._state == GestureState.CLICK_DOWN

    def test_3d_pinch_fallback_to_2d(self, controller: HandController) -> None:
        """Without 3D data, pinch should fall back to 2D pixel distance."""
        self._enter_cursor(controller)

        hand_pinch = _make_hand_result(_make_pinch_hand())  # keypoints_3d=None
        controller.update([hand_pinch])
        assert controller._state == GestureState.CLICK_DOWN

    def test_preferred_hand_selection(self, controller: HandController) -> None:
        """With two hands, preferred hand (Right) should be selected."""
        left = _make_hand_result(handedness="Left")
        right = _make_hand_result(handedness="Right")
        selected = controller._select_hand([left, right])
        assert selected is not None
        assert selected.handedness == "Right"

    def test_single_hand_used_regardless_of_handedness(self, controller: HandController) -> None:
        """With one hand (any handedness), it should be used."""
        left = _make_hand_result(handedness="Left")
        selected = controller._select_hand([left])
        assert selected is not None
        assert selected.handedness == "Left"

    def test_toggle_control(self, controller: HandController) -> None:
        """Toggle should flip control_active state."""
        assert controller.control_active is True
        controller.toggle_control()
        assert controller.control_active is False
        controller.toggle_control()
        assert controller.control_active is True

    def test_no_mouse_action_when_control_off(self, controller: HandController) -> None:
        """Mouse click/press/release should not be called when control is off."""
        controller.toggle_control()  # OFF
        hand_point = _make_hand_result(_make_pointing_hand())
        hand_pinch = _make_hand_result(_make_pinch_hand())
        # Get past IDLE → NEUTRAL → CURSOR
        for _ in range(5):
            controller.update([hand_point])
        # Try pinch (would normally fire click)
        controller.update([hand_pinch])
        controller.update([hand_point])
        # No mouse actions should have been called
        controller._mouse.click.assert_not_called()
        controller._mouse.press.assert_not_called()
