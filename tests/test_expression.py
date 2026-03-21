"""Tests for expression recognition module.

Unit tests run without RealSense hardware or the face_landmarker.task model.
Integration tests (requiring the model file) are skipped automatically when the
model is absent.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.expression import (
    EMOTION_RULES,
    ExpressionRecognizer,
    ExpressionResult,
    _EMOTION_MIN_CONFIDENCE,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MODEL_PATH = "models/face_landmarker.task"
_MODEL_AVAILABLE = os.path.exists(_MODEL_PATH)


def _make_recognizer_no_model() -> ExpressionRecognizer:
    """Return an ExpressionRecognizer instance bypassing model file I/O."""
    rec: ExpressionRecognizer = ExpressionRecognizer.__new__(ExpressionRecognizer)
    rec._landmarker = MagicMock()
    return rec


# ---------------------------------------------------------------------------
# ExpressionResult dataclass
# ---------------------------------------------------------------------------


class TestExpressionResult:
    def test_fields_accessible(self) -> None:
        result = ExpressionResult(
            emotion="happy",
            confidence=0.8,
            blendshapes={"mouthSmileLeft": 0.7, "mouthSmileRight": 0.6},
        )
        assert result.emotion == "happy"
        assert result.confidence == 0.8
        assert result.blendshapes["mouthSmileLeft"] == 0.7

    def test_face_landmarks_defaults_to_empty_list(self) -> None:
        result = ExpressionResult(emotion="neutral", confidence=1.0, blendshapes={})
        assert result.face_landmarks == []

    def test_face_landmarks_mutable_default_independent(self) -> None:
        r1 = ExpressionResult(emotion="neutral", confidence=1.0, blendshapes={})
        r2 = ExpressionResult(emotion="neutral", confidence=1.0, blendshapes={})
        r1.face_landmarks.append("x")
        assert r2.face_landmarks == [], "default_factory must create independent lists"

    def test_with_face_landmarks(self) -> None:
        landmarks = [object(), object()]
        result = ExpressionResult(
            emotion="sad", confidence=0.5, blendshapes={}, face_landmarks=landmarks
        )
        assert len(result.face_landmarks) == 2


# ---------------------------------------------------------------------------
# EMOTION_RULES sanity checks
# ---------------------------------------------------------------------------


class TestEmotionRules:
    def test_all_expected_emotions_present(self) -> None:
        expected = {"happy", "surprise", "angry", "sad", "neutral"}
        assert set(EMOTION_RULES.keys()) == expected

    def test_neutral_has_no_rules(self) -> None:
        assert EMOTION_RULES["neutral"] == []

    def test_rule_thresholds_in_valid_range(self) -> None:
        for emotion, rules in EMOTION_RULES.items():
            for name, threshold in rules:
                assert isinstance(name, str), f"{emotion}: rule name must be str"
                assert 0.0 < threshold <= 1.0, (
                    f"{emotion}.{name}: threshold {threshold} out of range"
                )


# ---------------------------------------------------------------------------
# _map_blendshapes_to_emotion (unit tests — no model required)
# ---------------------------------------------------------------------------


class TestMapBlendshapesToEmotion:
    @pytest.fixture
    def rec(self) -> ExpressionRecognizer:
        return _make_recognizer_no_model()

    # --- happy ---

    def test_happy_above_threshold(self, rec: ExpressionRecognizer) -> None:
        bs = {"mouthSmileLeft": 0.8, "mouthSmileRight": 0.7}
        emotion, confidence = rec._map_blendshapes_to_emotion(bs)
        assert emotion == "happy"
        assert confidence >= _EMOTION_MIN_CONFIDENCE

    def test_happy_exactly_at_threshold(self, rec: ExpressionRecognizer) -> None:
        bs = {"mouthSmileLeft": 0.4, "mouthSmileRight": 0.4}
        emotion, confidence = rec._map_blendshapes_to_emotion(bs)
        assert emotion == "happy"
        assert confidence == pytest.approx(1.0)

    # --- surprise ---

    def test_surprise_above_threshold(self, rec: ExpressionRecognizer) -> None:
        bs = {"eyeWideLeft": 0.9, "eyeWideRight": 0.8, "jawOpen": 0.6}
        emotion, confidence = rec._map_blendshapes_to_emotion(bs)
        assert emotion == "surprise"
        assert confidence >= _EMOTION_MIN_CONFIDENCE

    def test_surprise_missing_jaw_open_lowers_confidence(
        self, rec: ExpressionRecognizer
    ) -> None:
        full = {"eyeWideLeft": 0.9, "eyeWideRight": 0.8, "jawOpen": 0.6}
        partial = {"eyeWideLeft": 0.9, "eyeWideRight": 0.8, "jawOpen": 0.0}
        _, conf_full = rec._map_blendshapes_to_emotion(full)
        _, conf_partial = rec._map_blendshapes_to_emotion(partial)
        assert conf_partial < conf_full

    # --- angry ---

    def test_angry_above_threshold(self, rec: ExpressionRecognizer) -> None:
        bs = {"browDownLeft": 0.9, "browDownRight": 0.8}
        emotion, confidence = rec._map_blendshapes_to_emotion(bs)
        assert emotion == "angry"
        assert confidence >= _EMOTION_MIN_CONFIDENCE

    # --- sad ---

    def test_sad_above_threshold(self, rec: ExpressionRecognizer) -> None:
        bs = {"mouthFrownLeft": 0.7, "mouthFrownRight": 0.8}
        emotion, confidence = rec._map_blendshapes_to_emotion(bs)
        assert emotion == "sad"
        assert confidence >= _EMOTION_MIN_CONFIDENCE

    # --- neutral ---

    def test_neutral_on_empty_blendshapes(self, rec: ExpressionRecognizer) -> None:
        emotion, _ = rec._map_blendshapes_to_emotion({})
        assert emotion == "neutral"

    def test_neutral_on_low_values(self, rec: ExpressionRecognizer) -> None:
        bs = {"mouthSmileLeft": 0.05, "mouthSmileRight": 0.05}
        emotion, _ = rec._map_blendshapes_to_emotion(bs)
        assert emotion == "neutral"

    def test_neutral_confidence_near_one_for_empty(
        self, rec: ExpressionRecognizer
    ) -> None:
        _, confidence = rec._map_blendshapes_to_emotion({})
        assert confidence == pytest.approx(1.0)

    # --- general invariants ---

    def test_confidence_always_in_unit_interval(
        self, rec: ExpressionRecognizer
    ) -> None:
        test_cases: list[dict[str, float]] = [
            {},
            {"mouthSmileLeft": 1.0, "mouthSmileRight": 1.0},
            {"browDownLeft": 0.3, "browDownRight": 0.2},
            {"eyeWideLeft": 0.99, "eyeWideRight": 0.99, "jawOpen": 0.99},
        ]
        for bs in test_cases:
            _, confidence = rec._map_blendshapes_to_emotion(bs)
            assert 0.0 <= confidence <= 1.0, f"confidence={confidence} out of range for {bs}"

    def test_emotion_label_always_valid(self, rec: ExpressionRecognizer) -> None:
        valid = set(EMOTION_RULES.keys())
        test_cases: list[dict[str, float]] = [
            {},
            {"mouthSmileLeft": 0.9, "mouthSmileRight": 0.9},
            {"browDownLeft": 0.9, "browDownRight": 0.9},
        ]
        for bs in test_cases:
            emotion, _ = rec._map_blendshapes_to_emotion(bs)
            assert emotion in valid


# ---------------------------------------------------------------------------
# analyze() — mocked landmarker (no model file required)
# ---------------------------------------------------------------------------


class TestAnalyzeMocked:
    @pytest.fixture(autouse=True)
    def _patch_mediapipe(self):
        """Patch sys.modules so `import mediapipe as mp` inside analyze() succeeds."""
        mock_mp = MagicMock()
        mock_mp.ImageFormat.SRGB = 1
        mock_mp.Image.return_value = MagicMock()
        with patch.dict(sys.modules, {"mediapipe": mock_mp}):
            yield

    @pytest.fixture
    def rec(self) -> ExpressionRecognizer:
        return _make_recognizer_no_model()

    def _make_blendshape(self, name: str, score: float) -> MagicMock:
        bs = MagicMock()
        bs.category_name = name
        bs.score = score
        return bs

    def _make_landmark(self, x: float = 0.5, y: float = 0.5, z: float = 0.0) -> MagicMock:
        lm = MagicMock()
        lm.x, lm.y, lm.z = x, y, z
        return lm

    def _mock_detect_result(
        self,
        rec: ExpressionRecognizer,
        blendshape_dict: dict[str, float],
        num_landmarks: int = 5,
    ) -> None:
        mock_result = MagicMock()
        landmarks = [self._make_landmark() for _ in range(num_landmarks)]
        mock_result.face_landmarks = [landmarks]
        mock_result.face_blendshapes = [
            [self._make_blendshape(k, v) for k, v in blendshape_dict.items()]
        ]
        rec._landmarker.detect.return_value = mock_result

    def test_returns_none_when_no_face_detected(
        self, rec: ExpressionRecognizer
    ) -> None:
        mock_result = MagicMock()
        mock_result.face_landmarks = []
        rec._landmarker.detect.return_value = mock_result

        image = np.zeros((480, 640, 3), dtype=np.uint8)
        result = rec.analyze(image, timestamp_ms=0)
        assert result is None

    def test_returns_expression_result_on_face_detected(
        self, rec: ExpressionRecognizer
    ) -> None:
        self._mock_detect_result(
            rec, {"mouthSmileLeft": 0.8, "mouthSmileRight": 0.7}
        )
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        result = rec.analyze(image, timestamp_ms=33)

        assert result is not None
        assert isinstance(result, ExpressionResult)

    def test_emotion_is_happy_when_smile_blendshapes_high(
        self, rec: ExpressionRecognizer
    ) -> None:
        self._mock_detect_result(
            rec, {"mouthSmileLeft": 0.9, "mouthSmileRight": 0.9}
        )
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        result = rec.analyze(image, timestamp_ms=33)

        assert result is not None
        assert result.emotion == "happy"

    def test_blendshapes_dict_populated(self, rec: ExpressionRecognizer) -> None:
        bs_input = {"mouthSmileLeft": 0.7, "jawOpen": 0.2}
        self._mock_detect_result(rec, bs_input)
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        result = rec.analyze(image, timestamp_ms=0)

        assert result is not None
        assert result.blendshapes["mouthSmileLeft"] == pytest.approx(0.7)
        assert result.blendshapes["jawOpen"] == pytest.approx(0.2)

    def test_face_landmarks_populated(self, rec: ExpressionRecognizer) -> None:
        self._mock_detect_result(rec, {}, num_landmarks=10)
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        result = rec.analyze(image, timestamp_ms=0)

        assert result is not None
        assert len(result.face_landmarks) == 10

    def test_timestamp_ms_accepted_without_error(
        self, rec: ExpressionRecognizer
    ) -> None:
        """timestamp_ms is unused in IMAGE mode but must not raise."""
        self._mock_detect_result(rec, {})
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        for ts in [0, 33, 1000, 999999]:
            rec.analyze(image, timestamp_ms=ts)  # should not raise


# ---------------------------------------------------------------------------
# close()
# ---------------------------------------------------------------------------


class TestClose:
    def test_close_calls_landmarker_close(self) -> None:
        rec = _make_recognizer_no_model()
        rec.close()
        rec._landmarker.close.assert_called_once()


# ---------------------------------------------------------------------------
# Integration tests (requires models/face_landmarker.task)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _MODEL_AVAILABLE,
    reason=f"model file not found: {_MODEL_PATH}",
)
class TestExpressionRecognizerIntegration:
    @pytest.fixture(scope="class")
    def recognizer(self) -> ExpressionRecognizer:  # type: ignore[override]
        rec = ExpressionRecognizer(_MODEL_PATH)
        yield rec
        rec.close()

    def test_black_image_returns_none(
        self, recognizer: ExpressionRecognizer
    ) -> None:
        black = np.zeros((480, 640, 3), dtype=np.uint8)
        result = recognizer.analyze(black, timestamp_ms=0)
        assert result is None

    def test_noise_image_does_not_raise(
        self, recognizer: ExpressionRecognizer
    ) -> None:
        rng = np.random.default_rng(42)
        noise = rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)
        result = recognizer.analyze(noise, timestamp_ms=33)
        assert result is None or isinstance(result, ExpressionResult)

    def test_result_fields_valid_when_face_detected(
        self, recognizer: ExpressionRecognizer
    ) -> None:
        rng = np.random.default_rng(0)
        image = rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)
        result = recognizer.analyze(image, timestamp_ms=66)
        if result is not None:
            assert result.emotion in set(EMOTION_RULES.keys())
            assert 0.0 <= result.confidence <= 1.0
            assert isinstance(result.blendshapes, dict)
            assert isinstance(result.face_landmarks, list)
