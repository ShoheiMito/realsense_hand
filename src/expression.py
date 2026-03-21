"""Expression recognition module using MediaPipe FaceLandmarker (Tasks API)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

# Blendshape → emotion mapping rules.
# Each rule: (blendshape_name, activation_threshold)
# Confidence is computed as mean(value / threshold) capped at 1.0 per rule.
EMOTION_RULES: dict[str, list[tuple[str, float]]] = {
    "happy": [("mouthSmileLeft", 0.4), ("mouthSmileRight", 0.4)],
    "surprise": [("eyeWideLeft", 0.5), ("eyeWideRight", 0.5), ("jawOpen", 0.3)],
    "angry": [("browDownLeft", 0.5), ("browDownRight", 0.5)],
    "sad": [("mouthFrownLeft", 0.4), ("mouthFrownRight", 0.4)],
    "neutral": [],
}

# Minimum blended confidence to classify as non-neutral.
_EMOTION_MIN_CONFIDENCE: float = 0.3


@dataclass
class ExpressionResult:
    """Result of a single expression recognition inference."""

    emotion: str
    """Emotion label: 'happy', 'surprise', 'angry', 'sad', or 'neutral'."""

    confidence: float
    """Confidence score in [0.0, 1.0]."""

    blendshapes: dict[str, float]
    """Raw ARKit-compatible blendshape coefficients (52 values)."""

    face_landmarks: list = field(default_factory=list)
    """Normalized face landmark coordinates for visualization."""


class ExpressionRecognizer:
    """MediaPipe FaceLandmarker-based expression recognizer.

    Uses Tasks API in IMAGE (synchronous) mode.
    Blendshape-to-emotion mapping is rule-based for low latency and easy tuning.
    """

    def __init__(self, model_path: str) -> None:
        """Initialize FaceLandmarker with blendshape output enabled.

        Args:
            model_path: Path to face_landmarker.task model file.

        Raises:
            RuntimeError: If FaceLandmarker initialization fails.
        """
        import mediapipe as mp  # noqa: F401  (import check)
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=True,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        logger.info("ExpressionRecognizer initialized: %s", model_path)

    def analyze(
        self, rgb_image: np.ndarray, timestamp_ms: int
    ) -> ExpressionResult | None:
        """Recognize expression from an RGB image.

        Args:
            rgb_image: RGB image (H, W, 3) uint8.
            timestamp_ms: Frame timestamp in milliseconds.
                          Not used in IMAGE mode but kept for API consistency
                          with future VIDEO/LIVE_STREAM migration.

        Returns:
            ExpressionResult when a face is detected, None otherwise.
        """
        import mediapipe as mp

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.face_landmarks:
            return None

        # Convert blendshapes to plain dict (first face only)
        blendshapes: dict[str, float] = {}
        if result.face_blendshapes:
            for bs in result.face_blendshapes[0]:
                blendshapes[bs.category_name] = float(bs.score)

        # Face landmarks list (for overlay rendering in visualizer)
        face_landmarks: list = list(result.face_landmarks[0])

        emotion, confidence = self._map_blendshapes_to_emotion(blendshapes)

        return ExpressionResult(
            emotion=emotion,
            confidence=confidence,
            blendshapes=blendshapes,
            face_landmarks=face_landmarks,
        )

    def _map_blendshapes_to_emotion(
        self, blendshapes: dict[str, float]
    ) -> tuple[str, float]:
        """Map blendshape coefficients to an emotion label using rule-based classification.

        For each candidate emotion the confidence is:
            mean( min(blendshape_value / rule_threshold, 1.0) )
        over all rules defined for that emotion.

        The highest-scoring emotion wins. Falls back to 'neutral' when the best
        score is below _EMOTION_MIN_CONFIDENCE.

        Args:
            blendshapes: Blendshape name → score mapping (0.0–1.0).

        Returns:
            Tuple of (emotion_label, confidence_score).
        """
        best_emotion = "neutral"
        best_confidence = 0.0

        for emotion, rules in EMOTION_RULES.items():
            if not rules:
                # neutral is handled as fallback
                continue

            normalized_scores = [
                min(blendshapes.get(name, 0.0) / threshold, 1.0)
                for name, threshold in rules
            ]
            confidence = sum(normalized_scores) / len(normalized_scores)

            if confidence > best_confidence:
                best_confidence = confidence
                best_emotion = emotion

        if best_confidence < _EMOTION_MIN_CONFIDENCE:
            # Return neutral with inverse confidence (lower emotion signal → more neutral)
            return "neutral", round(1.0 - best_confidence, 4)

        return best_emotion, round(best_confidence, 4)

    def close(self) -> None:
        """Release FaceLandmarker resources."""
        self._landmarker.close()
        logger.info("ExpressionRecognizer closed.")
