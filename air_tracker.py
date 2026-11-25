"""Reusable MediaPipe hand-tracking wrapper based on AIR_PAINTING.py."""

import os
import tempfile

# Ensure TensorFlow/Keras avoids read-only or restricted home config paths
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
if "KERAS_HOME" not in os.environ:
    os.environ["KERAS_HOME"] = os.path.join(tempfile.gettempdir(), "keras_cache")

import cv2
import mediapipe as mp
import numpy as np

CANVAS_WIDTH = 640
CANVAS_HEIGHT = 480


class AirHandTracker:
    """Tracks a single hand from JPEG frames using MediaPipe Hands."""

    def __init__(
        self,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.6,
    ) -> None:
        self._mp_hands = mp.solutions.hands
        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process_jpeg(self, jpeg_bytes: bytes) -> dict:
        """Decode JPEG bytes and return hand landmarks in 640x480 space."""
        buffer = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if frame is None:
            return {"detected": False, "error": "invalid_frame"}

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._hands.process(frame_rgb)

        if not result.multi_hand_landmarks:
            return {"detected": False}

        hand_lms = result.multi_hand_landmarks[0]
        landmarks = [
            [int(lm.x * CANVAS_WIDTH), int(lm.y * CANVAS_HEIGHT)]
            for lm in hand_lms.landmark
        ]
        index_tip = landmarks[8]
        thumb_tip = landmarks[4]

        return {
            "detected": True,
            "landmarks": landmarks,
            "indexTip": index_tip,
            "thumbTip": thumb_tip,
            "deltaY": thumb_tip[1] - index_tip[1],
        }

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._hands.close()
