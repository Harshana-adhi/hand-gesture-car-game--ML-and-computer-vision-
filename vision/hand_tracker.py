"""
MediaPipe Hands wrapper: webcam frame -> 21 hand landmarks.

Note on API version: this project pins mediapipe>=1.0 (see requirements.txt),
which is the only mediapipe release with prebuilt wheels for Python 3.14 at
the time of writing. That release removed the old `mp.solutions.hands`
API entirely in favor of the newer Tasks API (`mediapipe.tasks.python.vision
.HandLandmarker`), which is what's used here. Functionally it's the same
pretrained hand-landmark model, just a different Python entry point.

The HandLandmarker model asset (hand_landmarker.task) is downloaded once via
download_model.py into models/assets/ and loaded from disk here.
"""
import os

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "assets", "hand_landmarker.task",
)


class HandTracker:
    """Wraps MediaPipe HandLandmarker for single-hand tracking on live frames."""

    # MediaPipe's palm detector is trained mostly on open/partially-open
    # hand shapes; a tightly closed fist (the "brake" gesture) gives it
    # less to work with and can dip below a stricter confidence threshold,
    # dropping the frame as "no hand detected" before it ever reaches the
    # trained gesture classifier. Lowered from the defaults (0.6/0.5) to
    # make detection more forgiving of fist/curled poses.
    def __init__(self, model_path: str = _MODEL_PATH, max_hands: int = 1,
                 min_detection_confidence: float = 0.4,
                 min_presence_confidence: float = 0.35):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Hand landmark model not found at {model_path}. "
                "Run download_model.py first."
            )
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=max_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_presence_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)

    def process(self, frame_rgb):
        """
        frame_rgb: an HxWx3 uint8 numpy array in RGB order (convert from
        OpenCV's default BGR with cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).

        Returns a list of 21 (x, y, z) normalized landmarks (x,y in [0,1]
        relative to image width/height, z relative depth) for the first
        detected hand, or None if no hand was detected in this frame.
        """
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self._landmarker.detect(mp_image)
        if not result.hand_landmarks:
            return None
        first_hand = result.hand_landmarks[0]
        return [(lm.x, lm.y, lm.z) for lm in first_hand]

    def close(self):
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
