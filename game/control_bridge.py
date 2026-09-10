"""
Control bridge interface: whatever supplies input to the game (keyboard
stub or the real trained hand-tracking pipeline) implements this same
interface, so the game loop never needs to know which one it's talking to.

Per CLAUDE.md Instruction #3: build and test the game with the keyboard
stub first (KeyboardControlBridge below); the real CV-driven bridge is
added in Phase 6 as a second implementation of the same interface, and
swapping main.py over to it is a one-line change.

Actions: "accelerate", "brake", "neutral", "boost".
"""
import os
import threading
import time
from collections import Counter, deque

import cv2
import joblib
import numpy as np
import pygame

from vision.features import finger_curl_vector, tilt_angle
from vision.hand_tracker import HandTracker


class ControlBridge:
    """Interface every control source implements."""

    def get_steering(self) -> float:
        """Returns a steering value in [-1, 1]. 0 = centered."""
        raise NotImplementedError

    def get_action(self) -> str:
        """Returns one of 'accelerate', 'brake', 'neutral', 'boost'."""
        raise NotImplementedError

    def is_hand_detected(self) -> bool:
        """
        Whether a hand is currently being tracked. Always True for the
        keyboard stub; meaningful once the real CV bridge is wired in
        (Phase 6) so the debug overlay (Phase 7) can show it.
        """
        return True

    def close(self):
        """Release any resources (camera, background thread, etc.)."""
        pass


class KeyboardControlBridge(ControlBridge):
    """
    Temporary stand-in used to build and debug the game itself before the
    trained CV pipeline is wired in (Phase 6).

    Left/Right arrows -> steering. Up = accelerate, Down = brake,
    Space = boost, nothing pressed = neutral.
    """

    def get_steering(self) -> float:
        keys = pygame.key.get_pressed()
        steering = 0.0
        if keys[pygame.K_LEFT]:
            steering -= 1.0
        if keys[pygame.K_RIGHT]:
            steering += 1.0
        return steering

    def get_action(self) -> str:
        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE]:
            return "boost"
        if keys[pygame.K_UP]:
            return "accelerate"
        if keys[pygame.K_DOWN]:
            return "brake"
        return "neutral"


_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STEERING_MODEL_PATH = os.path.join(_ROOT, "models", "saved", "steering_model.joblib")
GESTURE_MODEL_PATH = os.path.join(_ROOT, "models", "saved", "gesture_model.joblib")

STEERING_EMA_ALPHA = 0.3       # higher = more responsive, lower = smoother
GESTURE_VOTE_WINDOW = 5        # majority vote over the last N classifier predictions


class CVControlBridge(ControlBridge):
    """
    Runs MediaPipe hand tracking + the two trained models (steering
    regression, gesture classifier) in a background thread so webcam/model
    inference never blocks the game's render loop (Instruction #5).

    Applies an exponential moving average to the raw per-frame steering
    prediction, and a majority vote over the last few frames' gesture
    predictions (Instruction #6), so jittery single-frame misreads don't
    reach the game as control glitches.

    When no hand is detected in a frame, that frame contributes a neutral
    reading (steering 0.0, action "neutral") to the smoothing filters
    instead of a stale or bogus prediction, so control drifts gracefully
    back to center/neutral rather than snapping (Instruction #8).
    """

    def __init__(self, camera_index: int = 0):
        if not os.path.exists(STEERING_MODEL_PATH):
            raise FileNotFoundError(
                f"No trained steering model at {STEERING_MODEL_PATH}. "
                "Run `python -m models.train_steering_model` first."
            )
        if not os.path.exists(GESTURE_MODEL_PATH):
            raise FileNotFoundError(
                f"No trained gesture model at {GESTURE_MODEL_PATH}. "
                "Run `python -m models.train_gesture_model` first."
            )

        steering_bundle = joblib.load(STEERING_MODEL_PATH)
        gesture_bundle = joblib.load(GESTURE_MODEL_PATH)
        self._steering_model = steering_bundle["model"]
        self._gesture_model = gesture_bundle["model"]

        self._cap = cv2.VideoCapture(camera_index)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open webcam (device index {camera_index}).")
        self._tracker = HandTracker()

        self._lock = threading.Lock()
        self._smoothed_steering = 0.0
        self._raw_steering = 0.0
        self._gesture_history = deque(maxlen=GESTURE_VOTE_WINDOW)
        self._voted_action = "neutral"
        self._hand_detected = False
        self._last_landmarks = None

        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def _run_loop(self):
        while not self._stop_event.is_set():
            ok, frame = self._cap.read()
            if not ok:
                time.sleep(0.01)
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            landmarks = self._tracker.process(frame_rgb)

            if landmarks is not None:
                angle = tilt_angle(landmarks)
                raw_steering = float(self._steering_model.predict(np.array([[angle]]))[0])
                raw_steering = max(-1.0, min(1.0, raw_steering))

                curl = finger_curl_vector(landmarks)
                predicted_action = self._gesture_model.predict(np.array([curl]))[0]
            else:
                raw_steering = 0.0
                predicted_action = "neutral"

            with self._lock:
                self._hand_detected = landmarks is not None
                self._raw_steering = raw_steering
                self._smoothed_steering = (
                    STEERING_EMA_ALPHA * raw_steering
                    + (1 - STEERING_EMA_ALPHA) * self._smoothed_steering
                )
                self._gesture_history.append(predicted_action)
                self._voted_action = Counter(self._gesture_history).most_common(1)[0][0]

    def get_steering(self) -> float:
        with self._lock:
            return self._smoothed_steering

    def get_action(self) -> str:
        with self._lock:
            return self._voted_action

    def is_hand_detected(self) -> bool:
        with self._lock:
            return self._hand_detected

    def close(self):
        self._stop_event.set()
        self._thread.join(timeout=2.0)
        self._cap.release()
        self._tracker.close()
