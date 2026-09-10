"""
In-game recalibration flow: the same guided countdown/hold capture as
calibration/calibrate_steering.py and calibration/calibrate_gestures.py,
but driven by the running game's own CVControlBridge (which already owns
the one webcam handle) instead of opening a second camera, and rendered
in the pygame window instead of a separate OpenCV window.

Saved files use exactly the same format/paths as the standalone scripts,
so models/train_steering_model.py and models/train_gesture_model.py work
unchanged on data collected here.
"""
import json
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STEERING_DATA_PATH = os.path.join(_ROOT, "calibration", "data", "steering_calibration.json")
GESTURES_DATA_DIR = os.path.join(_ROOT, "calibration", "data", "gestures")

HOLD_SECONDS = 2.0
COUNTDOWN_SECONDS = 2.0
NUM_SWEEPS = 2
STEERING_POSITIONS = [("LEFT", -1.0), ("CENTER", 0.0), ("RIGHT", 1.0)]

GESTURE_TARGET_SAMPLES = 25
GESTURE_MAX_RECORD_SECONDS = 15.0

GESTURE_INSTRUCTIONS = {
    "accelerate": "OPEN PALM (fingers spread)",
    "brake": "FIST (fingers curled)",
    "neutral": "RELAXED RESTING HAND",
    "boost": "FOUR FINGERS (index+middle+ring+pinky extended, thumb curled)",
    "two_fingers": "TWO FINGERS (index + middle extended, peace sign)",
}

# Order shown in the settings list.
CALIBRATION_TARGETS = ["steering", "accelerate", "brake", "neutral", "boost", "two_fingers"]


def existing_sample_count(target: str) -> int:
    if target == "steering":
        if not os.path.exists(STEERING_DATA_PATH):
            return 0
        with open(STEERING_DATA_PATH) as f:
            return len(json.load(f))
    class_dir = os.path.join(GESTURES_DATA_DIR, target)
    if not os.path.isdir(class_dir):
        return 0
    return len([f for f in os.listdir(class_dir) if f.endswith(".json")])


class CalibrationSession:
    """
    Drives one recalibration run. Call update(dt, bridge) once per game
    tick while active; check .phase / .done / .aborted to know what to
    render. Call save() once .done is True to persist to disk in the same
    format the standalone calibration scripts use.
    """

    def __init__(self, target: str, delete_existing: bool):
        self.target = target
        self.delete_existing = delete_existing
        self.is_steering = target == "steering"

        if self.is_steering:
            self.steps = [(label, val) for _ in range(NUM_SWEEPS) for label, val in STEERING_POSITIONS]
        else:
            # Short label for the big "Get ready"/"Hold" text -- the full
            # instruction (GESTURE_INSTRUCTIONS) is shown as a subtitle by
            # the caller instead, since it's too long for that big font.
            self.steps = [(target.upper().replace("_", " "), None)]

        self.step_index = 0
        self.phase = "countdown"  # "countdown" | "hold"
        self.phase_timer = 0.0
        self.samples = []  # steering: [{"tilt_angle","target"}], gesture: [curl_vector]
        self.done = False
        self.aborted = False

    @property
    def current_label(self):
        if self.step_index >= len(self.steps):
            return None
        return self.steps[self.step_index][0]

    @property
    def current_sweep_progress(self):
        """(step_index+1, total_steps) for a simple progress readout."""
        return self.step_index + 1, len(self.steps)

    def abort(self):
        self.aborted = True

    def update(self, dt: float, bridge):
        if self.done or self.aborted:
            return

        self.phase_timer += dt
        label, target_val = self.steps[self.step_index]

        if self.phase == "countdown":
            if self.phase_timer >= COUNTDOWN_SECONDS:
                self.phase = "hold"
                self.phase_timer = 0.0
            return

        # phase == "hold"
        angle, curl = bridge.get_raw_features()
        if self.is_steering:
            if angle is not None:
                self.samples.append({"tilt_angle": angle, "target": target_val})
            hold_finished = self.phase_timer >= HOLD_SECONDS
        else:
            if curl is not None and len(self.samples) < GESTURE_TARGET_SAMPLES:
                self.samples.append(curl)
            hold_finished = (len(self.samples) >= GESTURE_TARGET_SAMPLES
                              or self.phase_timer >= GESTURE_MAX_RECORD_SECONDS)

        if hold_finished:
            self.step_index += 1
            self.phase_timer = 0.0
            self.phase = "countdown"
            if self.step_index >= len(self.steps):
                self.done = True

    def save(self) -> int:
        """Persists collected samples to disk; returns how many new samples were saved."""
        if self.is_steering:
            existing = []
            if not self.delete_existing and os.path.exists(STEERING_DATA_PATH):
                with open(STEERING_DATA_PATH) as f:
                    existing = json.load(f)
            combined = existing + self.samples
            os.makedirs(os.path.dirname(STEERING_DATA_PATH), exist_ok=True)
            with open(STEERING_DATA_PATH, "w") as f:
                json.dump(combined, f, indent=2)
            return len(self.samples)

        class_dir = os.path.join(GESTURES_DATA_DIR, self.target)
        os.makedirs(class_dir, exist_ok=True)
        if self.delete_existing:
            for fname in os.listdir(class_dir):
                if fname.endswith(".json"):
                    os.remove(os.path.join(class_dir, fname))
        existing = [f for f in os.listdir(class_dir) if f.startswith("sample_") and f.endswith(".json")]
        start_idx = len(existing) + 1
        for i, curl_vector in enumerate(self.samples):
            path = os.path.join(class_dir, f"sample_{start_idx + i:04d}.json")
            with open(path, "w") as f:
                json.dump({"class": self.target, "curl_vector": curl_vector}, f)
        return len(self.samples)
