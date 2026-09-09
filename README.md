# Hand-Gesture Controlled Car Game

Endless driving game controlled entirely by one hand in front of a webcam.
Tilt to steer, open palm to accelerate, fist to brake. See [CLAUDE.md](CLAUDE.md)
for the full project brief, technical spec, and build plan.

## Setup

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
python download_model.py       # downloads the MediaPipe hand-landmark model (~7.8MB)
```

### Environment notes (Python 3.14)

This machine only has Python 3.14 available, which changed two dependency choices
from the original spec:

- **`pygame-ce` instead of `pygame`.** Plain `pygame` has no prebuilt wheel for
  Python 3.14 yet and fails to build from source without MSYS2. `pygame-ce` is the
  actively maintained community fork, API-compatible (`import pygame` still works),
  with a prebuilt Windows wheel for 3.14.
- **MediaPipe Tasks API instead of the legacy `mp.solutions.hands` API.** The only
  mediapipe release with a Python 3.14 wheel (`mediapipe==1.0.1`) removed the old
  `solutions` API entirely. `vision/hand_tracker.py` uses the newer
  `mediapipe.tasks.python.vision.HandLandmarker` API instead — same pretrained
  21-point hand-landmark model, different entry point. It loads its model from a
  local file (`models/assets/hand_landmarker.task`), fetched once by
  `download_model.py` from Google's official MediaPipe model storage.

## Verifying the setup

```bash
python phase0_check.py   # confirms webcam + MediaPipe hand detection work
python phase1_check.py   # prints live tilt-angle / finger-curl values as you move your hand
```

## Calibration (Phase 3 — you run this, not Claude)

```bash
python -m calibration.calibrate_steering
python -m calibration.calibrate_gestures
```

See Section 8 of [CLAUDE.md](CLAUDE.md) for the full protocol.
