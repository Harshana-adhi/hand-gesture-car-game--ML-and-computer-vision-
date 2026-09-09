"""
Guided gesture calibration.

Prompts you to hold each gesture (open palm = accelerate, fist = brake,
relaxed resting hand = neutral, thumbs-up = boost) for a few seconds while
it records finger-curl-vector samples, showing an on-screen prompt and
countdown throughout.

Vary your hand's position/distance slightly during each hold so the
classifier doesn't overfit to one exact placement (Section 8 of CLAUDE.md).

Each recorded sample is saved as its own JSON file under
calibration/data/gestures/<class>/sample_NNN.json.

Controls:
    A window opens showing your webcam feed with the current prompt overlaid.
    Press 'q' at any time to abort without saving the current class's samples
    (classes already completed before that point stay saved).

Run:
    python -m calibration.calibrate_gestures
    python -m calibration.calibrate_gestures --skip-boost   # skip the optional boost class
"""
import argparse
import json
import os
import time

import cv2

from vision.features import finger_curl_vector
from vision.hand_tracker import HandTracker

TARGET_SAMPLES = 25
COUNTDOWN_SECONDS = 2.0
MAX_RECORD_SECONDS = 15.0  # safety cap in case detection is spotty

GESTURES = [
    ("accelerate", "OPEN PALM (fingers spread)"),
    ("brake", "FIST (fingers curled)"),
    ("neutral", "RELAXED RESTING HAND"),
    ("boost", "THUMBS UP"),
]

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "gestures")


def _draw_overlay(frame, text, sub_text="", color=(0, 255, 255)):
    cv2.putText(frame, text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3)
    if sub_text:
        cv2.putText(frame, sub_text, (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return frame


def _countdown(cap, class_name, instruction):
    start = time.time()
    while time.time() - start < COUNTDOWN_SECONDS:
        ok, frame = cap.read()
        if not ok:
            continue
        remaining = COUNTDOWN_SECONDS - (time.time() - start)
        # Mirror for display only -- landmark detection elsewhere always
        # runs on the raw, unflipped frame, so this doesn't touch samples.
        display_frame = cv2.flip(frame, 1)
        display_frame = _draw_overlay(display_frame, f"Get ready: {class_name}", f"{instruction} -- starting in {remaining:.1f}s")
        cv2.imshow("Gesture Calibration", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return False
    return True


def _record_class(cap, tracker, class_name, instruction):
    """Records curl-vector samples until TARGET_SAMPLES is hit or time runs out."""
    samples = []
    start = time.time()
    while len(samples) < TARGET_SAMPLES and time.time() - start < MAX_RECORD_SECONDS:
        ok, frame = cap.read()
        if not ok:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        landmarks = tracker.process(frame_rgb)

        if landmarks is not None:
            curl = finger_curl_vector(landmarks)
            samples.append(curl)
            status = f"{len(samples)}/{TARGET_SAMPLES} samples  -- vary position slightly"
            color = (0, 255, 0)
        else:
            status = f"{len(samples)}/{TARGET_SAMPLES} samples  (NO HAND DETECTED)"
            color = (0, 0, 255)

        # Mirror for display only -- curl_vector above was computed from
        # the raw, unflipped frame, so saved samples are unaffected.
        display_frame = cv2.flip(frame, 1)
        display_frame = _draw_overlay(display_frame, f"Hold: {class_name}", f"{instruction}  {status}", color=color)
        cv2.imshow("Gesture Calibration", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return samples, False
    return samples, True


def _save_class_samples(class_name, samples):
    class_dir = os.path.join(DATA_DIR, class_name)
    os.makedirs(class_dir, exist_ok=True)
    existing = [f for f in os.listdir(class_dir) if f.startswith("sample_") and f.endswith(".json")]
    start_idx = len(existing) + 1
    for i, curl_vector in enumerate(samples):
        idx = start_idx + i
        path = os.path.join(class_dir, f"sample_{idx:04d}.json")
        with open(path, "w") as f:
            json.dump({"class": class_name, "curl_vector": curl_vector}, f)
    return len(samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-boost", action="store_true",
                         help="Skip the optional boost (thumbs-up) gesture class.")
    args = parser.parse_args()

    gestures = GESTURES
    if args.skip_boost:
        gestures = [g for g in gestures if g[0] != "boost"]

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: could not open webcam (device index 0).")
        return

    saved_counts = {}
    with HandTracker() as tracker:
        for class_name, instruction in gestures:
            print(f"\n=== {class_name} ({instruction}) ===")
            if not _countdown(cap, class_name, instruction):
                print("Aborted by user.")
                break
            samples, keep_going = _record_class(cap, tracker, class_name, instruction)
            n_saved = _save_class_samples(class_name, samples)
            saved_counts[class_name] = saved_counts.get(class_name, 0) + n_saved
            print(f"  Saved {n_saved} samples for '{class_name}'")
            if not keep_going:
                print("Aborted by user.")
                break

    cap.release()
    cv2.destroyAllWindows()

    print("\n--- summary ---")
    for class_name, count in saved_counts.items():
        print(f"  {class_name}: {count} samples this run")


if __name__ == "__main__":
    main()
