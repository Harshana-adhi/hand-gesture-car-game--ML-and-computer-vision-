"""
Guided steering calibration.

Prompts you to hold your hand tilted fully LEFT, then CENTER, then fully
RIGHT, each for ~2 seconds, with an on-screen countdown. Repeats the full
left-center-right sweep twice (Section 8 of CLAUDE.md requires at least two
sweeps so there's real held-out data for validation, not just fitting data).

Every frame captured during a hold where a hand was detected is saved as one
labeled sample: {"tilt_angle": <raw feature>, "target": -1.0 / 0.0 / 1.0}.

Controls:
    A window opens showing your webcam feed with the current prompt overlaid.
    Press 'q' at any time to abort without saving.

Run:
    python -m calibration.calibrate_steering
"""
import json
import os
import time

import cv2

from vision.features import tilt_angle
from vision.hand_tracker import HandTracker

HOLD_SECONDS = 2.0
COUNTDOWN_SECONDS = 2.0
NUM_SWEEPS = 2
POSITIONS = [("LEFT", -1.0), ("CENTER", 0.0), ("RIGHT", 1.0)]

OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "steering_calibration.json")


def _draw_overlay(frame, text, sub_text="", color=(0, 255, 255)):
    cv2.putText(frame, text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    if sub_text:
        cv2.putText(frame, sub_text, (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return frame


def _countdown(cap, tracker, label):
    """Shows a countdown before recording starts for this position."""
    start = time.time()
    while time.time() - start < COUNTDOWN_SECONDS:
        ok, frame = cap.read()
        if not ok:
            continue
        remaining = COUNTDOWN_SECONDS - (time.time() - start)
        # Mirror for display only (feels like a selfie camera) -- landmark
        # detection elsewhere always runs on the raw, unflipped frame, so
        # this has no effect on the tilt-angle feature or saved samples.
        display_frame = cv2.flip(frame, 1)
        display_frame = _draw_overlay(display_frame, f"Get ready: {label}", f"starting in {remaining:.1f}s",
                                       color=(0, 200, 255))
        cv2.imshow("Steering Calibration", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return False
    return True


def _record_hold(cap, tracker, label, target):
    """Records tilt-angle samples for HOLD_SECONDS while showing 'Hold ...' + countdown."""
    samples = []
    start = time.time()
    while time.time() - start < HOLD_SECONDS:
        ok, frame = cap.read()
        if not ok:
            continue

        remaining = HOLD_SECONDS - (time.time() - start)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        landmarks = tracker.process(frame_rgb)

        if landmarks is not None:
            angle = tilt_angle(landmarks)
            samples.append({"tilt_angle": angle, "target": target})
            status = f"recording... {remaining:.1f}s  (angle={angle:.1f})"
            color = (0, 255, 0)
        else:
            status = f"recording... {remaining:.1f}s  (NO HAND DETECTED)"
            color = (0, 0, 255)

        # Mirror for display only -- landmarks/tilt_angle above were computed
        # from the raw, unflipped frame, so saved samples are unaffected.
        display_frame = cv2.flip(frame, 1)
        display_frame = _draw_overlay(display_frame, f"Hold {label}", status, color=color)
        cv2.imshow("Steering Calibration", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return samples, False
    return samples, True


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: could not open webcam (device index 0).")
        return

    all_samples = []
    aborted = False

    with HandTracker() as tracker:
        for sweep in range(1, NUM_SWEEPS + 1):
            if aborted:
                break
            print(f"\n=== Sweep {sweep}/{NUM_SWEEPS} ===")
            for label, target in POSITIONS:
                if not _countdown(cap, tracker, label):
                    aborted = True
                    break
                samples, keep_going = _record_hold(cap, tracker, label, target)
                all_samples.extend(samples)
                print(f"  {label}: recorded {len(samples)} samples (target={target})")
                if not keep_going:
                    aborted = True
                    break

    cap.release()
    cv2.destroyAllWindows()

    if aborted:
        print("\nAborted by user ('q' pressed). Nothing was saved.")
        return

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(all_samples, f, indent=2)

    print(f"\nSaved {len(all_samples)} labeled steering samples to {OUT_PATH}")
    by_target = {}
    for s in all_samples:
        by_target[s["target"]] = by_target.get(s["target"], 0) + 1
    print(f"Breakdown by target: {by_target}")


if __name__ == "__main__":
    main()
