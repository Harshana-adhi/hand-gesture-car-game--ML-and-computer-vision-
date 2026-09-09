"""
Phase 0 smoke test: confirm the webcam opens and MediaPipe HandLandmarker
can run against live frames. Prints per-frame detection status to the
console (no GUI window, so it works headlessly) for ~8 seconds.

Run this yourself with a hand in front of the webcam:
    python phase0_check.py
"""
import time

import cv2

from vision.hand_tracker import HandTracker


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: could not open webcam (device index 0).")
        return

    detected_frames = 0
    total_frames = 0

    with HandTracker() as tracker:
        start = time.time()
        while time.time() - start < 8.0:
            ok, frame = cap.read()
            if not ok:
                print("WARNING: failed to read frame from webcam.")
                continue

            total_frames += 1
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            landmarks = tracker.process(frame_rgb)

            hand_found = landmarks is not None
            if hand_found:
                detected_frames += 1

            print(f"frame {total_frames:4d}  hand_detected={hand_found}")

    cap.release()

    print("\n--- summary ---")
    print(f"total frames read: {total_frames}")
    print(f"frames with a hand detected: {detected_frames}")
    if detected_frames > 0:
        print("PASS: MediaPipe detected a hand from the live webcam feed.")
    else:
        print("No hand detected in this run. Re-run with a hand clearly "
              "in frame and reasonably lit to confirm.")


if __name__ == "__main__":
    main()
