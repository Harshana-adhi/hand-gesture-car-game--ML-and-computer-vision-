"""
Phase 1 smoke test: prints live tilt-angle and finger-curl-vector values
as the hand moves, so we can visually confirm both features respond
sensibly (tilt angle swings as the hand tilts, curl values drop toward 0
on a fist and rise toward ~1 on an open palm).

Run this yourself, tilting your hand left/right and opening/closing it:
    python phase1_check.py
"""
import time

import cv2

from vision.features import finger_curl_vector, tilt_angle
from vision.hand_tracker import HandTracker


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: could not open webcam (device index 0).")
        return

    with HandTracker() as tracker:
        start = time.time()
        while time.time() - start < 12.0:
            ok, frame = cap.read()
            if not ok:
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            landmarks = tracker.process(frame_rgb)

            if landmarks is None:
                print("no hand detected")
                continue

            angle = tilt_angle(landmarks)
            curl = finger_curl_vector(landmarks)
            curl_str = " ".join(f"{v:.2f}" for v in curl)
            print(f"tilt={angle:7.2f} deg   curl[thumb,index,middle,ring,pinky]=[{curl_str}]")

    cap.release()


if __name__ == "__main__":
    main()
