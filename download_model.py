"""
One-time setup: downloads the official MediaPipe HandLandmarker model asset
(hand_landmarker.task, ~7.8MB) from Google's public model storage into
models/assets/. Required because mediapipe's newer Tasks API loads this
model from a local file instead of bundling/auto-downloading it internally.

Run once:
    python download_model.py
"""
import os
import urllib.request

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
DEST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "assets")
DEST_PATH = os.path.join(DEST_DIR, "hand_landmarker.task")


def main():
    os.makedirs(DEST_DIR, exist_ok=True)
    if os.path.exists(DEST_PATH):
        print(f"Model already present at {DEST_PATH}, skipping download.")
        return
    print(f"Downloading hand_landmarker.task from {MODEL_URL} ...")
    urllib.request.urlretrieve(MODEL_URL, DEST_PATH)
    print(f"Saved to {DEST_PATH}")


if __name__ == "__main__":
    main()
