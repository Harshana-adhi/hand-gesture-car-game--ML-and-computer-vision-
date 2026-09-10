"""
Phase 6 smoke test: confirms CVControlBridge's background thread runs
without blocking, and that its smoothed steering/action outputs respond to
real hand movement in front of the webcam.

Run this yourself, tilting your hand and changing gestures over ~15s:
    python phase6_check.py
"""
import time

from game.control_bridge import CVControlBridge


def main():
    print("Starting CVControlBridge (loads trained models + opens webcam)...")
    bridge = CVControlBridge()
    print("Bridge started. Move your hand: tilt left/right, open palm, make a fist.\n")

    try:
        start = time.time()
        loop_count = 0
        while time.time() - start < 15.0:
            loop_start = time.time()
            steering = bridge.get_steering()
            action = bridge.get_action()
            hand = bridge.is_hand_detected()
            print(f"steering={steering:+.2f}  action={action:9s}  hand_detected={hand}")
            loop_count += 1
            # This loop itself should never block on the camera/model --
            # get_steering/get_action/is_hand_detected just read shared state.
            elapsed = time.time() - loop_start
            time.sleep(max(0.0, 0.2 - elapsed))
        print(f"\nMain-thread loop ran {loop_count} iterations over 15s without blocking on CV work.")
    finally:
        bridge.close()
        print("Bridge closed cleanly.")


if __name__ == "__main__":
    main()
