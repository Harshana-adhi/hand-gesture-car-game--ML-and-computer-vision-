"""
Feature engineering on 21 MediaPipe hand landmarks (normalized x, y, z).

Landmark indices (MediaPipe Hands convention):
    0  = wrist
    1-4   = thumb (CMC, MCP, IP, TIP)
    5-8   = index (MCP, PIP, DIP, TIP)
    9-12  = middle (MCP, PIP, DIP, TIP)
    13-16 = ring (MCP, PIP, DIP, TIP)
    17-20 = pinky (MCP, PIP, DIP, TIP)
"""
import math

WRIST = 0
MIDDLE_MCP = 9

# (MCP joint index, fingertip index) per finger, thumb through pinky.
_FINGER_JOINTS = [
    (2, 4),    # thumb: MCP -> TIP
    (5, 8),    # index
    (9, 12),   # middle
    (13, 16),  # ring
    (17, 20),  # pinky
]


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def tilt_angle(landmarks):
    """
    Palm tilt angle in degrees, relative to vertical (image y-axis),
    computed from the wrist -> middle-finger-MCP vector.

    0 degrees = hand pointing straight up in frame (vertical, neutral).
    Positive = tilted right, negative = tilted left (image x grows right,
    y grows down, so we measure atan2 of the vector's x-component against
    its inverted y-component to get an intuitive "upright = 0" reading).
    """
    wrist = landmarks[WRIST]
    middle_mcp = landmarks[MIDDLE_MCP]
    dx = middle_mcp[0] - wrist[0]
    dy = middle_mcp[1] - wrist[1]
    # Image y grows downward; a hand pointing "up" has negative dy.
    # atan2(dx, -dy) gives 0 when pointing straight up, +/- as it tilts.
    angle_rad = math.atan2(dx, -dy)
    return math.degrees(angle_rad)


def finger_curl_vector(landmarks):
    """
    5-dimensional "openness" vector, one ratio per finger (thumb..pinky):
        ratio = dist(fingertip, wrist) / dist(finger_MCP, wrist)

    An extended finger has a ratio near/above 1 (tip is far from wrist,
    at least as far as the knuckle). A curled finger has a ratio well
    below 1 (tip has folded back toward the wrist).
    """
    wrist = landmarks[WRIST]
    ratios = []
    for mcp_idx, tip_idx in _FINGER_JOINTS:
        mcp = landmarks[mcp_idx]
        tip = landmarks[tip_idx]
        mcp_dist = _dist(mcp, wrist)
        tip_dist = _dist(tip, wrist)
        ratio = tip_dist / mcp_dist if mcp_dist > 1e-6 else 0.0
        ratios.append(ratio)
    return ratios
