"""
Simple stylized hand icons built from pygame primitives (palm + 5 finger
capsules), used on the "how to play" instructions screen. No external
image assets, consistent with the rest of the game's rendering.

Each icon is described the same way the real gesture classifier "sees" a
hand: a boolean per finger (thumb, index, middle, ring, pinky) for
extended vs. curled -- so the icon for a gesture matches the actual
feature (finger_curl_vector) that drives recognition of it.
"""
import pygame

PALM_COLOR = (235, 195, 160)
OUTLINE_COLOR = (90, 60, 40)

ICON_SIZE = (110, 140)


def render_hand_icon(extended, rotation: float = 0.0, color=PALM_COLOR) -> pygame.Surface:
    """
    extended: 5 booleans [thumb, index, middle, ring, pinky] -- True = finger
    drawn extended outward, False = drawn curled against the palm.
    rotation: degrees, counter-clockwise, applied to the whole icon (used
    to depict a tilted hand for the steering instruction).
    """
    w, h = ICON_SIZE
    surf = pygame.Surface((w, h), pygame.SRCALPHA)

    palm_w, palm_h = w * 0.5, h * 0.42
    palm_rect = pygame.Rect(0, 0, palm_w, palm_h)
    palm_rect.center = (w / 2, h * 0.68)
    pygame.draw.ellipse(surf, color, palm_rect)
    pygame.draw.ellipse(surf, OUTLINE_COLOR, palm_rect, width=2)

    # Four fingers (index..pinky) fan across the top of the palm.
    finger_names = ["index", "middle", "ring", "pinky"]
    finger_w = palm_w * 0.19
    top_y = palm_rect.top
    xs = [palm_rect.left + palm_w * frac for frac in (0.14, 0.38, 0.62, 0.86)]
    for i, (name, fx) in enumerate(zip(finger_names, xs)):
        is_extended = extended[i + 1]
        finger_h = h * 0.4 if is_extended else h * 0.09
        finger_rect = pygame.Rect(0, 0, finger_w, finger_h)
        finger_rect.midbottom = (fx, top_y + h * 0.05)
        pygame.draw.rect(surf, color, finger_rect, border_radius=int(finger_w / 2))
        pygame.draw.rect(surf, OUTLINE_COLOR, finger_rect, width=2, border_radius=int(finger_w / 2))

    # Thumb sticks out to the side.
    thumb_extended = extended[0]
    thumb_w, thumb_h = w * 0.16, (h * 0.32 if thumb_extended else h * 0.14)
    thumb_surf = pygame.Surface((thumb_w, thumb_h), pygame.SRCALPHA)
    pygame.draw.rect(thumb_surf, color, thumb_surf.get_rect(), border_radius=int(thumb_w / 2))
    pygame.draw.rect(thumb_surf, OUTLINE_COLOR, thumb_surf.get_rect(), width=2, border_radius=int(thumb_w / 2))
    thumb_surf = pygame.transform.rotate(thumb_surf, 35 if thumb_extended else 10)
    thumb_pos = (palm_rect.left - thumb_surf.get_width() * 0.55, palm_rect.centery - thumb_surf.get_height() * 0.35)
    surf.blit(thumb_surf, thumb_pos)

    if rotation:
        surf = pygame.transform.rotate(surf, rotation)
    return surf


# Pre-built icons for each gesture used in the game.
def icon_open_palm(rotation=0.0):
    return render_hand_icon([True, True, True, True, True], rotation=rotation)


def icon_fist(rotation=0.0):
    return render_hand_icon([False, False, False, False, False], rotation=rotation)


def icon_four_fingers(rotation=0.0):
    return render_hand_icon([False, True, True, True, True], rotation=rotation)


def icon_two_fingers(rotation=0.0):
    return render_hand_icon([False, True, True, False, False], rotation=rotation)
