"""
Shared car-icon rendering: a stylized top-down car built from pygame
primitives (body, windshield/rear window, wheels, lights) -- no external
image assets, per CLAUDE.md Instruction #4 (primitive shapes only, no
sprite art pipeline), just a less bare-rectangle look than a plain box.
"""
import pygame

WHEEL_COLOR = (25, 25, 28)
WINDOW_COLOR = (25, 35, 45)
LIGHT_ON_COLOR = (255, 240, 150)
LIGHT_OFF_COLOR = (120, 40, 40)


def draw_car_icon(surface: pygame.Surface, rect: pygame.Rect, body_color, facing: str = "up"):
    """
    Draws a simple top-down car inside `rect`. `facing` is "up" (nose
    toward the top of the screen, e.g. the player car) or "down" (nose
    toward the bottom, e.g. oncoming obstacles).
    """
    x, y, w, h = rect.x, rect.y, rect.width, rect.height

    # Body.
    pygame.draw.rect(surface, body_color, rect, border_radius=w // 4)

    # Windshield/rear window: offset toward the "nose" end.
    window_h = h * 0.28
    window_y = y + h * 0.16 if facing == "up" else y + h * 0.56
    window_rect = pygame.Rect(int(x + w * 0.14), int(window_y), int(w * 0.72), int(window_h))
    pygame.draw.rect(surface, WINDOW_COLOR, window_rect, border_radius=max(2, w // 8))

    # Wheels: four small dark rectangles at the corners, slightly inset.
    wheel_w = max(4, int(w * 0.14))
    wheel_h = max(8, int(h * 0.22))
    for wx in (x - wheel_w // 2 + 2, x + w - wheel_w // 2 - 2):
        for wy in (y + h * 0.12, y + h * 0.68):
            pygame.draw.rect(surface, WHEEL_COLOR, (wx, int(wy), wheel_w, wheel_h), border_radius=2)

    # Lights: two small circles at the nose end (headlights on player cars
    # facing away from the camera read as taillights, which is fine either way).
    light_r = max(2, int(w * 0.07))
    light_y = y + h * 0.08 if facing == "up" else y + h - h * 0.08
    color = LIGHT_ON_COLOR if facing == "up" else LIGHT_OFF_COLOR
    for lx in (x + w * 0.22, x + w * 0.78):
        pygame.draw.circle(surface, color, (int(lx), int(light_y)), light_r)
