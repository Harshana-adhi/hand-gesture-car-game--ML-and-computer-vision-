"""
Endless driving game main loop. Controlled by real hand tracking
(CVControlBridge) by default -- pass --keyboard to fall back to
Left/Right/Up/Down/Space for debugging the game mechanics in isolation
from the CV pipeline (Instruction #3).

Layout: a live camera preview on the left, the game on the right.

Flow: MENU (instructions screen) -> show TWO FINGERS or press Enter/Space
to start -> PLAYING -> collision -> GAME_OVER -> show TWO FINGERS or press R
to play again (goes straight back to PLAYING, not back to the menu).

Run:
    python -m game.main              # hand-gesture control (default)
    python -m game.main --keyboard   # keyboard fallback for debugging
"""
import argparse
import json
import os
import sys

import pygame

from game.car import CAR_HEIGHT, Car
from game.control_bridge import CVControlBridge, KeyboardControlBridge
from game.hand_icons import icon_fist, icon_four_fingers, icon_open_palm, icon_two_fingers
from game.obstacles import ObstacleField
from game.particles import BoostParticles

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
CAMERA_PANEL_WIDTH = 260
WINDOW_WIDTH = CAMERA_PANEL_WIDTH + SCREEN_WIDTH

ROAD_MARGIN = 200
ROAD_LEFT = ROAD_MARGIN
ROAD_RIGHT = SCREEN_WIDTH - ROAD_MARGIN
CAR_SCREEN_Y = SCREEN_HEIGHT - 120

FPS = 60

BG_COLOR = (30, 30, 35)
ROAD_COLOR = (60, 60, 66)
LANE_LINE_COLOR = (200, 200, 200)
TEXT_COLOR = (240, 240, 240)
CAMERA_BG_COLOR = (15, 15, 18)

STATE_MENU = "MENU"
STATE_PLAYING = "PLAYING"
STATE_GAME_OVER = "GAME_OVER"

HIGH_SCORE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "highscore.json")


def load_high_score() -> float:
    if os.path.exists(HIGH_SCORE_PATH):
        try:
            with open(HIGH_SCORE_PATH) as f:
                return json.load(f).get("high_score", 0.0)
        except (json.JSONDecodeError, OSError):
            return 0.0
    return 0.0


def save_high_score(value: float):
    with open(HIGH_SCORE_PATH, "w") as f:
        json.dump({"high_score": value}, f)


class Game:
    def __init__(self, control_bridge):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Hand-Gesture Car Game")
        # All existing game rendering targets this surface unchanged; it's
        # composited onto self.screen (to the right of the camera panel)
        # at the end of draw().
        self.game_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))

        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 24)
        self.small_font = pygame.font.SysFont("consolas", 18)
        self.big_font = pygame.font.SysFont("consolas", 48)

        self.bridge = control_bridge
        self.high_score = load_high_score()
        self._lane_scroll = 0.0
        self.state = STATE_MENU

        self._menu_entries = [
            (["TILT HAND", "steer"], icon_open_palm(rotation=22)),
            (["OPEN PALM", "accelerate"], icon_open_palm()),
            (["FIST", "brake"], icon_fist()),
            (["FOUR FINGERS", "boost"], icon_four_fingers()),
            (["TWO FINGERS", "start / restart"], icon_two_fingers()),
        ]

        self._reset()

    def _reset(self):
        self.car = Car(ROAD_LEFT, ROAD_RIGHT, CAR_SCREEN_Y)
        self.obstacles = ObstacleField(ROAD_LEFT, ROAD_RIGHT, SCREEN_HEIGHT)
        self.particles = BoostParticles()
        self.distance = 0.0

    def _start_game(self):
        self._reset()
        self.state = STATE_PLAYING

    def _score(self) -> float:
        return self.distance / 10.0

    def handle_events(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                if self.state == STATE_MENU and event.key in (pygame.K_RETURN, pygame.K_SPACE):
                    self._start_game()
                elif self.state == STATE_GAME_OVER and event.key == pygame.K_r:
                    self._start_game()
        return True

    def update(self, dt: float):
        if self.state == STATE_MENU:
            if self.bridge.is_hand_detected() and self.bridge.get_action() == "two_fingers":
                self._start_game()
            return

        if self.state == STATE_GAME_OVER:
            if self.bridge.is_hand_detected() and self.bridge.get_action() == "two_fingers":
                self._start_game()
            return

        steering = self.bridge.get_steering()
        action = self.bridge.get_action()

        self.car.update(steering, action, dt)
        self.obstacles.update(self.car.forward_speed, dt)
        self.distance += self.car.forward_speed * dt

        if self.car.is_boosting:
            self.particles.spawn(self.car.x, self.car.y + CAR_HEIGHT / 2)
        self.particles.update(dt)

        self._lane_scroll = (self._lane_scroll + self.car.forward_speed * dt) % 60

        if self.obstacles.check_collision(self.car.rect()):
            self.state = STATE_GAME_OVER
            score = self._score()
            if score > self.high_score:
                self.high_score = score
                save_high_score(self.high_score)

    def draw(self):
        if self.state == STATE_MENU:
            self._draw_menu()
        else:
            self._draw_game()

        self._draw_camera_panel()
        self.screen.blit(self.game_surface, (CAMERA_PANEL_WIDTH, 0))
        pygame.display.flip()

    def _draw_game(self):
        surface = self.game_surface
        surface.fill(BG_COLOR)
        pygame.draw.rect(surface, ROAD_COLOR, (ROAD_LEFT, 0, ROAD_RIGHT - ROAD_LEFT, SCREEN_HEIGHT))

        for x in (ROAD_LEFT, ROAD_RIGHT):
            pygame.draw.line(surface, LANE_LINE_COLOR, (x, 0), (x, SCREEN_HEIGHT), 3)

        y = int(self._lane_scroll) - 60
        while y < SCREEN_HEIGHT:
            pygame.draw.line(surface, LANE_LINE_COLOR,
                              ((ROAD_LEFT + ROAD_RIGHT) // 2, y),
                              ((ROAD_LEFT + ROAD_RIGHT) // 2, y + 30), 4)
            y += 60

        self.obstacles.draw(surface)
        self.particles.draw(surface)
        self.car.draw(surface)

        score_text = self.font.render(f"Score: {self._score():.0f}", True, TEXT_COLOR)
        surface.blit(score_text, (16, 16))
        speed_text = self.font.render(f"Speed: {self.car.forward_speed:.0f}", True, TEXT_COLOR)
        surface.blit(speed_text, (16, 44))
        high_text = self.font.render(f"High score: {self.high_score:.0f}", True, TEXT_COLOR)
        surface.blit(high_text, (16, 72))
        boost_text = self.font.render(
            "Boost ready" if self.car.boost_ready else "Boost cooling down", True, TEXT_COLOR)
        surface.blit(boost_text, (16, 100))

        self._draw_debug_overlay()

        if not self.bridge.is_hand_detected():
            warn = self.big_font.render("HAND NOT DETECTED", True, (255, 60, 60))
            surface.blit(warn, warn.get_rect(center=(SCREEN_WIDTH // 2, 150)))

        if self.state == STATE_GAME_OVER:
            over_text = self.big_font.render("GAME OVER", True, (255, 80, 80))
            surface.blit(over_text, over_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 50)))
            restart_text = self.font.render("Show TWO FINGERS or press R to play again",
                                             True, TEXT_COLOR)
            surface.blit(restart_text, restart_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 10)))

    def _draw_menu(self):
        surface = self.game_surface
        surface.fill(BG_COLOR)

        title = self.big_font.render("HAND-GESTURE CAR GAME", True, TEXT_COLOR)
        surface.blit(title, title.get_rect(center=(SCREEN_WIDTH // 2, 60)))
        subtitle = self.font.render("How to play", True, TEXT_COLOR)
        surface.blit(subtitle, subtitle.get_rect(center=(SCREEN_WIDTH // 2, 110)))

        n = len(self._menu_entries)
        slot_width = SCREEN_WIDTH // n
        icon_y = 220
        for i, (label_lines, icon) in enumerate(self._menu_entries):
            cx = slot_width * i + slot_width // 2
            icon_rect = icon.get_rect(center=(cx, icon_y))
            surface.blit(icon, icon_rect)
            for j, line in enumerate(label_lines):
                color = (255, 220, 120) if j == 0 else TEXT_COLOR
                text = self.small_font.render(line, True, color)
                surface.blit(text, text.get_rect(center=(cx, icon_y + 100 + j * 22)))

        note = self.small_font.render(
            "(Steering is calibrated to your hand -- recalibrate if it feels off)",
            True, (170, 170, 170))
        surface.blit(note, note.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT - 90)))

        if not self.bridge.is_hand_detected():
            warn = self.font.render("HAND NOT DETECTED", True, (255, 90, 90))
            surface.blit(warn, warn.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT - 60)))

        footer = self.font.render("Show TWO FINGERS to start  (or press ENTER)", True, (120, 255, 150))
        surface.blit(footer, footer.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT - 30)))

    def _draw_camera_panel(self):
        panel_rect = pygame.Rect(0, 0, CAMERA_PANEL_WIDTH, SCREEN_HEIGHT)
        pygame.draw.rect(self.screen, CAMERA_BG_COLOR, panel_rect)

        frame = self.bridge.get_camera_frame()
        if frame is not None:
            h, w = frame.shape[0], frame.shape[1]
            frame_surf = pygame.image.frombuffer(frame.tobytes(), (w, h), "RGB")
            scale = min(CAMERA_PANEL_WIDTH / w, SCREEN_HEIGHT / h)
            new_size = (int(w * scale), int(h * scale))
            frame_surf = pygame.transform.smoothscale(frame_surf, new_size)
            pos = ((CAMERA_PANEL_WIDTH - new_size[0]) // 2, (SCREEN_HEIGHT - new_size[1]) // 2)
            self.screen.blit(frame_surf, pos)
        else:
            msg1 = self.small_font.render("NO CAMERA", True, TEXT_COLOR)
            msg2 = self.small_font.render("(keyboard mode)", True, TEXT_COLOR)
            self.screen.blit(msg1, msg1.get_rect(center=(CAMERA_PANEL_WIDTH // 2, SCREEN_HEIGHT // 2 - 12)))
            self.screen.blit(msg2, msg2.get_rect(center=(CAMERA_PANEL_WIDTH // 2, SCREEN_HEIGHT // 2 + 12)))

        label = self.small_font.render("LIVE CAMERA", True, TEXT_COLOR)
        self.screen.blit(label, (10, 8))
        pygame.draw.rect(self.screen, LANE_LINE_COLOR, panel_rect, width=2)

    def _draw_debug_overlay(self):
        """
        Live proof that trained models -- not fixed rules -- are driving
        the car: raw vs. smoothed steering, the voted gesture with its
        confidence, and hand-detected status (CLAUDE.md Section 9).
        """
        surface = self.game_surface
        hand = self.bridge.is_hand_detected()
        raw_steer = self.bridge.get_raw_steering()
        smoothed_steer = self.bridge.get_steering()
        action = self.bridge.get_action()
        confidence = self.bridge.get_gesture_confidence()

        lines = [
            "-- DEBUG --",
            f"hand detected: {'YES' if hand else 'NO'}",
            f"steering raw:      {raw_steer:+.2f}",
            f"steering smoothed: {smoothed_steer:+.2f}",
            f"gesture: {action} ({confidence * 100:.0f}%)",
        ]
        panel_width = 280
        panel_x = SCREEN_WIDTH - panel_width - 16
        line_height = 22
        panel_height = 16 + line_height * len(lines)
        panel = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
        panel.fill((0, 0, 0, 140))
        surface.blit(panel, (panel_x, 10))

        hand_color = (100, 255, 100) if hand else (255, 90, 90)
        for i, line in enumerate(lines):
            color = hand_color if line.startswith("hand detected") else TEXT_COLOR
            text = self.small_font.render(line, True, color)
            surface.blit(text, (panel_x + 10, 18 + line_height * i))

    def run(self):
        running = True
        while running:
            dt = self.clock.tick(FPS) / 1000.0
            running = self.handle_events()
            self.update(dt)
            self.draw()
        self.bridge.close()
        pygame.quit()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--keyboard", action="store_true",
                         help="Use keyboard controls instead of hand tracking (debugging).")
    args = parser.parse_args()

    bridge = KeyboardControlBridge() if args.keyboard else CVControlBridge()
    game = Game(bridge)
    game.run()


if __name__ == "__main__":
    sys.exit(main() or 0)
