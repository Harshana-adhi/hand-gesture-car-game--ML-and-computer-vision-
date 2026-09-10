"""
Endless driving game main loop. Controlled by real hand tracking
(CVControlBridge) by default -- pass --keyboard to fall back to
Left/Right/Up/Down/Space for debugging the game mechanics in isolation
from the CV pipeline (Instruction #3).

Run:
    python -m game.main              # hand-gesture control (default)
    python -m game.main --keyboard   # keyboard fallback for debugging
"""
import argparse
import json
import os
import sys

import pygame

from game.car import Car
from game.control_bridge import CVControlBridge, KeyboardControlBridge
from game.obstacles import ObstacleField

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
ROAD_MARGIN = 200
ROAD_LEFT = ROAD_MARGIN
ROAD_RIGHT = SCREEN_WIDTH - ROAD_MARGIN
CAR_SCREEN_Y = SCREEN_HEIGHT - 120

FPS = 60

BG_COLOR = (30, 30, 35)
ROAD_COLOR = (60, 60, 66)
LANE_LINE_COLOR = (200, 200, 200)
TEXT_COLOR = (240, 240, 240)

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
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Hand-Gesture Car Game")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 24)
        self.small_font = pygame.font.SysFont("consolas", 18)
        self.big_font = pygame.font.SysFont("consolas", 48)

        self.bridge = control_bridge
        self.high_score = load_high_score()
        self._lane_scroll = 0.0
        self._reset()

    def _reset(self):
        self.car = Car(ROAD_LEFT, ROAD_RIGHT, CAR_SCREEN_Y)
        self.obstacles = ObstacleField(ROAD_LEFT, ROAD_RIGHT, SCREEN_HEIGHT)
        self.distance = 0.0
        self.game_over = False

    def _score(self) -> float:
        return self.distance / 10.0

    def handle_events(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                if event.key == pygame.K_r and self.game_over:
                    self._reset()
        return True

    def update(self, dt: float):
        if self.game_over:
            return

        steering = self.bridge.get_steering()
        action = self.bridge.get_action()

        self.car.update(steering, action, dt)
        self.obstacles.update(self.car.forward_speed, dt)
        self.distance += self.car.forward_speed * dt

        self._lane_scroll = (self._lane_scroll + self.car.forward_speed * dt) % 60

        if self.obstacles.check_collision(self.car.rect()):
            self.game_over = True
            score = self._score()
            if score > self.high_score:
                self.high_score = score
                save_high_score(self.high_score)

    def draw(self):
        self.screen.fill(BG_COLOR)
        pygame.draw.rect(self.screen, ROAD_COLOR, (ROAD_LEFT, 0, ROAD_RIGHT - ROAD_LEFT, SCREEN_HEIGHT))

        for x in (ROAD_LEFT, ROAD_RIGHT):
            pygame.draw.line(self.screen, LANE_LINE_COLOR, (x, 0), (x, SCREEN_HEIGHT), 3)

        y = int(self._lane_scroll) - 60
        while y < SCREEN_HEIGHT:
            pygame.draw.line(self.screen, LANE_LINE_COLOR,
                              ((ROAD_LEFT + ROAD_RIGHT) // 2, y),
                              ((ROAD_LEFT + ROAD_RIGHT) // 2, y + 30), 4)
            y += 60

        self.obstacles.draw(self.screen)
        self.car.draw(self.screen)

        score_text = self.font.render(f"Score: {self._score():.0f}", True, TEXT_COLOR)
        self.screen.blit(score_text, (16, 16))
        speed_text = self.font.render(f"Speed: {self.car.forward_speed:.0f}", True, TEXT_COLOR)
        self.screen.blit(speed_text, (16, 44))
        high_text = self.font.render(f"High score: {self.high_score:.0f}", True, TEXT_COLOR)
        self.screen.blit(high_text, (16, 72))
        boost_text = self.font.render(
            "Boost ready" if self.car.boost_ready else "Boost cooling down", True, TEXT_COLOR)
        self.screen.blit(boost_text, (16, 100))

        self._draw_debug_overlay()

        if not self.bridge.is_hand_detected():
            warn = self.big_font.render("HAND NOT DETECTED", True, (255, 60, 60))
            self.screen.blit(warn, warn.get_rect(center=(SCREEN_WIDTH // 2, 150)))

        if self.game_over:
            over_text = self.big_font.render("GAME OVER", True, (255, 80, 80))
            self.screen.blit(over_text, over_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 30)))
            restart_text = self.font.render("Press R to restart", True, TEXT_COLOR)
            self.screen.blit(restart_text, restart_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 30)))

        pygame.display.flip()

    def _draw_debug_overlay(self):
        """
        Live proof that trained models -- not fixed rules -- are driving
        the car: raw vs. smoothed steering, the voted gesture with its
        confidence, and hand-detected status (CLAUDE.md Section 9).
        """
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
        panel_width = 260
        panel_x = SCREEN_WIDTH - panel_width - 10
        line_height = 22
        panel_height = 16 + line_height * len(lines)
        panel = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
        panel.fill((0, 0, 0, 140))
        self.screen.blit(panel, (panel_x, 10))

        hand_color = (100, 255, 100) if hand else (255, 90, 90)
        for i, line in enumerate(lines):
            color = hand_color if line.startswith("hand detected") else TEXT_COLOR
            text = self.small_font.render(line, True, color)
            self.screen.blit(text, (panel_x + 10, 18 + line_height * i))

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
