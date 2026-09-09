"""
Endless driving game main loop. Phase 5: keyboard-controlled only
(KeyboardControlBridge) -- Phase 6 swaps in the real hand-tracking bridge
by changing one line (see CV_BRIDGE_TODO below).

Controls (keyboard stand-in): Left/Right = steer, Up = accelerate,
Down = brake, Space = boost.

Run:
    python -m game.main
"""
import json
import os
import sys

import pygame

from game.car import Car
from game.control_bridge import KeyboardControlBridge
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

        if not self.bridge.is_hand_detected():
            warn = self.big_font.render("HAND NOT DETECTED", True, (255, 60, 60))
            self.screen.blit(warn, warn.get_rect(center=(SCREEN_WIDTH // 2, 40)))

        if self.game_over:
            over_text = self.big_font.render("GAME OVER", True, (255, 80, 80))
            self.screen.blit(over_text, over_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 30)))
            restart_text = self.font.render("Press R to restart", True, TEXT_COLOR)
            self.screen.blit(restart_text, restart_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 30)))

        pygame.display.flip()

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
    # CV_BRIDGE_TODO (Phase 6): swap this for game.control_bridge.CVControlBridge()
    # (or whatever the trained-pipeline bridge class ends up being named) --
    # everything else in Game stays the same since both implement ControlBridge.
    bridge = KeyboardControlBridge()
    game = Game(bridge)
    game.run()


if __name__ == "__main__":
    sys.exit(main() or 0)
