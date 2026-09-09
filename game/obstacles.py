"""
Obstacle spawning and scrolling. Obstacles move downward at the car's
forward speed (relative motion: the car has a fixed screen position, so
the world scrolls past it), spawn rate and count increase gradually with
elapsed survival time for a difficulty curve.
"""
import random

import pygame

OBSTACLE_WIDTH = 50
OBSTACLE_HEIGHT = 80
OBSTACLE_COLOR = (220, 70, 70)

BASE_SPAWN_INTERVAL = 1.1   # seconds between spawns at t=0
MIN_SPAWN_INTERVAL = 0.35   # spawn interval never drops below this
SPAWN_RAMP_SECONDS = 60.0   # time to ramp from BASE to MIN spawn interval


class Obstacle:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def update(self, scroll_speed: float, dt: float):
        self.y += scroll_speed * dt

    def rect(self) -> pygame.Rect:
        return pygame.Rect(int(self.x - OBSTACLE_WIDTH / 2), int(self.y - OBSTACLE_HEIGHT / 2),
                            OBSTACLE_WIDTH, OBSTACLE_HEIGHT)

    def draw(self, surface: pygame.Surface):
        pygame.draw.rect(surface, OBSTACLE_COLOR, self.rect(), border_radius=6)


class ObstacleField:
    def __init__(self, road_left: float, road_right: float, screen_height: float):
        self.road_left = road_left
        self.road_right = road_right
        self.screen_height = screen_height
        self.obstacles = []
        self._time_since_spawn = 0.0
        self.elapsed = 0.0

    def _spawn_interval(self) -> float:
        t = min(self.elapsed / SPAWN_RAMP_SECONDS, 1.0)
        return BASE_SPAWN_INTERVAL + t * (MIN_SPAWN_INTERVAL - BASE_SPAWN_INTERVAL)

    def _spawn(self):
        half_w = OBSTACLE_WIDTH / 2.0
        x = random.uniform(self.road_left + half_w, self.road_right - half_w)
        self.obstacles.append(Obstacle(x, -OBSTACLE_HEIGHT))

    def update(self, scroll_speed: float, dt: float):
        self.elapsed += dt
        self._time_since_spawn += dt
        if self._time_since_spawn >= self._spawn_interval():
            self._time_since_spawn = 0.0
            self._spawn()

        for obstacle in self.obstacles:
            obstacle.update(scroll_speed, dt)

        self.obstacles = [o for o in self.obstacles if o.y - OBSTACLE_HEIGHT / 2 < self.screen_height + 20]

    def draw(self, surface: pygame.Surface):
        for obstacle in self.obstacles:
            obstacle.draw(surface)

    def check_collision(self, car_rect: pygame.Rect) -> bool:
        return any(car_rect.colliderect(o.rect()) for o in self.obstacles)

    def reset(self):
        self.obstacles.clear()
        self._time_since_spawn = 0.0
        self.elapsed = 0.0
