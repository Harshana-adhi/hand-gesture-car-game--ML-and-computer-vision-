"""
Player car entity: fixed vertical screen position, lateral (x) position
driven by the steering control signal, forward speed driven by the
accelerate/brake/neutral/boost action signal.
"""
import pygame

CAR_WIDTH = 50
CAR_HEIGHT = 80

MAX_LATERAL_SPEED = 420.0  # px/sec at full steering deflection

MIN_FORWARD_SPEED = 90.0
CRUISE_FORWARD_SPEED = 240.0
MAX_FORWARD_SPEED = 480.0
BOOST_FORWARD_SPEED = 680.0
BOOST_DURATION = 1.2
BOOST_COOLDOWN = 4.0

ACCEL_RATE = 260.0     # px/sec^2 toward MAX_FORWARD_SPEED
BRAKE_RATE = 500.0     # px/sec^2 toward MIN_FORWARD_SPEED (sharp)
NEUTRAL_DRIFT_RATE = 140.0  # px/sec^2 toward CRUISE_FORWARD_SPEED

CAR_COLOR = (50, 160, 230)
BOOST_COLOR = (255, 200, 60)


class Car:
    def __init__(self, road_left: float, road_right: float, screen_y: float):
        self.road_left = road_left
        self.road_right = road_right
        self.x = (road_left + road_right) / 2.0
        self.y = screen_y
        self.forward_speed = CRUISE_FORWARD_SPEED
        self._boost_time_left = 0.0
        self._boost_cooldown_left = 0.0

    def update(self, steering: float, action: str, dt: float):
        steering = max(-1.0, min(1.0, steering))
        self.x += steering * MAX_LATERAL_SPEED * dt
        half_w = CAR_WIDTH / 2.0
        self.x = max(self.road_left + half_w, min(self.road_right - half_w, self.x))

        if self._boost_cooldown_left > 0:
            self._boost_cooldown_left = max(0.0, self._boost_cooldown_left - dt)

        if self._boost_time_left > 0:
            self._boost_time_left = max(0.0, self._boost_time_left - dt)
            self.forward_speed = BOOST_FORWARD_SPEED
            return

        if action == "boost" and self._boost_cooldown_left <= 0:
            self._boost_time_left = BOOST_DURATION
            self._boost_cooldown_left = BOOST_DURATION + BOOST_COOLDOWN
            self.forward_speed = BOOST_FORWARD_SPEED
            return

        if action == "accelerate":
            self.forward_speed = min(MAX_FORWARD_SPEED, self.forward_speed + ACCEL_RATE * dt)
        elif action == "brake":
            self.forward_speed = max(MIN_FORWARD_SPEED, self.forward_speed - BRAKE_RATE * dt)
        else:  # neutral (or unrecognized action -- fail safe to neutral)
            if self.forward_speed > CRUISE_FORWARD_SPEED:
                self.forward_speed = max(CRUISE_FORWARD_SPEED, self.forward_speed - NEUTRAL_DRIFT_RATE * dt)
            else:
                self.forward_speed = min(CRUISE_FORWARD_SPEED, self.forward_speed + NEUTRAL_DRIFT_RATE * dt)

    @property
    def is_boosting(self) -> bool:
        return self._boost_time_left > 0

    @property
    def boost_ready(self) -> bool:
        return self._boost_cooldown_left <= 0

    def rect(self) -> pygame.Rect:
        return pygame.Rect(int(self.x - CAR_WIDTH / 2), int(self.y - CAR_HEIGHT / 2), CAR_WIDTH, CAR_HEIGHT)

    def draw(self, surface: pygame.Surface):
        color = BOOST_COLOR if self.is_boosting else CAR_COLOR
        pygame.draw.rect(surface, color, self.rect(), border_radius=6)
