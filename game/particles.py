"""
Simple particle trail for the boost effect (Phase 8 stretch goal, Section
11): a handful of fading circles spawned behind the car while boosting.
No external assets -- just pygame primitives, consistent with the rest of
the game's rendering.
"""
import random

import pygame

PARTICLE_LIFETIME = 0.45  # seconds
PARTICLES_PER_FRAME = 3
PARTICLE_COLOR = (255, 190, 60)
PARTICLE_SPEED_Y = 260.0   # px/sec, drifts away from the car (downward)
PARTICLE_SPEED_X_SPREAD = 60.0  # px/sec, random sideways drift
MAX_RADIUS = 6


class _Particle:
    __slots__ = ("x", "y", "vx", "vy", "age")

    def __init__(self, x, y, vx, vy):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.age = 0.0


class BoostParticles:
    def __init__(self):
        self._particles = []

    def spawn(self, x: float, y: float):
        for _ in range(PARTICLES_PER_FRAME):
            vx = random.uniform(-PARTICLE_SPEED_X_SPREAD, PARTICLE_SPEED_X_SPREAD)
            vy = PARTICLE_SPEED_Y * random.uniform(0.7, 1.3)
            self._particles.append(_Particle(x, y, vx, vy))

    def update(self, dt: float):
        for p in self._particles:
            p.age += dt
            p.x += p.vx * dt
            p.y += p.vy * dt
        self._particles = [p for p in self._particles if p.age < PARTICLE_LIFETIME]

    def draw(self, surface: pygame.Surface):
        for p in self._particles:
            life_frac = 1.0 - (p.age / PARTICLE_LIFETIME)
            radius = max(1, int(MAX_RADIUS * life_frac))
            alpha = max(0, min(255, int(220 * life_frac)))
            particle_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(particle_surf, (*PARTICLE_COLOR, alpha), (radius, radius), radius)
            surface.blit(particle_surf, (int(p.x - radius), int(p.y - radius)))

    def reset(self):
        self._particles.clear()
