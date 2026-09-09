"""
Control bridge interface: whatever supplies input to the game (keyboard
stub or the real trained hand-tracking pipeline) implements this same
interface, so the game loop never needs to know which one it's talking to.

Per CLAUDE.md Instruction #3: build and test the game with the keyboard
stub first (KeyboardControlBridge below); the real CV-driven bridge is
added in Phase 6 as a second implementation of the same interface, and
swapping main.py over to it is a one-line change.

Actions: "accelerate", "brake", "neutral", "boost".
"""
import pygame


class ControlBridge:
    """Interface every control source implements."""

    def get_steering(self) -> float:
        """Returns a steering value in [-1, 1]. 0 = centered."""
        raise NotImplementedError

    def get_action(self) -> str:
        """Returns one of 'accelerate', 'brake', 'neutral', 'boost'."""
        raise NotImplementedError

    def is_hand_detected(self) -> bool:
        """
        Whether a hand is currently being tracked. Always True for the
        keyboard stub; meaningful once the real CV bridge is wired in
        (Phase 6) so the debug overlay (Phase 7) can show it.
        """
        return True

    def close(self):
        """Release any resources (camera, background thread, etc.)."""
        pass


class KeyboardControlBridge(ControlBridge):
    """
    Temporary stand-in used to build and debug the game itself before the
    trained CV pipeline is wired in (Phase 6).

    Left/Right arrows -> steering. Up = accelerate, Down = brake,
    Space = boost, nothing pressed = neutral.
    """

    def get_steering(self) -> float:
        keys = pygame.key.get_pressed()
        steering = 0.0
        if keys[pygame.K_LEFT]:
            steering -= 1.0
        if keys[pygame.K_RIGHT]:
            steering += 1.0
        return steering

    def get_action(self) -> str:
        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE]:
            return "boost"
        if keys[pygame.K_UP]:
            return "accelerate"
        if keys[pygame.K_DOWN]:
            return "brake"
        return "neutral"
