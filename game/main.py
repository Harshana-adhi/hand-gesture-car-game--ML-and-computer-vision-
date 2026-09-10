"""
Endless driving game main loop. Controlled by real hand tracking
(CVControlBridge) by default -- pass --keyboard to fall back to
Left/Right/Up/Down/Space for debugging the game mechanics in isolation
from the CV pipeline (Instruction #3).

Layout: a live camera preview on the left, the game on the right.

Flow: MENU (instructions screen) -> show TWO FINGERS or press Enter/Space
to start -> PLAYING -> collision -> GAME_OVER -> show TWO FINGERS or press R
to play again (goes straight back to PLAYING, not back to the menu).

From MENU, press C to open Settings -> Recalibrate any steering/gesture
target, choosing whether to delete existing data or add to it, then the
game re-runs the corresponding training script and reports the result.
Settings navigation is keyboard-only (Up/Down/Enter/Esc) -- it's a setup
screen, not part of the hands-free driving loop.

Run:
    python -m game.main              # hand-gesture control (default)
    python -m game.main --keyboard   # keyboard fallback for debugging
"""
import argparse
import json
import os
import subprocess
import sys

import pygame

from game.calibration_ui import (
    CALIBRATION_TARGETS,
    CalibrationSession,
    GESTURE_INSTRUCTIONS,
    GESTURE_TARGET_SAMPLES,
    existing_sample_count,
)
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
HIGHLIGHT_COLOR = (255, 220, 120)

STATE_MENU = "MENU"
STATE_PLAYING = "PLAYING"
STATE_GAME_OVER = "GAME_OVER"
STATE_SETTINGS = "SETTINGS"
STATE_DELETE_CHOICE = "DELETE_CHOICE"
STATE_CALIBRATING = "CALIBRATING"
STATE_TRAINING = "TRAINING"
STATE_TRAINING_RESULT = "TRAINING_RESULT"

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HIGH_SCORE_PATH = os.path.join(_ROOT, "highscore.json")

CALIBRATION_DISPLAY_NAMES = {
    "steering": "Steering (tilt left / center / right)",
    "accelerate": "Accelerate (open palm)",
    "brake": "Brake (fist)",
    "neutral": "Neutral (resting hand)",
    "boost": "Boost (four fingers)",
    "two_fingers": "Two fingers (start / restart)",
}


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


def _run_training(target: str) -> list:
    """Runs the appropriate retrain script as a subprocess and returns a
    short list of result lines pulled from its stdout, for on-screen display."""
    module = "models.train_steering_model" if target == "steering" else "models.train_gesture_model"
    try:
        result = subprocess.run(
            [sys.executable, "-m", module], cwd=_ROOT,
            capture_output=True, text=True, timeout=120,
        )
    except Exception as exc:
        return [f"Training failed to launch: {exc}"]

    if result.returncode != 0:
        tail = (result.stderr or "").strip().splitlines()[-5:]
        return [f"Training failed (exit {result.returncode}):"] + tail

    lines = result.stdout.strip().splitlines()
    if target == "steering":
        relevant = [l.strip() for l in lines if l.strip().startswith("Selected:") or "MAE" in l]
    else:
        relevant = [l.strip() for l in lines if "accuracy" in l.lower() or "Macro F1" in l]
    return relevant or ["(retrained -- no summary line captured, check console)"]


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

        # Settings / recalibration state.
        self._settings_items = CALIBRATION_TARGETS + ["Back to Menu"]
        self._settings_index = 0
        self._delete_choice_index = 0
        self._pending_target = None
        self._calibration_session = None
        self._training_result_lines = []
        self._training_target = None

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

    def _camera_available(self) -> bool:
        return self.bridge.get_camera_frame() is not None

    # ---- event handling ----

    def handle_events(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if not self._handle_keydown(event.key):
                    return False
        return True

    def _handle_keydown(self, key) -> bool:
        """Returns False to quit the whole game."""
        if self.state == STATE_MENU:
            if key == pygame.K_ESCAPE:
                return False
            if key in (pygame.K_RETURN, pygame.K_SPACE):
                self._start_game()
            elif key == pygame.K_c:
                self.state = STATE_SETTINGS
                self._settings_index = 0

        elif self.state == STATE_GAME_OVER:
            if key == pygame.K_ESCAPE:
                return False
            if key == pygame.K_r:
                self._start_game()

        elif self.state == STATE_PLAYING:
            if key == pygame.K_ESCAPE:
                return False

        elif self.state == STATE_SETTINGS:
            if key == pygame.K_ESCAPE:
                self.state = STATE_MENU
            elif key == pygame.K_UP:
                self._settings_index = (self._settings_index - 1) % len(self._settings_items)
            elif key == pygame.K_DOWN:
                self._settings_index = (self._settings_index + 1) % len(self._settings_items)
            elif key == pygame.K_RETURN:
                choice = self._settings_items[self._settings_index]
                if choice == "Back to Menu":
                    self.state = STATE_MENU
                elif self._camera_available():
                    self._pending_target = choice
                    self._delete_choice_index = 0
                    self.state = STATE_DELETE_CHOICE

        elif self.state == STATE_DELETE_CHOICE:
            if key == pygame.K_ESCAPE:
                self.state = STATE_SETTINGS
            elif key in (pygame.K_UP, pygame.K_DOWN):
                self._delete_choice_index = 1 - self._delete_choice_index
            elif key == pygame.K_1:
                self._delete_choice_index = 0
                self._begin_calibration()
            elif key == pygame.K_2:
                self._delete_choice_index = 1
                self._begin_calibration()
            elif key == pygame.K_RETURN:
                self._begin_calibration()

        elif self.state == STATE_CALIBRATING:
            if key == pygame.K_ESCAPE:
                self._calibration_session.abort()
                self.state = STATE_SETTINGS

        elif self.state == STATE_TRAINING_RESULT:
            if key in (pygame.K_RETURN, pygame.K_ESCAPE):
                self.state = STATE_SETTINGS

        return True

    def _begin_calibration(self):
        delete_existing = self._delete_choice_index == 0
        self._calibration_session = CalibrationSession(self._pending_target, delete_existing)
        self.state = STATE_CALIBRATING

    # ---- update ----

    def update(self, dt: float):
        if self.state == STATE_MENU:
            if self.bridge.is_hand_detected() and self.bridge.get_action() == "two_fingers":
                self._start_game()
            return

        if self.state == STATE_GAME_OVER:
            if self.bridge.is_hand_detected() and self.bridge.get_action() == "two_fingers":
                self._start_game()
            return

        if self.state in (STATE_SETTINGS, STATE_DELETE_CHOICE):
            return

        if self.state == STATE_CALIBRATING:
            session = self._calibration_session
            session.update(dt, self.bridge)
            if session.done:
                session.save()
                self._training_target = session.target
                self.state = STATE_TRAINING
            return

        if self.state == STATE_TRAINING:
            # Drawn once with "Training..." before this blocking call, per
            # the run() loop's handle_events -> update -> draw order.
            self._training_result_lines = _run_training(self._training_target)
            self.state = STATE_TRAINING_RESULT
            return

        if self.state == STATE_TRAINING_RESULT:
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

    # ---- drawing ----

    def draw(self):
        if self.state == STATE_MENU:
            self._draw_menu()
        elif self.state == STATE_SETTINGS:
            self._draw_settings()
        elif self.state == STATE_DELETE_CHOICE:
            self._draw_delete_choice()
        elif self.state == STATE_CALIBRATING:
            self._draw_calibrating()
        elif self.state == STATE_TRAINING:
            self._draw_training()
        elif self.state == STATE_TRAINING_RESULT:
            self._draw_training_result()
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
                color = HIGHLIGHT_COLOR if j == 0 else TEXT_COLOR
                text = self.small_font.render(line, True, color)
                surface.blit(text, text.get_rect(center=(cx, icon_y + 100 + j * 22)))

        note = self.small_font.render(
            "(Steering is calibrated to your hand -- press C to recalibrate)",
            True, (170, 170, 170))
        surface.blit(note, note.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT - 90)))

        if not self.bridge.is_hand_detected():
            warn = self.font.render("HAND NOT DETECTED", True, (255, 90, 90))
            surface.blit(warn, warn.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT - 60)))

        footer = self.font.render("Show TWO FINGERS to start  (or press ENTER, or C for Settings)",
                                   True, (120, 255, 150))
        surface.blit(footer, footer.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT - 30)))

    def _draw_settings(self):
        surface = self.game_surface
        surface.fill(BG_COLOR)

        title = self.big_font.render("SETTINGS", True, TEXT_COLOR)
        surface.blit(title, title.get_rect(center=(SCREEN_WIDTH // 2, 60)))
        subtitle = self.font.render("Recalibrate a control", True, TEXT_COLOR)
        surface.blit(subtitle, subtitle.get_rect(center=(SCREEN_WIDTH // 2, 105)))

        if not self._camera_available():
            warn = self.font.render("Calibration requires camera mode", True, (255, 90, 90))
            surface.blit(warn, warn.get_rect(center=(SCREEN_WIDTH // 2, 160)))
            warn2 = self.small_font.render("(restart without --keyboard)", True, (255, 90, 90))
            surface.blit(warn2, warn2.get_rect(center=(SCREEN_WIDTH // 2, 186)))

        start_y = 220
        for i, item in enumerate(self._settings_items):
            selected = i == self._settings_index
            if item == "Back to Menu":
                label = "Back to Menu"
            else:
                count = existing_sample_count(item)
                label = f"{CALIBRATION_DISPLAY_NAMES.get(item, item)}  [{count} samples]"
            color = HIGHLIGHT_COLOR if selected else TEXT_COLOR
            prefix = "> " if selected else "  "
            text = self.small_font.render(prefix + label, True, color)
            surface.blit(text, (60, start_y + i * 30))

        footer = self.small_font.render("Up/Down to select, Enter to choose, Esc to go back",
                                         True, (170, 170, 170))
        surface.blit(footer, footer.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT - 30)))

    def _draw_delete_choice(self):
        surface = self.game_surface
        surface.fill(BG_COLOR)

        target = self._pending_target
        title = self.big_font.render(CALIBRATION_DISPLAY_NAMES.get(target, target), True, TEXT_COLOR)
        surface.blit(title, title.get_rect(center=(SCREEN_WIDTH // 2, 80)))

        count = existing_sample_count(target)
        info = self.font.render(f"Currently {count} samples on disk", True, TEXT_COLOR)
        surface.blit(info, info.get_rect(center=(SCREEN_WIDTH // 2, 130)))

        options = [
            "1: Delete current data and recalibrate",
            "2: Keep current data and add more",
        ]
        for i, option in enumerate(options):
            selected = i == self._delete_choice_index
            color = HIGHLIGHT_COLOR if selected else TEXT_COLOR
            prefix = "> " if selected else "  "
            text = self.font.render(prefix + option, True, color)
            surface.blit(text, text.get_rect(center=(SCREEN_WIDTH // 2, 220 + i * 44)))

        footer = self.small_font.render("Press 1 or 2, or Up/Down + Enter -- Esc to go back",
                                         True, (170, 170, 170))
        surface.blit(footer, footer.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT - 30)))

    def _draw_calibrating(self):
        surface = self.game_surface
        surface.fill(BG_COLOR)
        session = self._calibration_session

        title = self.big_font.render(CALIBRATION_DISPLAY_NAMES.get(session.target, session.target),
                                      True, TEXT_COLOR)
        surface.blit(title, title.get_rect(center=(SCREEN_WIDTH // 2, 70)))

        label = session.current_label
        if session.is_steering:
            step_num, total_steps = session.current_sweep_progress
            progress = self.small_font.render(f"Step {step_num}/{total_steps}", True, (170, 170, 170))
            surface.blit(progress, progress.get_rect(center=(SCREEN_WIDTH // 2, 110)))
        else:
            instruction = GESTURE_INSTRUCTIONS.get(session.target, "")
            instr_text = self.small_font.render(instruction, True, (170, 170, 170))
            surface.blit(instr_text, instr_text.get_rect(center=(SCREEN_WIDTH // 2, 110)))

        if session.phase == "countdown":
            remaining = max(0.0, 2.0 - session.phase_timer)
            big = self.big_font.render(f"Get ready: {label}", True, (0, 200, 255))
            surface.blit(big, big.get_rect(center=(SCREEN_WIDTH // 2, 260)))
            sub = self.font.render(f"starting in {remaining:.1f}s", True, TEXT_COLOR)
            surface.blit(sub, sub.get_rect(center=(SCREEN_WIDTH // 2, 320)))
        else:
            hand_ok = self.bridge.is_hand_detected()
            color = (0, 255, 0) if hand_ok else (255, 60, 60)
            big = self.big_font.render(f"Hold: {label}", True, color)
            surface.blit(big, big.get_rect(center=(SCREEN_WIDTH // 2, 260)))
            if session.is_steering:
                sub_text = f"{len(session.samples)} samples collected so far"
            else:
                sub_text = f"{len(session.samples)}/{GESTURE_TARGET_SAMPLES} samples"
                if not hand_ok:
                    sub_text += "  (NO HAND DETECTED)"
            sub = self.font.render(sub_text, True, TEXT_COLOR)
            surface.blit(sub, sub.get_rect(center=(SCREEN_WIDTH // 2, 320)))

        footer = self.small_font.render("Esc to abort without saving", True, (170, 170, 170))
        surface.blit(footer, footer.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT - 30)))

    def _draw_training(self):
        surface = self.game_surface
        surface.fill(BG_COLOR)
        text = self.big_font.render("Training...", True, TEXT_COLOR)
        surface.blit(text, text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)))

    def _draw_training_result(self):
        surface = self.game_surface
        surface.fill(BG_COLOR)
        title = self.big_font.render("Retrained", True, (120, 255, 150))
        surface.blit(title, title.get_rect(center=(SCREEN_WIDTH // 2, 100)))

        for i, line in enumerate(self._training_result_lines):
            text = self.small_font.render(line, True, TEXT_COLOR)
            surface.blit(text, text.get_rect(center=(SCREEN_WIDTH // 2, 180 + i * 26)))

        footer = self.font.render("Press ENTER to return to Settings", True, (170, 170, 170))
        surface.blit(footer, footer.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT - 60)))

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
