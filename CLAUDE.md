# Hand-Gesture Controlled Car Game — Project Brief & Build Plan

**Purpose of this document:** this is a full context feed for Claude Code. Paste or place it at the root of the project repo (e.g. as `CLAUDE.md` or `PROJECT_BRIEF.md`) before starting the build. It contains the concept, the technical spec, the phased development plan, and direct instructions for Claude Code to follow while building.

---

## 1. What this project is

A simple endless driving game — dodge oncoming obstacles, survive as long as possible — controlled **entirely by one hand in front of a webcam**, no keyboard at all. Tilt your hand to steer; open your palm to accelerate; make a fist to brake.

**The real-world problem this technique addresses:** gesture-based, hands-free control interfaces are an active area in accessible computing and automotive HMI (human-machine interface) research right now — the same underlying technique (calibrated, trained gesture-to-control mapping rather than fixed hardcoded rules) shows up in accessible-driving-assist prototypes and touchless vehicle interfaces.

**What makes this a genuine ML project, not a wrapper:** MediaPipe gives you hand *landmarks* — it has no concept of "steering" or "accelerating." Every part of the control system is built and trained by you, and it has two genuinely different ML problems in it, not one:

1. **Continuous control (regression)** — mapping your hand's tilt angle to a steering value is a *trained, calibrated regression*, not a hardcoded formula. You collect a short calibration sequence (tilt fully left / center / fully right), fit a regression model to it, and the game steers using *your* personal, possibly non-linear, possibly asymmetric comfortable tilt range — not an assumed-linear guess.
2. **Discrete control (classification)** — accelerate/brake/neutral (and optionally boost) are recognized by a gesture classifier trained on your own recorded hand shapes, the same technique family as sign-language or gesture-recognition systems.

Having both a regression problem and a classification problem, each genuinely trained and evaluated, is the technical core of this project — and it's a nice contrast with most portfolio projects, which only ever do one or the other.

---

### A key assumption: one calibrated hand, consistent setup

- **This is single-hand control, calibrated to you.** Steering is trained on your own tilt-angle range — a different person's hand geometry (or your other hand) won't steer correctly without recalibrating. This is expected, not a bug — say so plainly in the demo, the same way any personally-calibrated control system would.
- **Keep lighting and camera framing reasonably consistent** between calibration and play, for the same reason as the earlier landmark-based projects: MediaPipe's landmark-detection quality (not your trained models) is what degrades in poor lighting.
- **Pick a dominant hand and stick with it** for calibration and play in a single session. Recalibrating for the other hand takes under a minute if you want to switch.

---

## 2. How the system works, end to end

```
Data                     →  Preprocessing          →  Feature engineering        →  ML models                              →  Evaluation             →  Inference                  →  Application/UI
──────────────────────      ─────────────────────     ─────────────────────────     ─────────────────────────────────────     ──────────────────────    ─────────────────────────     ──────────────────────────
Webcam hand-tracking         MediaPipe Hands           Steering: wrist→finger        Regression model (linear or small          Held-out calibration     Live per-frame steering       Pygame driving game:
during a guided               (frozen, pretrained)      angle, normalized               MLP) fit on your calibration data          points: regression         value (smoothed) +             hand-tilt steers the
calibration sequence           extracts 21 hand          Gesture: per-finger             mapping tilt → steering value              error; gesture               majority-voted gesture         car, gestures control
+ recorded gesture              landmarks per              curl ratios (5-dim              Classifier (Random Forest/SVM)             classifier's held-out         each game frame, fed             speed, obstacles
samples                          frame                      "openness" vector)              mapping curl vector → gesture              accuracy/F1, confusion         via a threaded control          scroll and score
                                                                                                                                          matrix                         bridge
```

---

## 3. Instructions for Claude Code — read this first

These are the operating rules for building this project. Follow them for the whole session, not just the first prompt.

1. **The ML has to be real, not simulated.** Never fabricate calibration data or gesture samples, and never hardcode or invent evaluation numbers. If real calibration data isn't collected yet, build the parts that don't need it and wait.
2. **Calibration and gesture-sample collection are human steps you cannot do yourself.** You cannot hold up a hand to a webcam. When the plan reaches Phase 3 (Section 8), stop and clearly ask the user to run the calibration scripts. Do not train the steering regression or gesture classifier until real recorded data exists on disk.
3. **Build and test the game mechanics with keyboard input first, then swap the input source to the trained CV pipeline.** This decouples "is the game fun and bug-free" from "is the hand-tracking control working" — debug them separately, not simultaneously. `game/control_bridge.py` should expose a simple interface (`get_steering() -> float`, `get_action() -> str`) that both a keyboard-driven stub and the real CV pipeline can implement identically, so swapping one for the other is a one-line change.
4. **Use pygame for the game itself, not a full game engine.** This is a one-day build — a lightweight 2D loop with primitive shapes (rectangles/circles) for the car and obstacles is entirely sufficient; do not spend time on sprite art or a heavier engine unless core mechanics and control are already solid and there's time left over.
5. **Run hand-tracking inference in a way that doesn't stall the game's frame rate.** MediaPipe inference and pygame's render loop should not block each other — run the CV pipeline in a background thread that continuously updates shared "current steering value" / "current gesture" state, and have the game loop simply read the latest values each frame. Don't make the game loop wait on a webcam frame.
6. **Apply temporal smoothing to both control signals** — an exponential moving average on the raw steering angle, and a short majority-vote window (e.g. the last 5 classifier predictions) on the gesture classification — before they reach the game. Raw per-frame values are visibly jittery and make the game feel broken even when the underlying models are working correctly.
7. **Both models must be evaluated with real numbers**, not just "does the game feel okay": regression error on held-out calibration points for steering, and accuracy/F1 + confusion matrix on held-out gesture samples for the classifier.
8. **Handle "hand not detected" gracefully everywhere** — default to neutral/no input and show an on-screen warning, never crash or throw when a frame has no detected hand (this will happen constantly in normal play as the hand moves in and out of frame).
9. **Don't add a login system, cloud deployment, leaderboard server, or persistence beyond local high-score storage.** This is a local single-machine prototype for one player.
10. **Ask before making a scope decision that trades off correctness for speed** — e.g. skipping the neutral-gesture calibration class, using fewer than two calibration sweeps, or dropping the smoothing filters. Small game-feel decisions (obstacle speed, spawn rate, visuals) don't need sign-off.
11. **Never run `git commit` (or any command that changes repository state — `git add`, `git push`, etc.) yourself.** At the end of each phase, once its "Definition of Done" is met, suggest a commit message summarizing what was built and how it was verified, and stop there — the user reviews the diff and commits manually. This applies for the whole project, not just the first phase.

---

## 4. Tech stack

| Layer | Choice | Why |
|---|---|---|
| Hand tracking | MediaPipe Hands (Google, free, pretrained, CPU) | Real-time 21-point hand landmarks, no custom training needed for detection itself |
| Webcam capture | OpenCV (`cv2.VideoCapture`) | Standard, simple, pairs directly with MediaPipe |
| Game | pygame | Fastest path to a working 2D game loop; huge amount of reference material; no unnecessary engine overhead |
| Steering regression | scikit-learn (`LinearRegression`, compared against a small `MLPRegressor`) | Keep both — report which one actually generalizes better on held-out calibration points, don't assume the fancier one wins |
| Gesture classifier | scikit-learn (`RandomForestClassifier` or `SVC`) | Fast to train on a small hand-shape feature set |
| Threading | Python's built-in `threading` | Decouples CV inference from the game's render loop |
| Evaluation | matplotlib | Calibration-fit plot, confusion matrix |

```
pip install opencv-python mediapipe pygame scikit-learn numpy matplotlib
```

---

## 5. Repo structure

```
gesture-car-game/
├── README.md
├── CLAUDE.md                       # this document
├── requirements.txt
├── vision/
│   ├── hand_tracker.py             # MediaPipe wrapper: frame → landmarks
│   └── features.py                 # tilt-angle feature; per-finger curl ("openness") feature vector
├── calibration/
│   ├── calibrate_steering.py       # guided capture: hold left/center/right, saves labeled samples
│   ├── calibrate_gestures.py       # guided capture: accelerate/brake/neutral/(boost) samples
│   └── data/
│       ├── steering_calibration.json
│       └── gestures/
│           ├── accelerate/
│           ├── brake/
│           ├── neutral/
│           └── boost/              # optional
├── models/
│   ├── train_steering_model.py     # fits + compares regression models, saves the better one
│   ├── train_gesture_model.py      # trains + evaluates the gesture classifier
│   └── saved/
│       ├── steering_model.joblib
│       └── gesture_model.joblib
├── game/
│   ├── main.py                     # pygame loop: spawns obstacles, scoring, collision, game-over/restart
│   ├── car.py                      # car entity + position update from steering/action input
│   ├── obstacles.py                # spawning + scrolling logic
│   └── control_bridge.py           # threaded CV pipeline → shared (steering, action) state; also a keyboard stub for early testing (Instruction #3)
├── evaluate.py                     # regression error + gesture accuracy/F1/confusion matrix, saved as numbers and plots
└── assets/                         # not needed — primitive pygame shapes are enough (Instruction #4)
```

---

## 6. Feature engineering spec

**Steering feature (from a single tracked hand):** compute the 2D vector from the wrist landmark to the middle-finger MCP (knuckle) landmark, and take its angle relative to vertical in the image plane. This "palm tilt angle" is the raw steering feature — stable, cheap to compute, and intuitive to control (tilt your hand like a steering wheel).

**Gesture feature (same tracked hand):** for each of the five fingers, compute a curl ratio — roughly, the distance from fingertip to wrist divided by the distance from that finger's MCP joint to wrist. An extended finger has a ratio near 1 (fingertip is far from the wrist); a curled finger has a ratio well below 1. This gives a 5-dimensional "hand openness" vector that cleanly separates open palm, fist, and thumbs-up shapes.

**Smoothing (applied at inference time, not training time):**
- Steering: exponential moving average over the last few frames' raw tilt-angle-derived steering values.
- Gesture: majority vote over the last ~5 frames' classifier predictions, so a single misclassified frame doesn't cause a control glitch.

---

## 7. Model spec

**Steering — regression.** Fit `LinearRegression` and a small `MLPRegressor` on the calibration data (tilt-angle feature → target steering value in [-1, 1]), evaluate both on held-out calibration points, and use whichever generalizes better. Don't assume the MLP wins by default — with this little data and a genuinely near-linear underlying relationship, plain linear regression sometimes performs just as well or better, and that's a legitimate, reportable result.

**Accelerate/brake/neutral — classification.** `RandomForestClassifier` or `SVC` on the 5-dimensional finger-curl feature vector. Include an explicit **neutral** class trained on your actual relaxed resting hand shape (don't assume "no gesture" is a hardcoded default — a naturally slightly-curled resting hand can otherwise get misclassified as a weak fist/brake).

**Validation:**
- Steering: hold out a few calibration points (not used in fitting) and report regression error (e.g. mean absolute error in steering units) on them.
- Gestures: stratified train/test split or k-fold cross-validation over recorded samples; report accuracy, F1, and a confusion matrix (which gestures get confused with which — likely candidates: fist vs. neutral, thumbs-up vs. accelerate).

---

## 8. Calibration & data collection protocol

**Steering calibration** (`calibrate_steering.py`, guided/scripted): prompt the user to hold their hand tilted fully left for ~2 seconds (records samples labeled steering = -1), then centered for ~2 seconds (label 0), then fully right for ~2 seconds (label +1). **Repeat the full left-center-right sweep at least twice** so there's real data to hold out for validation, not just enough to fit on.

**Gesture calibration** (`calibrate_gestures.py`, guided/scripted): record roughly 20-30 samples each of accelerate (open palm), brake (fist), and neutral (relaxed resting hand); optionally boost (thumbs-up) if you want the extra mechanic. Vary hand position/distance slightly within each set so the classifier doesn't overfit to one exact hand placement.

Both scripts should give clear on-screen countdowns/prompts ("Hold LEFT... 2... 1...") so the human collecting data knows exactly what's expected at each moment — this is the one part of the pipeline that's entirely on the user, not Claude Code (Instruction #2).

---

## 9. Game design spec

**Genre:** top-down endless driving/dodging game. The player's car has a fixed forward position on screen; the road scrolls downward (or the car scrolls "forward" visually) with obstacle cars/objects spawning ahead and moving toward the player at increasing speed over time.

**Controls, mapped from the trained models:**
- Steering value (smoothed, in [-1, 1]) → car's lateral (x-axis) velocity, clamped to road bounds.
- `accelerate` → forward speed increases toward a max.
- `brake` → forward speed decreases sharply toward a minimum (not necessarily zero — full stop isn't very fun).
- `neutral` → forward speed drifts back toward a default cruising speed.
- `boost` (optional) → temporary speed spike with a cooldown, for a bit of extra flair.

**Scoring:** survival time and/or distance traveled; game ends on collision with an obstacle, with a restart prompt.

**Debug overlay (keep this on during development and the demo):** show the live raw and smoothed steering value, the current classified gesture and its confidence, and whether a hand is currently detected — this is what makes the "it's really being controlled by trained models, not fixed rules" claim visibly true during a live demo, not just asserted.

---

## 10. Phased development plan

Use this as the literal task order. Each phase lists what "done" means — treat that as the gate before moving to the next phase.

### Phase 0 — Setup (~20 min)
- Initialize repo structure from Section 5, `requirements.txt`, basic `README.md`.
- **Done when:** a minimal script confirms MediaPipe detects a hand from the live webcam feed.

### Phase 1 — Hand tracking & feature extraction (~45-60 min)
- Implement `vision/hand_tracker.py` and `vision/features.py` (tilt angle + finger-curl vector).
- **Done when:** running a quick test script prints live, sensibly-changing tilt-angle and curl-vector values as you move your hand.

### Phase 2 — Calibration tools (~45-60 min)
- Implement `calibrate_steering.py` and `calibrate_gestures.py` with clear guided prompts, saving labeled samples to `calibration/data/`.
- **Done when:** running both scripts end-to-end produces correctly-shaped, correctly-labeled sample files on disk (can be tested with placeholder/throwaway data at this stage — real data collection is Phase 3).

### Phase 3 — Calibration data collection (~20-30 min, human-in-the-loop)
- **Stop and ask the user** to run both calibration scripts for real, per Section 8's protocol. Claude Code cannot perform this step.
- **Done when:** real steering-calibration and gesture samples exist on disk in the target quantities.

### Phase 4 — Model training & evaluation (~45-60 min)
- Implement `train_steering_model.py` (fit + compare regression models, save the better one) and `train_gesture_model.py` (train + cross-validate the classifier).
- Implement `evaluate.py`: regression error on held-out calibration points, classifier accuracy/F1/confusion matrix.
- **Done when:** both models train on real data and you have real, reportable evaluation numbers.

### Phase 5 — Core game build, keyboard-controlled first (~60-90 min)
- Build `game/main.py`, `car.py`, `obstacles.py` with **keyboard arrow-key control** as a temporary stand-in (Instruction #3) — get spawning, scrolling, collision, and scoring solid before touching the CV pipeline at all.
- **Done when:** the game is playable and fun with a keyboard, obstacles spawn and scroll correctly, collisions end the game, score displays.

### Phase 6 — Control integration (~45-60 min)
- Implement `control_bridge.py`: a background thread running the hand-tracking + trained-model pipeline, updating shared smoothed steering/gesture state; swap the game's input source from keyboard to this bridge.
- **Done when:** the car responds to real hand tilt and gestures in real time, with smoothing making it feel controllable rather than jittery.

### Phase 7 — Debug overlay & polish (~30-45 min)
- Add the live steering/gesture/confidence/hand-detected debug overlay (Section 9); tune obstacle speed/spawn rate for a good difficulty curve; add a restart flow.
- **Done when:** you can play a full round hands-free, watch the debug overlay confirm the models are doing the driving, and restart after a collision.

### Phase 8 — Stretch goals, if time remains
- See Section 11.

---

## 11. Stretch goals (only after Section 10's core phases are solid)

- **Two-hand virtual wheel:** replace single-hand tilt with the angle between both hands, for a more physically immersive "gripping a wheel" feel.
- **Boost gesture with cooldown** and a simple visual/particle effect when triggered.
- **Difficulty scaling:** obstacle speed and spawn density increase with score/time.
- **Steering confidence visualization:** show the regression model's uncertainty, not just its point estimate.
- **Best-run replay:** log a session's control inputs and play them back.

---

## 12. Demo script (for presenting the finished prototype)

1. Briefly show the calibration screen and the real evaluation numbers — "this steering model is trained on my own calibration sweep, not a hardcoded formula, and here's its held-out error."
2. Play a round hands-free: tilt to steer around obstacles, open palm to speed up, fist to brake — with the debug overlay visibly confirming live tilt angle → steering value and gesture → action.
3. Deliberately move your hand out of frame briefly to show the graceful "hand not detected" handling rather than a crash.
4. Close on the framing: two different ML problems — a trained regression for continuous control, a trained classifier for discrete actions — working together in real time, calibrated specifically to you.

---

## 13. Known pitfalls & mitigations

- **Jittery raw control** → always apply the smoothing filters from Section 6/Instruction #6; never feed raw per-frame model output straight into the game.
- **Neutral hand shape misclassified as brake** → collect explicit neutral-class calibration samples from your actual resting hand, don't assume a hardcoded "no gesture detected" default (Section 7).
- **CV inference blocking the game's frame rate** → run hand-tracking + model inference in a background thread (Instruction #5), never in the main render loop.
- **Too few calibration sweeps** → fit looks fine but generalizes poorly; always do at least two full left-center-right sweeps and hold some points out for a real validation number.
- **Lighting/framing drift between calibration and play** → keep the setup consistent, per Section 1's key assumption; this is a MediaPipe detection-quality issue, not something the trained models can fix.
- **Debugging game bugs and CV bugs at the same time** → build and test the game with keyboard input first (Instruction #3); only swap to the real control pipeline once the game itself is solid.
