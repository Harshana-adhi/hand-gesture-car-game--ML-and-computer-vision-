"""
Evaluation report for both trained models, using real recorded data only.

Steering: reproduces the same held-out split used at training time (fixed
random_state), reports MAE, and saves a calibration-fit plot (tilt_angle vs.
target, train vs. held-out points, plus the fitted model curve).

Gestures: reads the cross-validation results captured during training
(models/train_gesture_model.py already ran stratified k-fold CV) and saves
a confusion matrix plot.

Run this after both train_*.py scripts have produced models/saved/*.joblib:
    python evaluate.py
"""
import json
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

_ROOT = os.path.dirname(os.path.abspath(__file__))
STEERING_DATA_PATH = os.path.join(_ROOT, "calibration", "data", "steering_calibration.json")
STEERING_MODEL_PATH = os.path.join(_ROOT, "models", "saved", "steering_model.joblib")
GESTURE_MODEL_PATH = os.path.join(_ROOT, "models", "saved", "gesture_model.joblib")
OUT_DIR = os.path.join(_ROOT, "evaluation_results")

RANDOM_STATE = 42
TEST_SIZE = 0.25


def evaluate_steering():
    if not os.path.exists(STEERING_MODEL_PATH):
        print("No steering model found -- run `python -m models.train_steering_model` first.")
        return None

    with open(STEERING_DATA_PATH) as f:
        samples = json.load(f)
    X = np.array([[s["tilt_angle"]] for s in samples])
    y = np.array([s["target"] for s in samples])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    bundle = joblib.load(STEERING_MODEL_PATH)
    model, model_name = bundle["model"], bundle["model_name"]

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"[Steering] model={model_name}  held-out MAE={mae:.4f}  "
          f"(n_train={len(X_train)}, n_test={len(X_test)})")

    # Calibration-fit plot.
    x_line = np.linspace(X.min() - 5, X.max() + 5, 200).reshape(-1, 1)
    y_line = model.predict(x_line)

    plt.figure(figsize=(7, 5))
    plt.scatter(X_train, y_train, label="train", alpha=0.5, s=20)
    plt.scatter(X_test, y_test, label="held-out test", alpha=0.8, s=40, marker="x")
    plt.plot(x_line, y_line, color="black", linewidth=1.5, label=f"{model_name} fit")
    plt.xlabel("tilt_angle (degrees)")
    plt.ylabel("steering target")
    plt.title(f"Steering calibration fit ({model_name}, held-out MAE={mae:.4f})")
    plt.legend()
    plt.tight_layout()

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "steering_calibration_fit.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  saved plot -> {out_path}")

    return {"model_name": model_name, "held_out_mae": float(mae),
            "n_train": len(X_train), "n_test": len(X_test)}


def evaluate_gestures():
    if not os.path.exists(GESTURE_MODEL_PATH):
        print("No gesture model found -- run `python -m models.train_gesture_model` first.")
        return None

    bundle = joblib.load(GESTURE_MODEL_PATH)
    classes = bundle["classes"]
    cm = np.array(bundle["cv_confusion_matrix"])
    acc_mean = bundle["cv_accuracy_mean"]
    acc_std = bundle["cv_accuracy_std"]
    macro_f1 = bundle["cv_macro_f1"]

    print(f"[Gestures] cross-validated accuracy={acc_mean:.4f} (+/- {acc_std:.4f})  "
          f"macro F1={macro_f1:.4f}")
    print(f"  confusion matrix (rows=true, cols=predicted), labels={classes}:")
    print(cm)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"Gesture confusion matrix (acc={acc_mean:.3f}, F1={macro_f1:.3f})")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax)
    plt.tight_layout()

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "gesture_confusion_matrix.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  saved plot -> {out_path}")

    return {"classes": classes, "cv_accuracy_mean": acc_mean,
            "cv_accuracy_std": acc_std, "cv_macro_f1": macro_f1,
            "confusion_matrix": cm.tolist()}


def main():
    print("=== Steering regression evaluation ===")
    steering_metrics = evaluate_steering()
    print("\n=== Gesture classification evaluation ===")
    gesture_metrics = evaluate_gestures()

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump({"steering": steering_metrics, "gestures": gesture_metrics}, f, indent=2)
    print(f"\nSaved metrics.json -> {os.path.join(OUT_DIR, 'metrics.json')}")


if __name__ == "__main__":
    main()
