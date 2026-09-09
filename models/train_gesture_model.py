"""
Trains a RandomForestClassifier on real recorded finger-curl-vector gesture
samples (accelerate/brake/neutral/boost), evaluates it with stratified
k-fold cross-validation (accuracy + F1 + confusion matrix), then refits on
all available data and saves to models/saved/gesture_model.joblib.

Run:
    python -m models.train_gesture_model
"""
import json
import os

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(_ROOT, "calibration", "data", "gestures")
SAVE_PATH = os.path.join(_ROOT, "models", "saved", "gesture_model.joblib")

RANDOM_STATE = 42
N_SPLITS = 5


def load_data():
    X, y = [], []
    classes_found = []
    for class_name in sorted(os.listdir(DATA_DIR)):
        class_dir = os.path.join(DATA_DIR, class_name)
        if not os.path.isdir(class_dir):
            continue
        sample_files = [f for f in os.listdir(class_dir) if f.endswith(".json")]
        if not sample_files:
            continue
        classes_found.append(class_name)
        for fname in sample_files:
            with open(os.path.join(class_dir, fname)) as f:
                sample = json.load(f)
            X.append(sample["curl_vector"])
            y.append(sample["class"])

    if len(classes_found) < 2:
        raise ValueError(
            f"Found gesture samples for only {classes_found} -- need at least 2 classes. "
            "Run `python -m calibration.calibrate_gestures` first (Phase 3)."
        )
    return np.array(X), np.array(y), classes_found


def main():
    X, y, classes_found = load_data()
    print(f"Loaded {len(X)} gesture samples across classes: {classes_found}")
    counts = {c: int((y == c).sum()) for c in classes_found}
    print(f"Per-class counts: {counts}")

    min_class_count = min(counts.values())
    n_splits = min(N_SPLITS, min_class_count)
    if n_splits < 2:
        raise ValueError("Smallest class has fewer than 2 samples -- can't cross-validate.")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    clf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)

    acc_scores = cross_val_score(clf, X, y, cv=skf, scoring="accuracy")
    y_pred_cv = cross_val_predict(clf, X, y, cv=skf)
    f1 = f1_score(y, y_pred_cv, average="macro")
    cm = confusion_matrix(y, y_pred_cv, labels=classes_found)

    print(f"\n{n_splits}-fold stratified CV accuracy: {acc_scores.mean():.4f} (+/- {acc_scores.std():.4f})")
    print(f"Macro F1: {f1:.4f}")
    print(f"\nConfusion matrix (rows=true, cols=predicted), labels={classes_found}:")
    print(cm)

    # Refit on all available data for the deployed model.
    final_clf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    final_clf.fit(X, y)

    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    joblib.dump({
        "model": final_clf,
        "classes": classes_found,
        "cv_accuracy_mean": float(acc_scores.mean()),
        "cv_accuracy_std": float(acc_scores.std()),
        "cv_macro_f1": float(f1),
        "cv_confusion_matrix": cm.tolist(),
    }, SAVE_PATH)
    print(f"\nSaved final model (refit on all data) to {SAVE_PATH}")


if __name__ == "__main__":
    main()
