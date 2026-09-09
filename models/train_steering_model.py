"""
Fits LinearRegression and a small MLPRegressor on real steering-calibration
data (tilt_angle -> target in [-1, 1]), evaluates both on a held-out split,
and saves whichever generalizes better to models/saved/steering_model.joblib.

Run:
    python -m models.train_steering_model
"""
import json
import os

import joblib
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(_ROOT, "calibration", "data", "steering_calibration.json")
SAVE_PATH = os.path.join(_ROOT, "models", "saved", "steering_model.joblib")

RANDOM_STATE = 42
TEST_SIZE = 0.25


def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"No steering calibration data at {DATA_PATH}. "
            "Run `python -m calibration.calibrate_steering` first (Phase 3)."
        )
    with open(DATA_PATH) as f:
        samples = json.load(f)
    if len(samples) < 10:
        raise ValueError(
            f"Only {len(samples)} steering samples found -- too few to fit/evaluate. "
            "Re-run calibrate_steering.py."
        )
    X = np.array([[s["tilt_angle"]] for s in samples])
    y = np.array([s["target"] for s in samples])
    return X, y


def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Loaded {len(X)} samples -> train={len(X_train)}, held-out test={len(X_test)}")

    linear = LinearRegression()
    linear.fit(X_train, y_train)
    linear_pred = linear.predict(X_test)
    linear_mae = mean_absolute_error(y_test, linear_pred)

    mlp = MLPRegressor(
        hidden_layer_sizes=(8,),
        activation="tanh",
        max_iter=5000,
        random_state=RANDOM_STATE,
    )
    mlp.fit(X_train, y_train.ravel())
    mlp_pred = mlp.predict(X_test)
    mlp_mae = mean_absolute_error(y_test, mlp_pred)

    print(f"\nLinearRegression held-out MAE: {linear_mae:.4f}")
    print(f"MLPRegressor    held-out MAE: {mlp_mae:.4f}")

    if linear_mae <= mlp_mae:
        best_name, best_model, best_mae = "LinearRegression", linear, linear_mae
    else:
        best_name, best_model, best_mae = "MLPRegressor", mlp, mlp_mae

    print(f"\nSelected: {best_name} (held-out MAE={best_mae:.4f})")

    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    joblib.dump({"model": best_model, "model_name": best_name}, SAVE_PATH)
    print(f"Saved best model to {SAVE_PATH}")


if __name__ == "__main__":
    main()
