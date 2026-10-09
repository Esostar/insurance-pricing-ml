"""Overfit / underfit diagnostics for the winning model."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROCESSED = Path("data/processed/insurance_clean.csv")
MODEL_PATH = Path("models/best_model.joblib")
OUT = Path("reports/metrics/diagnostics.json")
RANDOM_STATE = 42

CAT_FEATS = ["sex", "region", "bmi_category", "age_bin"]
NUM_FEATS = ["age", "bmi", "children", "is_smoker", "smoker_bmi", "smoker_age"]
TARGET = "log_charges"


def metrics(y_true_log, y_pred_log):
    true = np.expm1(y_true_log)
    pred = np.expm1(y_pred_log)
    return {
        "mae": float(mean_absolute_error(true, pred)),
        "rmse": float(np.sqrt(mean_squared_error(true, pred))),
        "r2": float(r2_score(true, pred)),
    }


def main():
    df = pd.read_csv(PROCESSED)
    X = df[CAT_FEATS + NUM_FEATS]
    y = df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    bundle = joblib.load(MODEL_PATH)
    pipe = bundle["pipeline"]

    train_m = metrics(y_tr.values, pipe.predict(X_tr))
    test_m = metrics(y_te.values, pipe.predict(X_te))

    gap_r2 = train_m["r2"] - test_m["r2"]
    gap_rmse = test_m["rmse"] - train_m["rmse"]

    # Learning curve: R^2 on train vs CV across training set sizes
    sizes, train_scores, val_scores = learning_curve(
        pipe, X, y, cv=5, scoring="r2",
        train_sizes=np.linspace(0.1, 1.0, 6),
        random_state=RANDOM_STATE, n_jobs=-1,
    )

    diagnosis = {
        "winner": bundle["name"],
        "train": train_m,
        "test": test_m,
        "gap_r2": gap_r2,
        "gap_rmse": gap_rmse,
        "verdict": (
            "overfitting" if gap_r2 > 0.10
            else "underfitting" if test_m["r2"] < 0.70
            else "balanced"
        ),
        "learning_curve": {
            "sizes": sizes.tolist(),
            "train_r2_mean": train_scores.mean(axis=1).tolist(),
            "val_r2_mean": val_scores.mean(axis=1).tolist(),
            "val_r2_std": val_scores.std(axis=1).tolist(),
        },
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(diagnosis, indent=2))

    print(f"Winner: {diagnosis['winner']}")
    print(f"Train  R2={train_m['r2']:.4f}  RMSE={train_m['rmse']:,.0f}")
    print(f"Test   R2={test_m['r2']:.4f}  RMSE={test_m['rmse']:,.0f}")
    print(f"Gap    R2={gap_r2:+.4f}   RMSE={gap_rmse:+,.0f}")
    print(f"Verdict: {diagnosis['verdict']}")
    print("\nLearning curve (val R2 by training size):")
    for s, t, v in zip(sizes, train_scores.mean(axis=1), val_scores.mean(axis=1)):
        print(f"  n={int(s):4d}  train={t:.3f}  val={v:.3f}")


if __name__ == "__main__":
    main()
