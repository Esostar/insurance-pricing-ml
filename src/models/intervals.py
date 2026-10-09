"""RF per-tree prediction intervals with split-conformal calibration."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from src.pipelines.predict import prepare_input

MODEL_PATH = Path("models/best_model.joblib")
PROCESSED = Path("data/processed/insurance_clean.csv")
OUT = Path("reports/metrics/intervals.json")
RANDOM_STATE = 42

CAT_FEATS = ["sex", "region", "bmi_category", "age_bin"]
NUM_FEATS = ["age", "bmi", "children", "is_smoker", "smoker_bmi", "smoker_age"]
TARGET = "log_charges"


def _forest_from_pipeline(pipe):
    model = pipe.named_steps["model"]
    if not hasattr(model, "estimators_"):
        raise TypeError("Model is not a tree ensemble with estimators_.")
    return model


def _per_tree_preds(pipe, X):
    Xt = pipe.named_steps["prep"].transform(X)
    forest = _forest_from_pipeline(pipe)
    return np.array([t.predict(Xt) for t in forest.estimators_]), Xt


def _splits():
    """Split into calibration and test sets (disjoint)."""
    df = pd.read_csv(PROCESSED)
    X = df[CAT_FEATS + NUM_FEATS]
    y = df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    X_cal, X_te2, y_cal, y_te2 = train_test_split(
        X_te, y_te, test_size=0.5, random_state=RANDOM_STATE
    )
    return (X_cal, y_cal), (X_te2, y_te2)


def _calibration_scale(target=0.90):
    (X_cal, y_cal), _ = _splits()
    bundle = joblib.load(MODEL_PATH)
    pipe = bundle["pipeline"]

    per_tree, _ = _per_tree_preds(pipe, X_cal)
    center = per_tree.mean(axis=0)
    half = (np.quantile(per_tree, 0.95, axis=0)
            - np.quantile(per_tree, 0.05, axis=0)) / 2.0

    for s in np.linspace(0.5, 3.0, 51):
        lo = center - s * half
        hi = center + s * half
        cov = ((y_cal.values >= lo) & (y_cal.values <= hi)).mean()
        if cov >= target:
            return float(s), float(cov)
    return 3.0, float(cov)


_CAL = {}


def _scale():
    if "s" not in _CAL:
        s, cov = _calibration_scale()
        _CAL["s"] = s
        _CAL["cov"] = cov
    return _CAL["s"], _CAL["cov"]


def predict_interval(row: dict, lo: float = 0.05, hi: float = 0.95) -> dict:
    bundle = joblib.load(MODEL_PATH)
    pipe = bundle["pipeline"]
    X = prepare_input(row)
    per_tree, _ = _per_tree_preds(pipe, X)

    center = float(per_tree.mean())
    half = float((np.quantile(per_tree, hi) - np.quantile(per_tree, lo)) / 2.0)
    s, _ = _scale()

    return {
        "point": float(np.expm1(center)),
        "lo": float(np.expm1(center - s * half)),
        "hi": float(np.expm1(center + s * half)),
        "level": hi - lo,
        "calibration_scale": s,
    }


def coverage_report():
    _, (X_te, y_te) = _splits()
    bundle = joblib.load(MODEL_PATH)
    pipe = bundle["pipeline"]
    per_tree, _ = _per_tree_preds(pipe, X_te)
    center = per_tree.mean(axis=0)
    half = (np.quantile(per_tree, 0.95, axis=0)
            - np.quantile(per_tree, 0.05, axis=0)) / 2.0

    s, cal_cov = _scale()
    lo = center - s * half
    hi = center + s * half
    cov = float(((y_te.values >= lo) & (y_te.values <= hi)).mean())

    return {
        "target_coverage": 0.90,
        "calibration_coverage": cal_cov,
        "test_coverage": cov,
        "calibration_scale": s,
        "mean_width_usd": float(np.mean(np.expm1(hi) - np.expm1(lo))),
        "median_width_usd": float(np.median(np.expm1(hi) - np.expm1(lo))),
    }


if __name__ == "__main__":
    sample = {"age": 35, "sex": "male", "bmi": 32.5, "children": 2,
              "smoker": "yes", "region": "southeast"}
    print("Sample 90% interval:")
    print(json.dumps(predict_interval(sample), indent=2))

    cov = coverage_report()
    print("\nCoverage report (calibration vs test on disjoint splits):")
    print(json.dumps(cov, indent=2))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(cov, indent=2))
