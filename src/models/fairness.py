"""Fairness audit: distribution of residuals across demographic groups."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

MODEL_PATH = Path("models/best_model.joblib")
PROCESSED = Path("data/processed/insurance_clean.csv")
OUT = Path("reports/metrics/fairness.json")
RANDOM_STATE = 42

CAT_FEATS = ["sex", "region", "bmi_category", "age_bin"]
NUM_FEATS = ["age", "bmi", "children", "is_smoker", "smoker_bmi", "smoker_age"]
TARGET = "log_charges"

GROUPS = ["sex", "region", "smoker", "bmi_category", "age_bin"]


def _predict_usd(pipe, X):
    return np.expm1(pipe.predict(X))


def audit():
    df = pd.read_csv(PROCESSED)
    X = df[CAT_FEATS + NUM_FEATS]
    y = df[TARGET]
    _, X_te, _, y_te = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    bundle = joblib.load(MODEL_PATH)
    pipe = bundle["pipeline"]

    pred_usd = _predict_usd(pipe, X_te)
    true_usd = np.expm1(y_te.values)
    resid = pred_usd - true_usd  # positive = overcharge

    # Reattach group columns from the original dataframe using the index
    df_te = df.loc[X_te.index, GROUPS].reset_index(drop=True)

    report = {"overall": {
        "mae": float(np.mean(np.abs(resid))),
        "mean_residual": float(np.mean(resid)),
        "n": int(len(resid)),
    }}

    for g in GROUPS:
        rows = {}
        for level, idx in df_te.groupby(g).groups.items():
            r = resid[idx.values]
            rows[str(level)] = {
                "n": int(len(r)),
                "mean_residual": float(np.mean(r)),
                "mae": float(np.mean(np.abs(r))),
            }
        # disparity: max - min of |mean_residual| across groups
        means = [abs(v["mean_residual"]) for v in rows.values()]
        report[g] = {
            "groups": rows,
            "max_abs_mean_residual": float(max(means)),
            "disparity_usd": float(max(means) - min(means)),
        }

    return report


if __name__ == "__main__":
    rep = audit()
    print(json.dumps(rep, indent=2))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, indent=2))
