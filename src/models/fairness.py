"""Fairness audit: residuals across demographic groups (raw-target model)."""
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
GROUPS = ["sex", "region", "smoker", "bmi_category", "age_bin"]


def audit():
    df = pd.read_csv(PROCESSED)
    X = df[CAT_FEATS + NUM_FEATS]
    y_usd = df["charges"]

    _, X_te, _, y_te = train_test_split(
        X, y_usd, test_size=0.2, random_state=RANDOM_STATE
    )
    bundle = joblib.load(MODEL_PATH)
    pipe = bundle["pipeline"]
    target_space = bundle.get("target", "raw_target")

    pred = pipe.predict(X_te)
    if target_space == "log_target":
        pred = np.expm1(pred)

    resid = pred - y_te.values
    df_te = df.loc[X_te.index, GROUPS].reset_index(drop=True)

    report = {
        "target_space": target_space,
        "overall": {
            "mae": float(np.mean(np.abs(resid))),
            "mean_residual": float(np.mean(resid)),
            "n": int(len(resid)),
        },
    }
    for g in GROUPS:
        rows = {}
        for level, idx in df_te.groupby(g).groups.items():
            r = resid[idx.values]
            rows[str(level)] = {
                "n": int(len(r)),
                "mean_residual": float(np.mean(r)),
                "mae": float(np.mean(np.abs(r))),
            }
        means = [abs(v["mean_residual"]) for v in rows.values()]
        report[g] = {
            "groups": rows,
            "disparity_usd": float(max(means) - min(means)),
        }
    return report


if __name__ == "__main__":
    rep = audit()
    print(f"Target space: {rep['target_space']}")
    print(f"Overall bias: {rep['overall']['mean_residual']:+.2f} | "
          f"MAE: {rep['overall']['mae']:.2f}")
    for g in GROUPS:
        print(f"{g:14s} disparity = ${rep[g]['disparity_usd']:.0f}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, indent=2))
