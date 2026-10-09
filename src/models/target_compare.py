"""Compare training target: raw charges vs log1p(charges)."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROCESSED = Path("data/processed/insurance_clean.csv")
OUT = Path("reports/metrics/target_compare.json")
RANDOM_STATE = 42

CAT_FEATS = ["sex", "region", "bmi_category", "age_bin"]
NUM_FEATS = ["age", "bmi", "children", "is_smoker", "smoker_bmi", "smoker_age"]


def make_pipe():
    return Pipeline([
        ("prep", ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_FEATS),
            ("num", StandardScaler(), NUM_FEATS),
        ])),
        ("model", RandomForestRegressor(
            n_estimators=400, min_samples_leaf=8, max_features=0.5,
            max_depth=30, min_samples_split=2,
            random_state=RANDOM_STATE, n_jobs=-1,
        )),
    ])


def evaluate(y_true_usd, y_pred_usd):
    return {
        "mae": float(mean_absolute_error(y_true_usd, y_pred_usd)),
        "rmse": float(np.sqrt(mean_squared_error(y_true_usd, y_pred_usd))),
        "r2": float(r2_score(y_true_usd, y_pred_usd)),
        "bias": float(np.mean(y_pred_usd - y_true_usd)),
    }


def main():
    df = pd.read_csv(PROCESSED)
    X = df[CAT_FEATS + NUM_FEATS]
    y_raw = df["charges"]
    y_log = df["log_charges"]

    X_tr, X_te, y_raw_tr, y_raw_te = train_test_split(
        X, y_raw, test_size=0.2, random_state=RANDOM_STATE
    )
    _, _, y_log_tr, y_log_te = train_test_split(
        X, y_log, test_size=0.2, random_state=RANDOM_STATE
    )

    # Raw target
    p_raw = make_pipe()
    p_raw.fit(X_tr, y_raw_tr)
    pred_raw = p_raw.predict(X_te)

    # Log target, inverse transformed
    p_log = make_pipe()
    p_log.fit(X_tr, y_log_tr)
    pred_log = np.expm1(p_log.predict(X_te))

    results = {
        "raw_target": evaluate(y_raw_te.values, pred_raw),
        "log_target": evaluate(y_raw_te.values, pred_log),
    }
    for k, m in results.items():
        print(f"{k:12s} MAE={m['mae']:9.2f}  RMSE={m['rmse']:9.2f}  "
              f"R2={m['r2']:.4f}  bias={m['bias']:+.2f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2))

    # Save the winner
    winner = "raw_target" if results["raw_target"]["mae"] < results["log_target"]["mae"] else "log_target"
    print(f"\nWinner: {winner}")
    joblib.dump(
        {"name": "rf", "target": winner, "pipeline": p_raw if winner == "raw_target" else p_log},
        "models/best_model.joblib",
    )
    print("Saved -> models/best_model.joblib")


if __name__ == "__main__":
    main()
