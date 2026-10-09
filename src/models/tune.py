"""Tune RF + GBR, pick winner by test RMSE on original scale, save artifact."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROCESSED = Path("data/processed/insurance_clean.csv")
MODEL_OUT = Path("models/best_model.joblib")
META_OUT = Path("reports/metrics/tuned.json")
RANDOM_STATE = 42

CAT_FEATS = ["sex", "region", "bmi_category", "age_bin"]
NUM_FEATS = ["age", "bmi", "children", "is_smoker", "smoker_bmi", "smoker_age"]
TARGET = "log_charges"


def make_preprocessor():
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_FEATS),
        ("num", StandardScaler(), NUM_FEATS),
    ])


def evaluate(pipe, X_te, y_te):
    pred = np.expm1(pipe.predict(X_te))
    true = np.expm1(y_te.values)
    return {
        "mae": float(mean_absolute_error(true, pred)),
        "rmse": float(np.sqrt(mean_squared_error(true, pred))),
        "r2": float(r2_score(true, pred)),
    }


def tune(name, estimator, param_dist, X_tr, y_tr):
    pipe = Pipeline([("prep", make_preprocessor()), ("model", estimator)])
    search = RandomizedSearchCV(
        pipe, param_dist, n_iter=30, cv=3,
        scoring="neg_mean_absolute_error",
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    search.fit(X_tr, y_tr)
    print(f"[{name}] best params: {search.best_params_}")
    return search.best_estimator_


def main():
    df = pd.read_csv(PROCESSED)
    X = df[CAT_FEATS + NUM_FEATS]
    y = df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    rf_params = {
        "model__n_estimators": [200, 400, 800],
        "model__max_depth": [None, 10, 20, 30],
        "model__min_samples_leaf": [1, 2, 4, 8],
        "model__min_samples_split": [2, 5, 10],
        "model__max_features": ["sqrt", "log2", 0.5],
    }
    gbr_params = {
        "model__n_estimators": [200, 400, 600],
        "model__learning_rate": [0.03, 0.05, 0.08, 0.1],
        "model__max_depth": [2, 3, 4],
        "model__min_samples_leaf": [1, 2, 4],
        "model__subsample": [0.7, 0.85, 1.0],
    }

    rf_best = tune("rf", RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
                   rf_params, X_tr, y_tr)
    gbr_best = tune("gbr", GradientBoostingRegressor(random_state=RANDOM_STATE),
                    gbr_params, X_tr, y_tr)

    results = {
        "rf": evaluate(rf_best, X_te, y_te),
        "gbr": evaluate(gbr_best, X_te, y_te),
    }
    for name, m in results.items():
        print(f"{name:4s} MAE={m['mae']:9.2f}  RMSE={m['rmse']:9.2f}  R2={m['r2']:.4f}")

    winner = min(results.items(), key=lambda x: x[1]["rmse"])
    print(f"\nWinner: {winner[0]}")

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"name": winner[0],
                 "pipeline": rf_best if winner[0] == "rf" else gbr_best}, MODEL_OUT)

    META_OUT.parent.mkdir(parents=True, exist_ok=True)
    META_OUT.write_text(json.dumps(
        {"winner": winner[0], "results": results,
         "features": {"categorical": CAT_FEATS, "numeric": NUM_FEATS,
                      "target": TARGET}},
        indent=2,
    ))
    print(f"Saved -> {MODEL_OUT}")


if __name__ == "__main__":
    main()
