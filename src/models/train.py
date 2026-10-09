"""Train baseline models, compare on original-scale metrics."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

PROCESSED = Path("data/processed/insurance_clean.csv")
METRICS = Path("reports/metrics/baselines.json")
RANDOM_STATE = 42

CAT_FEATS = ["sex", "region", "bmi_category", "age_bin"]
NUM_FEATS = ["age", "bmi", "children", "is_smoker", "smoker_bmi", "smoker_age"]
TARGET = "log_charges"


def load() -> pd.DataFrame:
    return pd.read_csv(PROCESSED)


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_FEATS),
        ("num", StandardScaler(), NUM_FEATS),
    ])


def inverse_log(y):
    return np.expm1(y)


def evaluate(name, model, X_tr, y_tr, X_te, y_te):
    pipe = Pipeline([("prep", make_preprocessor()), ("model", model)])
    pipe.fit(X_tr, y_tr)
    pred_log = pipe.predict(X_te)
    pred = inverse_log(pred_log)
    true = inverse_log(y_te.values)
    return {
        "mae": float(mean_absolute_error(true, pred)),
        "rmse": float(np.sqrt(mean_squared_error(true, pred))),
        "r2": float(r2_score(true, pred)),
        "cv_r2_log": float(np.mean(cross_val_score(
            pipe, X_tr, y_tr, cv=KFold(5, shuffle=True, random_state=RANDOM_STATE),
            scoring="r2",
        ))),
    }


def main():
    df = load()
    X = df[CAT_FEATS + NUM_FEATS]
    y = df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    models = {
        "linear": LinearRegression(),
        "ridge": Ridge(alpha=1.0),
        "rf": RandomForestRegressor(n_estimators=400, random_state=RANDOM_STATE, n_jobs=-1),
        "gbr": GradientBoostingRegressor(random_state=RANDOM_STATE),
        "xgb": XGBRegressor(
            n_estimators=300, max_depth=3, learning_rate=0.05,
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
    }

    results = {name: evaluate(name, m, X_tr, y_tr, X_te, y_te)
               for name, m in models.items()}

    METRICS.parent.mkdir(parents=True, exist_ok=True)
    METRICS.write_text(json.dumps(results, indent=2))

    for name, m in sorted(results.items(), key=lambda x: x[1]["mae"]):
        print(f"{name:8s} MAE={m['mae']:9.2f}  RMSE={m['rmse']:9.2f}  "
              f"R2={m['r2']:.4f}  CV_R2(log)={m['cv_r2_log']:.4f}")


if __name__ == "__main__":
    main()
