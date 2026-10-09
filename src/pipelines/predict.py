"""Load saved model, expose predict(), predict_interval(), explain()."""
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import shap
from sklearn.model_selection import train_test_split

MODEL_PATH = Path("models/best_model.joblib")
PROCESSED = Path("data/processed/insurance_clean.csv")
RANDOM_STATE = 42

CAT_FEATS = ["sex", "region", "bmi_category", "age_bin"]
NUM_FEATS = ["age", "bmi", "children", "is_smoker", "smoker_bmi", "smoker_age"]
TARGET = "log_charges"

_cache = {}


def _load():
    if "bundle" in _cache:
        return _cache["bundle"]
    bundle = joblib.load(MODEL_PATH)
    _cache["bundle"] = bundle
    _cache["smear"] = _smearing_factor(bundle)
    _cache["scale"] = _calibration_scale(bundle)
    return bundle


def prepare_input(row: dict) -> pd.DataFrame:
    df = pd.DataFrame([row])
    df["is_smoker"] = (df["smoker"] == "yes").astype(int)
    df["smoker_bmi"] = df["is_smoker"] * df["bmi"]
    df["smoker_age"] = df["is_smoker"] * df["age"]

    def bmi_cat(b):
        if b < 18.5: return "underweight"
        if b < 25: return "normal"
        if b < 30: return "overweight"
        return "obese"
    df["bmi_category"] = df["bmi"].apply(bmi_cat)
    df["age_bin"] = pd.cut(
        df["age"], bins=[17, 25, 35, 45, 55, 65],
        labels=["18-25", "26-35", "36-45", "46-55", "56-64"],
    ).astype(str)
    return df


def _train_frame():
    df = pd.read_csv(PROCESSED)
    X = df[CAT_FEATS + NUM_FEATS]
    y = df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    return X_tr, y_tr, X_te, y_te


def _smearing_factor(bundle):
    pipe = bundle["pipeline"]
    X_tr, y_tr, _, _ = _train_frame()
    resid_log = y_tr.values - pipe.predict(X_tr)
    return float(np.mean(np.exp(resid_log)))


def _calibration_scale(bundle, target=0.90):
    pipe = bundle["pipeline"]
    _, _, X_te, y_te = _train_frame()
    X_cal, _, y_cal, _ = train_test_split(
        X_te, y_te, test_size=0.5, random_state=RANDOM_STATE
    )
    Xt = pipe.named_steps["prep"].transform(X_cal)
    forest = pipe.named_steps["model"]
    per_tree = np.array([t.predict(Xt) for t in forest.estimators_])
    center = per_tree.mean(axis=0)
    half = (np.quantile(per_tree, 0.95, axis=0)
            - np.quantile(per_tree, 0.05, axis=0)) / 2.0
    for s in np.linspace(0.5, 3.0, 51):
        lo = center - s * half
        hi = center + s * half
        cov = ((y_cal.values >= lo) & (y_cal.values <= hi)).mean()
        if cov >= target:
            return float(s)
    return 3.0


def predict(row: dict) -> float:
    bundle = _load()
    log_pred = bundle["pipeline"].predict(prepare_input(row))[0]
    return float(np.expm1(log_pred) * _cache["smear"])


def predict_interval(row: dict, lo: float = 0.05, hi: float = 0.95) -> dict:
    bundle = _load()
    pipe = bundle["pipeline"]
    X = prepare_input(row)
    Xt = pipe.named_steps["prep"].transform(X)
    forest = pipe.named_steps["model"]
    per_tree = np.array([t.predict(Xt)[0] for t in forest.estimators_])

    center = float(per_tree.mean())
    half = float((np.quantile(per_tree, hi) - np.quantile(per_tree, lo)) / 2.0)
    s = _cache["scale"]
    smear = _cache["smear"]

    return {
        "point": float(np.expm1(center) * smear),
        "lo": float(np.expm1(center - s * half) * smear),
        "hi": float(np.expm1(center + s * half) * smear),
        "level": hi - lo,
        "smearing_factor": smear,
        "calibration_scale": s,
    }


def explain(row: dict, top_k: int = 6) -> dict:
    bundle = _load()
    pipe = bundle["pipeline"]
    X = prepare_input(row)
    Xt = pipe.named_steps["prep"].transform(X)
    model = pipe.named_steps["model"]
    explainer = shap.TreeExplainer(model)
    sv = np.asarray(explainer.shap_values(Xt))[0]
    base = float(np.asarray(explainer.expected_value).ravel()[0])
    raw_names = list(pipe.named_steps["prep"].get_feature_names_out())
    clean_names = [n.split("__", 1)[-1] for n in raw_names]
    pairs = sorted(zip(clean_names, sv), key=lambda x: abs(x[1]), reverse=True)[:top_k]
    return {
        "prediction": predict(row),
        "base_value_log": base,
        "top_contributions_log_space": [
            {"feature": f, "shap_value": float(v)} for f, v in pairs
        ],
    }


if __name__ == "__main__":
    import json
    s = {"age": 35, "sex": "male", "bmi": 32.5, "children": 2,
         "smoker": "yes", "region": "southeast"}
    print("predict:", predict(s))
    print("interval:", json.dumps(predict_interval(s), indent=2))
