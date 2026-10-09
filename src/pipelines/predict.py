"""Load saved model; expose predict(), predict_interval(), explain()."""
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import shap

MODEL_PATH = Path("models/best_model.joblib")
PROCESSED = Path("data/processed/insurance_clean.csv")
RANDOM_STATE = 42

CAT_FEATS = ["sex", "region", "bmi_category", "age_bin"]
NUM_FEATS = ["age", "bmi", "children", "is_smoker", "smoker_bmi", "smoker_age"]

_cache = {}


def _load():
    if "bundle" in _cache:
        return _cache["bundle"]
    bundle = joblib.load(MODEL_PATH)
    _cache["bundle"] = bundle
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


def _target_space():
    return _load().get("target", "raw_target")


def predict(row: dict) -> float:
    _load()
    pipe = _cache["bundle"]["pipeline"]
    raw = pipe.predict(prepare_input(row))[0]
    if _target_space() == "log_target":
        return float(np.expm1(raw))
    return float(raw)


def predict_interval(row: dict, lo: float = 0.05, hi: float = 0.95) -> dict:
    _load()
    pipe = _cache["bundle"]["pipeline"]
    X = prepare_input(row)
    Xt = pipe.named_steps["prep"].transform(X)
    forest = pipe.named_steps["model"]
    per_tree = np.array([t.predict(Xt)[0] for t in forest.estimators_])

    if _target_space() == "log_target":
        per_tree = np.expm1(per_tree)

    return {
        "point": float(np.mean(per_tree)),
        "lo": float(np.quantile(per_tree, lo)),
        "hi": float(np.quantile(per_tree, hi)),
        "level": hi - lo,
        "target_space": _target_space(),
    }


def explain(row: dict, top_k: int = 6) -> dict:
    _load()
    pipe = _cache["bundle"]["pipeline"]
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
        "base_value": base,
        "target_space": _target_space(),
        "top_contributions": [
            {"feature": f, "shap_value": float(v)} for f, v in pairs
        ],
    }
