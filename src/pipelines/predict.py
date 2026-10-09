"""Load saved model, expose predict() and explain() for a single customer."""
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import shap

MODEL_PATH = Path("models/best_model.joblib")
_cache = {}


def _load():
    if "bundle" not in _cache:
        _cache["bundle"] = joblib.load(MODEL_PATH)
    return _cache["bundle"]


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


def predict(row: dict) -> float:
    bundle = _load()
    X = prepare_input(row)
    return float(np.expm1(bundle["pipeline"].predict(X)[0]))


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
    sample = {
        "age": 35, "sex": "male", "bmi": 32.5, "children": 2,
        "smoker": "yes", "region": "southeast",
    }
    print("prediction:", predict(sample))
    print(json.dumps(explain(sample), indent=2))
