import math
from src.pipelines.predict import predict, explain, prepare_input, predict_interval

SAMPLE = {
    "age": 35, "sex": "male", "bmi": 32.5, "children": 2,
    "smoker": "yes", "region": "southeast",
}


def test_prepare_input_adds_features():
    df = prepare_input(SAMPLE)
    assert df["is_smoker"].iloc[0] == 1
    assert df["bmi_category"].iloc[0] == "obese"
    assert math.isclose(df["smoker_bmi"].iloc[0], 32.5)


def test_predict_returns_positive_float():
    p = predict(SAMPLE)
    assert isinstance(p, float)
    assert p > 0


def test_smoker_pays_more_than_nonsmoker():
    non = {**SAMPLE, "smoker": "no"}
    assert predict(SAMPLE) > predict(non)


def test_explain_returns_top_features():
    exp = explain(SAMPLE, top_k=3)
    assert len(exp["top_contributions"]) == 3
    assert exp["prediction"] > 0


def test_interval_brackets_point():
    iv = predict_interval(SAMPLE)
    assert iv["lo"] < iv["point"] < iv["hi"]
