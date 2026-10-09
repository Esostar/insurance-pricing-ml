from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

PAYLOAD = {
    "age": 35, "sex": "male", "bmi": 32.5, "children": 2,
    "smoker": "yes", "region": "southeast",
}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_predict_endpoint():
    r = client.post("/predict", json=PAYLOAD)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["prediction"] > 0
    assert len(body["top_contributions"]) > 0
    assert body["target_space"] == "raw_target"


def test_predict_interval_endpoint():
    r = client.post("/predict_interval", json=PAYLOAD)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["lo"] < body["point"] < body["hi"]


def test_predict_rejects_bad_payload():
    bad = {**PAYLOAD, "age": 5}
    r = client.post("/predict", json=bad)
    assert r.status_code == 422
