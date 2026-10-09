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
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] > 0
    assert len(body["top_contributions_log_space"]) > 0


def test_predict_rejects_bad_payload():
    bad = {**PAYLOAD, "age": 5}
    r = client.post("/predict", json=bad)
    assert r.status_code == 422
