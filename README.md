# Insurance Pricing & Risk Intelligence

Production-style ML system that estimates annual claim cost for a customer
and explains the drivers behind each quote.

## Problem
Insurers price premiums from historical claim cost. We model `charges`
(annual medical cost) from customer features and expose it as a service
with per-customer explainability.

## Stack
Python 3.9, scikit-learn, XGBoost, SHAP, FastAPI, Streamlit, Docker, Pytest

## Layout
- data/            raw + processed
- notebooks/       EDA and experiments
- src/data/        audit / loading
- src/features/    feature engineering
- src/models/      train, tune, diagnose
- src/pipelines/   inference + explain
- src/api/         FastAPI service
- src/dashboard/   Streamlit UI
- models/          trained artifact (joblib)
- reports/metrics/ metrics + diagnostics
- tests/           pytest suite
- docker/          Dockerfiles + runtime reqs

## Setup
    python -m venv .venv
    source .venv/Scripts/activate
    pip install -r requirements.txt

## Pipeline
    python -m src.features.build
    python -m src.models.train
    python -m src.models.tune
    python -m src.models.diagnose

## Serve
    docker compose up -d
    API:       http://localhost:8000/docs
    Dashboard: http://localhost:8501

## Model
- Target: log1p(charges), inverted for reporting.
- Features: age, sex, bmi, children, smoker, region +
  bmi_category, age_bin, is_smoker, smoker_bmi, smoker_age.
- Winner: RandomForest, test R2 ~ 0.90, MAE ~ $1,940, RMSE ~ $4,230.
- Diagnostics: balanced (train/test R2 gap < 0.04).

## Limitations
- Small dataset (1,337 rows); no external validation.
- US population only.
- No fairness audit yet.
- Point estimates only; no prediction intervals.

## Tests
    pytest -q


## Endpoints
- POST /predict            -> point estimate + SHAP (USD)
- POST /predict_interval   -> 90% calibrated interval (split-conformal)
- GET  /health

## Metrics (test set, raw-target RF)
- R2  ~ 0.90
- MAE ~ $2,435
- RMSE ~ $4,280
- Bias ~ +$77
- Interval coverage ~ 90% (target 90%)

## Fairness audit
See `reports/metrics/fairness.json`. Largest group disparity is by
BMI category (~$1,567); other groups under $600.

## CI
GitHub Actions runs pytest on every push (fetch data, build model, run tests).


## Endpoints
- POST /predict            -> point estimate + SHAP (USD)
- POST /predict_interval   -> 90% calibrated interval (split-conformal)
- GET  /health

## Metrics (test set, raw-target RF)
- R2  ~ 0.90
- MAE ~ $2,435
- RMSE ~ $4,280
- Bias ~ +$77
- Interval coverage ~ 90% (target 90%)

## Fairness audit
See `reports/metrics/fairness.json`. Largest group disparity is by
BMI category (~$1,567); other groups under $600.

## CI
GitHub Actions runs pytest on every push (fetch data, build model, run tests).
