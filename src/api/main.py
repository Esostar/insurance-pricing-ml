"""FastAPI inference service for insurance pricing."""
from fastapi import FastAPI, HTTPException
from .schemas import CustomerIn, PredictionOut, IntervalOut, HealthOut
from src.pipelines.predict import predict, explain, predict_interval

app = FastAPI(
    title="Insurance Pricing Intelligence API",
    version="0.2.0",
    description="Predict claim cost and explain drivers for a single customer.",
)


@app.get("/health", response_model=HealthOut)
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionOut)
def predict_endpoint(customer: CustomerIn):
    try:
        row = customer.model_dump()
        exp = explain(row)
        return {
            "prediction": exp["prediction"],
            "base_value": exp["base_value"],
            "target_space": exp["target_space"],
            "top_contributions": exp["top_contributions"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_interval", response_model=IntervalOut)
def predict_interval_endpoint(customer: CustomerIn):
    try:
        return predict_interval(customer.model_dump())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
