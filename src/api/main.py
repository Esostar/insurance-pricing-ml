"""FastAPI inference service for insurance pricing."""
from fastapi import FastAPI, HTTPException
from .schemas import CustomerIn, PredictionOut, HealthOut
from src.pipelines.predict import predict, explain

app = FastAPI(
    title="Insurance Pricing Intelligence API",
    version="0.1.0",
    description="Predict claim cost and explain drivers for a single customer.",
)


@app.get("/health", response_model=HealthOut)
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionOut)
def predict_endpoint(customer: CustomerIn):
    try:
        row = customer.model_dump()
        pred = predict(row)
        exp = explain(row)
        return {
            "prediction": pred,
            "base_value_log": exp["base_value_log"],
            "top_contributions_log_space": exp["top_contributions_log_space"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
