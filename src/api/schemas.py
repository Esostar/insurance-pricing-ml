from pydantic import BaseModel, Field
from typing import Literal


class CustomerIn(BaseModel):
    age: int = Field(..., ge=18, le=100)
    sex: Literal["male", "female"]
    bmi: float = Field(..., ge=10, le=70)
    children: int = Field(..., ge=0, le=10)
    smoker: Literal["yes", "no"]
    region: Literal["northeast", "northwest", "southeast", "southwest"]


class PredictionOut(BaseModel):
    prediction: float
    base_value_log: float
    top_contributions_log_space: list[dict]


class HealthOut(BaseModel):
    status: str
