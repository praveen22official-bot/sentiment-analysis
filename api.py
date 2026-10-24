"""REST API serving the trained churn model.  uvicorn api:app --port 8001"""
import os

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from pipeline import CAT, NUM

MODEL_PATH = os.getenv("MODEL_PATH", "model.joblib")
app = FastAPI(title="Churn prediction API")
_model = None


def get_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise HTTPException(503, "Model not trained yet. Run: python pipeline.py")
        _model = joblib.load(MODEL_PATH)
    return _model


class Customer(BaseModel):
    tenure_months: int = Field(ge=0)
    monthly_charges: float = Field(ge=0)
    support_calls: int = Field(ge=0)
    late_payments: int = Field(ge=0)
    contract: str
    internet_service: str
    payment_method: str


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": os.path.exists(MODEL_PATH)}


@app.post("/predict")
def predict(c: Customer):
    row = pd.DataFrame([c.model_dump()])[NUM + CAT]
    p = float(get_model().predict_proba(row)[0, 1])
    return {"churn_probability": round(p, 4), "will_churn": p >= 0.5}
