import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
from fastapi.testclient import TestClient

import pipeline


def test_clean_handles_missing_and_duplicates():
    df = pipeline.make_synthetic(200)
    df.loc[0, "monthly_charges"] = None
    df.loc[1, "contract"] = None
    df = pd.concat([df, df.iloc[[5]]])
    out = pipeline.clean(df)
    assert out[pipeline.NUM].isna().sum().sum() == 0
    assert len(out) == 200


def test_train_and_serve(tmp_path, monkeypatch):
    model_path = str(tmp_path / "m.joblib")
    monkeypatch.chdir(tmp_path)
    metrics = pipeline.train(pipeline.make_synthetic(2000), model_path)
    assert metrics["roc_auc"] > 0.65

    monkeypatch.setenv("MODEL_PATH", model_path)
    import importlib
    import api

    importlib.reload(api)
    c = TestClient(api.app)
    body = dict(tenure_months=2, monthly_charges=110, support_calls=5, late_payments=3,
                contract="month-to-month", internet_service="fiber", payment_method="bank")
    r = c.post("/predict", json=body)
    assert r.status_code == 200 and 0 <= r.json()["churn_probability"] <= 1
    assert c.post("/predict", json={**body, "tenure_months": -1}).status_code == 422
