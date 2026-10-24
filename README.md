# Customer Churn Prediction Pipeline

Data cleaning → feature encoding → XGBoost → saved model → REST API, plus an Airflow DAG for weekly retraining.

```bash
pip install -r requirements.txt
python pipeline.py            # trains on synthetic data, writes model.joblib + metrics.json
uvicorn api:app --port 8001   # POST /predict
pytest                        # needs: pip install pytest httpx
```

Example request:
```bash
curl -X POST localhost:8001/predict -H 'content-type: application/json' -d '{
 "tenure_months":2,"monthly_charges":110,"support_calls":5,"late_payments":3,
 "contract":"month-to-month","internet_service":"fiber","payment_method":"bank"}'
```

The bundled dataset is **synthetic**, so the metrics in `metrics.json` describe the demo data only. Use `--csv` with real customer data and report those results instead. AWS deployment is not included; the API containerises with any standard Python image.
