"""Airflow DAG: retrain the churn model weekly. (Not executed in this repo's tests;
requires an Airflow installation. Point `csv` at your real data source.)"""
from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator

import pandas as pd

from pipeline import make_synthetic, train


def retrain():
    df = make_synthetic()  # replace with: pd.read_csv("s3://.../customers.csv")
    print(train(df, "model.joblib"))


with DAG("churn_weekly_retrain", start_date=datetime(2026, 1, 1), schedule="@weekly", catchup=False) as dag:
    PythonOperator(task_id="retrain_model", python_callable=retrain)
