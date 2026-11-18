"""Airflow DAG: weekly retraining. Mount this project at /opt/recsys inside the Airflow container
(or edit PROJECT). Tasks run the same scripts you run by hand."""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

PROJECT = "/opt/recsys"
with DAG("recsys_weekly_retrain", start_date=datetime(2026, 11, 1), schedule="@weekly", catchup=False,
         default_args={"retries": 1, "retry_delay": timedelta(minutes=5)}, tags=["recsys"]) as dag:
    steps = [("data_prep", "python data_prep.py --ml-dir $ML_DIR --sample-users 20000"),
             ("train_retrieval", "python train_retrieval.py --epochs 10"),
             ("build_faiss", "python build_faiss.py"),
             ("train_ranker", "python train_ranker.py --epochs 5"),
             ("evaluate", "python evaluate.py"),
             ("export_triton", "python export_triton.py")]
    prev = None
    for name, cmd in steps:
        t = BashOperator(task_id=name, bash_command=f"cd {PROJECT}/src && TF_USE_LEGACY_KERAS=1 {cmd}")
        if prev:
            prev >> t
        prev = t
