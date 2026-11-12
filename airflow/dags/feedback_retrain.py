# every 30 min: if enough new feedback came in (retrain.min_feedback), export it
# into data/feedback, dvc add it and start training_pipeline.
# trigger with conf {"min_feedback": N} to use a different threshold once.

import logging
import os
from datetime import datetime, timedelta

import requests
import yaml
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import ShortCircuitOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from run_metrics import push_run_result

PROJECT = os.getenv("PROJECT_DIR", "/opt/project")
PIPELINE_BIN = os.getenv("PIPELINE_BIN", "/opt/pipeline/bin")
BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8000").rstrip("/")
ENV = {"PATH": f"{PIPELINE_BIN}:{os.environ.get('PATH', '/usr/bin:/bin')}"}

log = logging.getLogger(__name__)


def enough_feedback(dag_run=None):
    min_needed = (dag_run.conf or {}).get("min_feedback")
    if min_needed is None:
        with open(f"{PROJECT}/params.yaml") as f:
            min_needed = yaml.safe_load(f)["retrain"]["min_feedback"]
    r = requests.get(f"{BACKEND_URL}/feedback/stats", timeout=10)
    r.raise_for_status()
    pending = r.json()["since_last_export"]
    log.info("feedback since last export: %d (need %d)", pending, min_needed)
    return pending >= int(min_needed)


with DAG(
    dag_id="feedback_retrain",
    description="Export new user feedback into DVC and retrain when there is enough",
    start_date=datetime(2026, 11, 1),
    schedule=timedelta(minutes=30),
    catchup=False,
    max_active_runs=1,
    default_args={"owner": "malariascope", "retries": 0},
    on_success_callback=push_run_result,
    on_failure_callback=push_run_result,
    tags=["malariascope", "feedback"],
) as dag:
    check = ShortCircuitOperator(
        task_id="enough_feedback", python_callable=enough_feedback
    )
    export = BashOperator(
        task_id="export_feedback",
        bash_command=(
            f"cd {PROJECT} && PYTHONPATH=src python -m malariascope.data.feedback"
            " && dvc add data/feedback"
            " && PYTHONPATH=src python -m malariascope.pushmetrics feedback_export"
            " results/feedback_export.json"
            " && python -c 'import json; print(json.dumps("
            'json.load(open("results/feedback_export.json"))))\''
        ),
        env=ENV,
        append_env=True,
    )
    # does not wait: with the SequentialExecutor a waiting task would hold the only
    # slot and the training run could never start
    retrain = TriggerDagRunOperator(
        task_id="start_training",
        trigger_dag_id="training_pipeline",
        conf={"reason": "feedback"},
        wait_for_completion=False,
    )

    check >> export >> retrain
