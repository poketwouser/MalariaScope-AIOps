# one task per dvc stage. dvc skips stages whose inputs didn't change, so a run
# without new data takes under a minute.

import os
from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator
from run_metrics import push_run_result

PROJECT = os.getenv("PROJECT_DIR", "/opt/project")
PIPELINE_BIN = os.getenv("PIPELINE_BIN", "/opt/pipeline/bin")
# the stage commands call plain `python`, so the pipeline venv goes first on PATH
ENV = {"PATH": f"{PIPELINE_BIN}:{os.environ.get('PATH', '/usr/bin:/bin')}"}
PRINT_JSON = (
    "python -c 'import json,sys; print(json.dumps(json.load(open(sys.argv[1]))))'"
)

# stage -> metrics file printed at the end (eda only writes figures)
STAGES = {
    "ingest": "results/ingest.json",
    "validate": "results/validate.json",
    "split": "results/split.json",
    "preprocess": "results/preprocess.json",
    "eda": None,
    "train": "results/train.json",
    "evaluate": "results/evaluate.json",
    "promote": "results/promote.json",
}


def repro(stage, metrics):
    cmd = f"cd {PROJECT} && dvc repro {stage}"
    # push before printing: the last stdout line must stay the JSON (the XCom)
    cmd += (
        f" && PYTHONPATH=src python -m malariascope.pushmetrics {stage} {metrics or ''}"
    )
    if metrics:
        cmd += f" && {PRINT_JSON} {metrics}"
    return BashOperator(task_id=stage, bash_command=cmd, env=ENV, append_env=True)


with DAG(
    dag_id="training_pipeline",
    description="DVC stages: ingest → validate → split → preprocess → train → "
    "evaluate → promote, then dvc push",
    start_date=datetime(2026, 11, 1),
    schedule=None,  # started from the UI, the REST API or feedback_retrain
    catchup=False,
    max_active_runs=1,  # one DVC run at a time on the shared repo
    default_args={"owner": "malariascope", "retries": 0},
    on_success_callback=push_run_result,
    on_failure_callback=push_run_result,
    tags=["malariascope", "dvc"],
) as dag:
    t = {stage: repro(stage, metrics) for stage, metrics in STAGES.items()}
    push = BashOperator(
        task_id="dvc_push",
        bash_command=f"cd {PROJECT} && dvc push",
        env=ENV,
        append_env=True,
    )

    t["ingest"] >> t["validate"] >> t["split"] >> t["preprocess"] >> t["train"]
    t["validate"] >> t["eda"]
    t["train"] >> t["evaluate"] >> t["promote"] >> push
    t["eda"] >> push
