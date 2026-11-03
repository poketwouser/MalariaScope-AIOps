"""Training pipeline DAG: the DVC stages as Airflow tasks (LLD section 6).

Each task runs `dvc repro <stage>` in the repo mounted at /opt/project, with the
pipeline venv on PATH. DVC skips a stage whose inputs did not change, so a run
with no new data finishes in under a minute, while a run after a feedback export
retrains, evaluates and goes through the promotion gate. Promotion only moves
the @champion alias; the model server follows the alias by itself.

Each task prints the stage's metrics JSON as its last line, which Airflow keeps
as the task's XCom. The Streamlit Pipeline page reads throughput from there.
"""

import os
from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator

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
