"""MLflow helpers: tracking URI and the reproducibility tags put on every run.

Every run is tagged with the Git commit and the DVC hash of the data it used,
so it can be traced back to exact code and data (development guideline:
"reproducible via a specific Git commit hash and a corresponding MLflow run ID").
"""

import os
import subprocess
from pathlib import Path

import yaml

# the tracking server (Lecture 3) runs with a SQLite backend, which the Model
# Registry needs (Lecture 10), and proxies artifacts so their URIs stay portable:
#   mlflow server --backend-store-uri sqlite:///mlflow.db \
#     --artifacts-destination ./mlartifacts --port 5000
DEFAULT_TRACKING_URI = "http://localhost:5000"
MODEL_NAME = "malariascope"


def tracking_uri():
    return os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)


def git(*args):
    try:
        out = subprocess.run(["git", *args], capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def dvc_out_hash(stage, path, lock_file="dvc.lock"):
    """Return the md5 DVC recorded for an output of a stage, or "" if unknown."""
    lock = Path(lock_file)
    if not lock.exists():
        return ""
    stages = yaml.safe_load(lock.read_text()).get("stages", {})
    for out in stages.get(stage, {}).get("outs", []):
        if out.get("path") == path:
            return out.get("md5", "")
    return ""


def source_is_dirty():
    """True if tracked files other than pipeline outputs have uncommitted changes.

    dvc.lock and results/ are left out: DVC deletes a stage's outputs before it
    runs and rewrites dvc.lock between stages, so under `dvc repro` they always
    differ from the commit and say nothing about the code.
    """
    return bool(
        git(
            "status",
            "--porcelain",
            "--untracked-files=no",
            "--",
            ".",
            ":(exclude)dvc.lock",
            ":(exclude)results",
        )
    )


def run_tags(data_stage, data_path):
    """Tags that tie a run to its code and data versions."""
    return {
        "git_commit": git("rev-parse", "HEAD") or "unknown",
        # uncommitted changes mean the commit alone can't reproduce the run
        "git_dirty": str(source_is_dirty()),
        "data_path": data_path,
        "data_md5": dvc_out_hash(data_stage, data_path) or "unknown",
        # set by `dvc exp run`, links the MLflow run to the DVC experiment
        "dvc_exp": os.getenv("DVC_EXP_NAME", ""),
    }
