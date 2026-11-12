"""mlflow helpers. every run gets the git commit and the dvc hash of its data"""

import os
import subprocess
from pathlib import Path

import yaml

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
    lock = Path(lock_file)
    if not lock.exists():
        return ""
    stages = yaml.safe_load(lock.read_text()).get("stages", {})
    for out in stages.get(stage, {}).get("outs", []):
        if out.get("path") == path:
            return out.get("md5", "")
    return ""


def source_is_dirty():
    # dvc.lock and results/ are left out on purpose. dvc deletes a stage's outputs
    # before running it and rewrites dvc.lock between stages, so during `dvc repro`
    # they are always "dirty" even when the code is clean
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
    return {
        "git_commit": git("rev-parse", "HEAD") or "unknown",
        # uncommitted changes mean the commit alone can't reproduce the run
        "git_dirty": str(source_is_dirty()),
        "data_path": data_path,
        "data_md5": dvc_out_hash(data_stage, data_path) or "unknown",
        # set by `dvc exp run`, links the MLflow run to the DVC experiment
        "dvc_exp": os.getenv("DVC_EXP_NAME", ""),
    }
