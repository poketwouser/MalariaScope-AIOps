"""Promote stage: register the new model and promote it only if it is better.

Follows the Lecture 10 registry pattern:
  1. register the run's model as a new version and point "challenger" at it
  2. score challenger and current "champion" on the SAME current test split,
     each with its own val-tuned threshold (stored on the version as a tag)
  3. move "champion" to the challenger only if its accuracy is at least
     promote.min_gain higher and its parasitized recall stays >= min_recall
  4. the replaced version keeps the "previous_champion" alias, so a rollback is
     a single alias move (see rollback.py)

Serving loads models:/malariascope@champion, which is resolved at start-up, so
the model server must be restarted after a promotion (Lecture 10).

Output: results/promote.json   decision, versions and the comparison

Run: PYTHONPATH=src python -m malariascope.promote
"""

import argparse
import json
from pathlib import Path

import mlflow
import numpy as np
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from malariascope.common import get_logger, load_params, write_json
from malariascope.evaluate import compute_metrics
from malariascope.labels import POSITIVE
from malariascope.tracking import MODEL_NAME, tracking_uri

log = get_logger("promote")

CHAMPION, CHALLENGER, PREVIOUS = "champion", "challenger", "previous_champion"


def version_by_alias(client, alias):
    try:
        return client.get_model_version_by_alias(MODEL_NAME, alias)
    except MlflowException:
        return None


def score_version(version, x_test, y_test):
    """Test metrics of a registered version at its own tuned threshold."""
    model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{version.version}")
    p_pos = np.asarray(model.predict(x_test))[:, POSITIVE]
    threshold = float(version.tags["decision_threshold"])
    return {
        "version": int(version.version),
        "threshold": threshold,
        **compute_metrics(y_test, p_pos, threshold),
    }


def decide(challenger, champion, min_gain, min_recall):
    """Return (promote, reason) for challenger vs champion metrics."""
    if challenger["recall_parasitized"] < min_recall:
        return False, (
            f"challenger recall {challenger['recall_parasitized']:.4f} "
            f"is below the floor {min_recall}"
        )
    if champion is None:
        return True, "no champion yet"
    gain = challenger["accuracy"] - champion["accuracy"]
    if gain >= min_gain:
        return True, f"accuracy gain {gain:+.4f} >= {min_gain}"
    return False, f"accuracy gain {gain:+.4f} < {min_gain}"


def register(client, run_id, evaluation):
    """Register the run's model and tag the version with its evaluation."""
    version = mlflow.register_model(f"runs:/{run_id}/model", MODEL_NAME)
    run = client.get_run(run_id)
    tuned = evaluation["tuned"]
    tags = {
        "decision_threshold": evaluation["threshold"],
        "test_accuracy": tuned["accuracy"],
        "test_recall_parasitized": tuned["recall_parasitized"],
        "git_commit": run.data.tags.get("git_commit", ""),
        "data_md5": run.data.tags.get("data_md5", ""),
    }
    for key, value in tags.items():
        client.set_model_version_tag(MODEL_NAME, version.version, key, str(value))
    client.set_registered_model_alias(MODEL_NAME, CHALLENGER, version.version)
    log.info(
        "registered %s version %s from run %s", MODEL_NAME, version.version, run_id
    )
    return client.get_model_version(MODEL_NAME, version.version)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--evaluation", default="results/evaluate.json")
    parser.add_argument("--out", default="results/promote.json")
    args = parser.parse_args()

    params = load_params("promote")
    evaluation = json.loads(Path(args.evaluation).read_text())
    with np.load(Path(args.data_dir) / "test.npz") as f:
        x_test, y_test = f["x"], f["y"]

    mlflow.set_tracking_uri(tracking_uri())
    client = MlflowClient()
    challenger_v = register(client, evaluation["run_id"], evaluation)
    champion_v = version_by_alias(client, CHAMPION)

    challenger = score_version(challenger_v, x_test, y_test)
    champion = score_version(champion_v, x_test, y_test) if champion_v else None
    promote, reason = decide(
        challenger, champion, params["min_gain"], params["min_recall"]
    )

    if promote:
        if champion_v:
            client.set_registered_model_alias(MODEL_NAME, PREVIOUS, champion_v.version)
        client.set_registered_model_alias(MODEL_NAME, CHAMPION, challenger_v.version)
    log.info(
        "%s version %s: %s",
        "PROMOTED" if promote else "kept champion,",
        challenger_v.version,
        reason,
    )

    if promote:
        champion_now = int(challenger_v.version)
    else:
        champion_now = int(champion_v.version) if champion_v else None
    write_json(
        {
            "promoted": promote,
            "reason": reason,
            "challenger": challenger,
            "champion_before": champion,
            "champion_now": champion_now,
        },
        args.out,
    )


if __name__ == "__main__":
    main()
