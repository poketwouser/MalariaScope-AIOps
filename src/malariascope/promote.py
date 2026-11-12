"""Promote stage: register the new model, make it champion only if it is better.

challenger and champion are scored on the same test split, each at its own
tuned threshold. the old champion keeps a previous_champion alias for rollback.
the model server picks up the new @champion by itself (serve.py).
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


def score_alias(client, alias, x_test, y_test):
    # load the model by alias, like the serving exercise, and score it at the
    # threshold stored on that version
    version = client.get_model_version_by_alias(MODEL_NAME, alias)
    model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@{alias}")
    p_pos = np.asarray(model.predict(x_test))[:, POSITIVE]
    threshold = float(version.tags["decision_threshold"])
    return {
        "version": int(version.version),
        "threshold": threshold,
        **compute_metrics(y_test, p_pos, threshold),
    }


def promotion_gate(challenger, champion, min_gain, min_recall):
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

    challenger = score_alias(client, CHALLENGER, x_test, y_test)
    champion = score_alias(client, CHAMPION, x_test, y_test) if champion_v else None
    promote, reason = promotion_gate(
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
