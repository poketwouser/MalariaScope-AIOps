"""Evaluate stage: test metrics at 0.5 and at a threshold tuned on val.

parasitized is the positive class. the threshold is picked on the val split
(never on test) so recall reaches evaluate.target_recall.
"""

import argparse
import json
from pathlib import Path

import keras
import matplotlib.pyplot as plt
import mlflow
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from malariascope.common import get_logger, load_params, write_json
from malariascope.labels import CLASS_NAMES, POSITIVE
from malariascope.tracking import tracking_uri

log = get_logger("evaluate")


def predict_labels(p_pos, threshold):
    other = 1 - POSITIVE
    return np.where(p_pos >= threshold, POSITIVE, other)


def pick_threshold(y_true, p_pos, target_recall):
    _, tpr, thresholds = roc_curve(y_true == POSITIVE, p_pos)
    # thresholds fall as tpr rises, so the first one reaching the target is the
    # highest, i.e. the one that costs the fewest false positives
    reached = np.flatnonzero(tpr >= target_recall)
    return float(min(thresholds[reached[0]], 1.0))


def compute_metrics(y_true, p_pos, threshold=0.5):
    y_pred = predict_labels(p_pos, threshold)
    is_pos = y_true == POSITIVE
    pred_pos = y_pred == POSITIVE
    tn, fp, fn, tp = confusion_matrix(is_pos, pred_pos, labels=[False, True]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "recall_parasitized": float(recall_score(is_pos, pred_pos)),
        "precision_parasitized": float(precision_score(is_pos, pred_pos)),
        "f1_parasitized": float(f1_score(is_pos, pred_pos)),
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "roc_auc": float(roc_auc_score(is_pos, p_pos)),
        "false_negatives": int(fn),
        "false_positives": int(fp),
        "n": int(len(y_true)),
    }


def plot_confusion(y_true, y_pred, threshold, out):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASS_NAMES)))
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            colour = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, cm[i, j], ha="center", va="center", color=colour)
    ax.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    ax.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(f"Confusion matrix (test, threshold {threshold:.3f})")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_roc(y_true, p_pos, auc, out):
    fpr, tpr, _ = roc_curve(y_true == POSITIVE, p_pos)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot(fpr, tpr, label=f"model (AUC {auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", label="chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve (parasitized = positive)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--model", default="models/model.keras")
    parser.add_argument("--train-metrics", default="results/train.json")
    parser.add_argument("--metrics", default="results/evaluate.json")
    parser.add_argument("--plots-dir", default="results/plots")
    args = parser.parse_args()

    params = load_params("evaluate")
    model = keras.models.load_model(args.model)

    def positive_probs(split):
        with np.load(Path(args.data_dir) / f"{split}.npz") as f:
            x, y = f["x"], f["y"]
        return y, model.predict(x, batch_size=256, verbose=0)[:, POSITIVE]

    y_val, p_val = positive_probs("val")
    y_test, p_test = positive_probs("test")

    threshold = pick_threshold(y_val, p_val, params["target_recall"])
    default = compute_metrics(y_test, p_test, 0.5)
    tuned = compute_metrics(y_test, p_test, threshold)
    log.info("test metrics at 0.5: %s", default)
    log.info("val-tuned threshold %.4f, test metrics: %s", threshold, tuned)

    plots = Path(args.plots_dir)
    plots.mkdir(parents=True, exist_ok=True)
    y_pred = predict_labels(p_test, threshold)
    plot_confusion(y_test, y_pred, threshold, plots / "confusion_matrix.png")
    plot_roc(y_test, p_test, tuned["roc_auc"], plots / "roc_curve.png")
    report = classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=4)
    log.info("classification report (tuned threshold):\n%s", report)

    run_id = json.loads(Path(args.train_metrics).read_text())["run_id"]
    mlflow.set_tracking_uri(tracking_uri())
    with mlflow.start_run(run_id=run_id):
        mlflow.log_param("decision_threshold", round(threshold, 6))
        mlflow.log_param("target_recall", params["target_recall"])
        mlflow.log_metrics({f"test_{k}": v for k, v in default.items()})
        mlflow.log_metrics({f"test_tuned_{k}": v for k, v in tuned.items()})
        mlflow.log_artifact(str(plots / "confusion_matrix.png"))
        mlflow.log_artifact(str(plots / "roc_curve.png"))
        mlflow.log_text(report, "classification_report.txt")

    write_json(
        {"run_id": run_id, "threshold": threshold, "default": default, "tuned": tuned},
        args.metrics,
    )


if __name__ == "__main__":
    main()
