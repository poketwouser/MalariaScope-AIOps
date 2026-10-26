"""Evaluate stage: score the trained model on the held-out test split.

Parasitized is the positive class: missing an infection is the costly error,
so recall on it is reported alongside accuracy. Metrics, the confusion matrix,
the ROC curve and the classification report are logged into the same MLflow
run as training.

Output:
    results/evaluate.json              test metrics (DVC metrics)
    results/plots/confusion_matrix.png
    results/plots/roc_curve.png

Run: PYTHONPATH=src python -m malariascope.evaluate
"""

import argparse
import json
from pathlib import Path

import keras
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from malariascope.common import get_logger, write_json  # noqa: E402
from malariascope.model import CLASS_NAMES  # noqa: E402
from malariascope.tracking import tracking_uri  # noqa: E402

log = get_logger("evaluate")

POSITIVE = CLASS_NAMES.index("parasitized")


def compute_metrics(y_true, probs):
    """Test metrics with parasitized as the positive class."""
    y_pred = probs.argmax(axis=1)
    is_pos = y_true == POSITIVE
    pred_pos = y_pred == POSITIVE
    tn, fp, fn, tp = confusion_matrix(is_pos, pred_pos, labels=[False, True]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "recall_parasitized": float(recall_score(is_pos, pred_pos)),
        "precision_parasitized": float(precision_score(is_pos, pred_pos)),
        "f1_parasitized": float(f1_score(is_pos, pred_pos)),
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "roc_auc": float(roc_auc_score(is_pos, probs[:, POSITIVE])),
        "false_negatives": int(fn),
        "false_positives": int(fp),
        "n_test": int(len(y_true)),
    }


def plot_confusion(y_true, y_pred, out):
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
    ax.set_title(
        "Confusion matrix (test split, counts)\n"
        "Top-right cell = infections the model missed.",
        fontsize=10,
        loc="left",
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_roc(y_true, probs, auc, out):
    fpr, tpr, _ = roc_curve(y_true == POSITIVE, probs[:, POSITIVE])
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot(fpr, tpr, color="#D55E00", label=f"model (AUC {auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#999999", label="chance")
    ax.set_xlabel("False positive rate (linear, 0-1)")
    ax.set_ylabel("True positive rate (linear, 0-1)")
    ax.legend(loc="lower right")
    ax.set_title(
        "ROC curve, parasitized as positive\n"
        "Closer to the top-left corner is better.",
        fontsize=10,
        loc="left",
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--model", default="models/model.keras")
    parser.add_argument("--train-metrics", default="results/train.json")
    parser.add_argument("--metrics", default="results/evaluate.json")
    parser.add_argument("--plots-dir", default="results/plots")
    args = parser.parse_args()

    with np.load(Path(args.data_dir) / "test.npz") as f:
        x_test, y_test = f["x"], f["y"]
    model = keras.models.load_model(args.model)
    probs = model.predict(x_test, batch_size=256, verbose=0)
    metrics = compute_metrics(y_test, probs)
    log.info("test metrics: %s", metrics)

    plots = Path(args.plots_dir)
    plots.mkdir(parents=True, exist_ok=True)
    plot_confusion(y_test, probs.argmax(axis=1), plots / "confusion_matrix.png")
    plot_roc(y_test, probs, metrics["roc_auc"], plots / "roc_curve.png")
    report = classification_report(
        y_test, probs.argmax(axis=1), target_names=CLASS_NAMES, digits=4
    )
    log.info("classification report:\n%s", report)

    run_id = json.loads(Path(args.train_metrics).read_text())["run_id"]
    mlflow.set_tracking_uri(tracking_uri())
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items()})
        mlflow.log_artifact(str(plots / "confusion_matrix.png"))
        mlflow.log_artifact(str(plots / "roc_curve.png"))
        mlflow.log_text(report, "classification_report.txt")

    write_json({"run_id": run_id, **metrics}, args.metrics)


if __name__ == "__main__":
    main()
