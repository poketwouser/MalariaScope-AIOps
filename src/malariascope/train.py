"""Train stage: fit the CNN on the train split and track the run in MLflow.

Logged to MLflow: every train param, data sizes, per-epoch loss and accuracy
(train and val), training time, Git and DVC tags, the training curves, the
model summary and the model itself with an input/output signature.

Output:
    models/model.keras                  trained model (DVC output)
    results/train.json                  run id and best val metrics (DVC metrics)
    results/plots/training_curves.png   loss and accuracy per epoch (DVC plot)

Run: PYTHONPATH=src python -m malariascope.train
"""

import argparse
import os
import platform
from pathlib import Path

import keras
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402

from malariascope.common import Timer, get_logger, load_params, write_json  # noqa: E402
from malariascope.model import build_model  # noqa: E402
from malariascope.tracking import run_tags, tracking_uri  # noqa: E402

log = get_logger("train")


def load_split(data_dir, split):
    with np.load(Path(data_dir) / f"{split}.npz") as f:
        return f["x"], f["y"]


class MlflowEpochLogger(keras.callbacks.Callback):
    """Log Keras' epoch metrics to the active MLflow run."""

    def on_epoch_end(self, epoch, logs=None):
        mlflow.log_metrics({k: float(v) for k, v in (logs or {}).items()}, step=epoch)


def plot_history(history, out):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    epochs = np.arange(1, len(history["loss"]) + 1)
    for ax, key, label in (
        (axes[0], "loss", "Loss"),
        (axes[1], "accuracy", "Accuracy"),
    ):
        ax.plot(epochs, history[key], marker="o", color="#0072B2", label="train")
        ax.plot(epochs, history[f"val_{key}"], marker="s", color="#D55E00", label="val")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(f"{label} (linear scale)")
        ax.legend()
        ax.set_title(
            f"{label} per epoch\nA widening train/val gap would mean overfitting.",
            fontsize=10,
            loc="left",
        )
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def train(params, image_size, data_dir, model_path, plots_dir):
    x_train, y_train = load_split(data_dir, "train")
    x_val, y_val = load_split(data_dir, "val")
    keras.utils.set_random_seed(params["seed"])

    model = build_model(
        image_size=image_size,
        filters=params["filters"],
        dense_units=params["dense_units"],
        dropout=params["dropout"],
        augment=params["augment"],
        learning_rate=params["learning_rate"],
    )
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=params["patience"], restore_best_weights=True
        ),
        MlflowEpochLogger(),
    ]

    mlflow.log_params(
        {**params, "filters": str(params["filters"]), "image_size": image_size}
    )
    mlflow.log_params({"n_train": len(y_train), "n_val": len(y_val)})
    mlflow.set_tags(
        {
            "python": platform.python_version(),
            "keras": keras.__version__,
            "cpu": platform.processor() or platform.machine(),
        }
    )

    with Timer() as t:
        history = model.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            epochs=params["epochs"],
            batch_size=params["batch_size"],
            callbacks=callbacks,
            verbose=2,
        )
    history = history.history
    best = int(np.argmin(history["val_loss"]))
    summary = {
        "best_epoch": best + 1,
        "epochs_run": len(history["loss"]),
        "val_loss": float(history["val_loss"][best]),
        "val_accuracy": float(history["val_accuracy"][best]),
        "train_seconds": round(t.seconds, 1),
    }
    mlflow.log_metrics({f"best_{k}": v for k, v in summary.items()})
    log.info("training done: %s", summary)

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)

    curves = Path(plots_dir) / "training_curves.png"
    curves.parent.mkdir(parents=True, exist_ok=True)
    plot_history(history, curves)
    mlflow.log_artifact(str(curves))
    lines = []
    model.summary(print_fn=lines.append)
    mlflow.log_text("\n".join(lines), "model_summary.txt")

    sample = x_val[:4]
    mlflow.tensorflow.log_model(
        model,
        name="model",
        signature=infer_signature(sample, model.predict(sample, verbose=0)),
        input_example=sample[:1],
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--model", default="models/model.keras")
    parser.add_argument("--metrics", default="results/train.json")
    parser.add_argument("--plots-dir", default="results/plots")
    args = parser.parse_args()

    params = load_params("train")
    image_size = load_params("preprocess")["image_size"]

    mlflow.set_tracking_uri(tracking_uri())
    # `mlflow run` has already activated a run in its --experiment-name, and setting
    # another experiment here would make start_run fail (MLflow Exercise 2)
    if not os.getenv("MLFLOW_RUN_ID"):
        mlflow.set_experiment(params["experiment"])
    with mlflow.start_run(run_name="train") as run:
        mlflow.set_tags(run_tags("preprocess", args.data_dir))
        summary = train(params, image_size, args.data_dir, args.model, args.plots_dir)

    write_json({"run_id": run.info.run_id, **summary}, args.metrics)
    log.info("mlflow run %s", run.info.run_id)


if __name__ == "__main__":
    main()
