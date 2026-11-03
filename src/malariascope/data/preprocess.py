"""Preprocess stage: resize the split images in parallel with Ray Data.

Output:
    data/processed/{train,val,test}.npz   x: (N, size, size, 3) uint8, y: (N,) int
    results/preprocess.json               rows per split, timing, throughput

Run: PYTHONPATH=src python -m malariascope.data.preprocess
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import ray
from PIL import Image

from malariascope.common import Timer, get_logger, load_params, write_json
from malariascope.image import to_model_input

log = get_logger("preprocess")

SPLITS = ("train", "val", "test")


def resize_batch(batch, roots, size):
    """Ray Data batch function: load and resize each image in the batch.

    roots maps the manifest's "source" column (raw or feedback) to its folder.
    """
    images = []
    for rel_path, source in zip(batch["path"], batch["source"]):
        with Image.open(Path(roots[source]) / rel_path) as img:
            images.append(to_model_input(img, size))
    return {
        "x": np.stack(images),
        "y": batch["label"].to_numpy(dtype=np.int64),
        "split": batch["split"].to_numpy(dtype=str),
    }


def preprocess(manifest, roots, size, batch_size, num_blocks):
    """Resize all images with Ray and return {split: (x, y)}."""
    # several blocks, or Ray runs everything as a single task on one CPU
    ds = ray.data.from_pandas(manifest, override_num_blocks=num_blocks).map_batches(
        resize_batch,
        batch_format="pandas",
        batch_size=batch_size,
        fn_kwargs={
            "roots": {k: str(Path(v).resolve()) for k, v in roots.items()},
            "size": size,
        },
    )
    parts = {s: ([], []) for s in SPLITS}
    for batch in ds.iter_batches(batch_size=batch_size, batch_format="numpy"):
        for s in SPLITS:
            mask = batch["split"] == s
            parts[s][0].append(batch["x"][mask])
            parts[s][1].append(batch["y"][mask])
    return {
        s: (np.concatenate(xs), np.concatenate(ys)) for s, (xs, ys) in parts.items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", default="data/split/manifest.csv")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--feedback-dir", default="data/feedback")
    parser.add_argument("--out-dir", default="data/processed")
    parser.add_argument("--metrics", default="results/preprocess.json")
    args = parser.parse_args()

    params = load_params("preprocess")
    manifest = pd.read_csv(args.manifest)
    if "source" not in manifest:
        manifest["source"] = "raw"
    roots = {"raw": args.raw_dir, "feedback": args.feedback_dir}
    log.info("preprocessing %d images to %dpx", len(manifest), params["image_size"])

    ray.init()
    # progress bars flood the DVC/Airflow logs; throughput is logged instead
    ctx = ray.data.DataContext.get_current()
    ctx.enable_progress_bars = False
    # keep manifest order so the .npz files are identical from run to run
    ctx.execution_options.preserve_order = True
    try:
        with Timer() as t:
            data = preprocess(
                manifest,
                roots,
                params["image_size"],
                params["batch_size"],
                params["num_blocks"],
            )
    finally:
        ray.shutdown()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for s, (x, y) in data.items():
        np.savez(out_dir / f"{s}.npz", x=x, y=y)
        log.info("%s: x %s, y %s", s, x.shape, np.bincount(y).tolist())

    rate = len(manifest) / t.seconds
    log.info(
        "resized %d images in %.1fs (%.0f images/s)", len(manifest), t.seconds, rate
    )
    write_json(
        {
            "sizes": {s: int(len(y)) for s, (_, y) in data.items()},
            "image_size": params["image_size"],
            "feedback_in_train": int((manifest["source"] == "feedback").sum()),
            "seconds": round(t.seconds, 2),
            "images_per_sec": round(rate, 1),
        },
        args.metrics,
    )


if __name__ == "__main__":
    main()
