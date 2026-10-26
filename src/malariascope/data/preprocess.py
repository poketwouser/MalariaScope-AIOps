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


def resize_batch(batch, root, size):
    """Ray Data batch function: load and resize each image in the batch."""
    images = []
    for rel_path in batch["path"]:
        with Image.open(Path(root) / rel_path) as img:
            images.append(to_model_input(img, size))
    return {
        "x": np.stack(images),
        "y": batch["label"].to_numpy(dtype=np.int64),
        "split": batch["split"].to_numpy(dtype=str),
    }


def preprocess(manifest, root, size, batch_size):
    """Resize all images with Ray and return {split: (x, y)}."""
    ds = ray.data.from_pandas(manifest).map_batches(
        resize_batch,
        batch_format="pandas",
        batch_size=batch_size,
        fn_kwargs={"root": str(Path(root).resolve()), "size": size},
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
    parser.add_argument("--out-dir", default="data/processed")
    parser.add_argument("--metrics", default="results/preprocess.json")
    args = parser.parse_args()

    params = load_params("preprocess")
    manifest = pd.read_csv(args.manifest)
    log.info("preprocessing %d images to %dpx", len(manifest), params["image_size"])

    ray.init()
    try:
        with Timer() as t:
            data = preprocess(
                manifest, args.raw_dir, params["image_size"], params["batch_size"]
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
            "seconds": round(t.seconds, 2),
            "images_per_sec": round(rate, 1),
        },
        args.metrics,
    )


if __name__ == "__main__":
    main()
