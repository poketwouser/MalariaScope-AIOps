"""Validate stage: check every raw image in parallel with Ray Data.

Each image must decode, be RGB, have both sides within [min_side, max_side] and
not be blank (pixel std >= min_std). Exact duplicates (same SHA-1) are dropped
so the same picture can't end up in two splits.

Output:
    data/validated/manifest.csv   valid rows plus width, height, mean, std, sha1
    results/validate.json         counts per drop reason, timing, throughput

Run: PYTHONPATH=src python -m malariascope.data.validate
"""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import ray
from PIL import Image, UnidentifiedImageError

from malariascope.common import Timer, get_logger, load_params, write_json

log = get_logger("validate")


def check_image(path, min_side, max_side, min_std):
    """Return (reason, stats) for one image file. reason is "ok" when it passes."""
    stats = {"width": 0, "height": 0, "mean": 0.0, "std": 0.0, "sha1": ""}
    try:
        data = Path(path).read_bytes()
        stats["sha1"] = hashlib.sha1(data).hexdigest()
        with Image.open(path) as img:
            img.load()
            mode = img.mode
            stats["width"], stats["height"] = img.size
            pixels = np.asarray(img.convert("RGB"), dtype=np.float32)
    except (OSError, UnidentifiedImageError) as e:
        log.warning("cannot read %s: %s", path, e)
        return "unreadable", stats

    stats["mean"] = float(pixels.mean())
    stats["std"] = float(pixels.std())
    if mode != "RGB":
        return "not_rgb", stats
    if min(img.size) < min_side or max(img.size) > max_side:
        return "bad_size", stats
    if stats["std"] < min_std:
        return "blank", stats
    return "ok", stats


def check_batch(batch, root, min_side, max_side, min_std):
    """Ray Data batch function: add the check result and image stats as columns."""
    reasons, stats = [], []
    for rel_path in batch["path"]:
        reason, s = check_image(Path(root) / rel_path, min_side, max_side, min_std)
        reasons.append(reason)
        stats.append(s)
    batch = batch.copy()
    batch["reason"] = reasons
    for key in ("width", "height", "mean", "std", "sha1"):
        batch[key] = [s[key] for s in stats]
    return batch


def validate(manifest, root, params):
    """Run the checks on all rows with Ray and return the checked DataFrame."""
    ds = ray.data.from_pandas(manifest).map_batches(
        check_batch,
        batch_format="pandas",
        batch_size=params["batch_size"],
        fn_kwargs={
            "root": str(Path(root).resolve()),
            "min_side": params["min_side"],
            "max_side": params["max_side"],
            "min_std": params["min_std"],
        },
    )
    checked = ds.to_pandas()
    # mark repeats of an already-seen valid image as duplicates
    dup = checked["reason"].eq("ok") & checked.duplicated("sha1", keep="first")
    checked.loc[dup, "reason"] = "duplicate"
    return checked.sort_values("path").reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--out-dir", default="data/validated")
    parser.add_argument("--metrics", default="results/validate.json")
    args = parser.parse_args()

    params = load_params("validate")
    manifest = pd.read_csv(Path(args.raw_dir) / "manifest.csv")
    log.info("validating %d images", len(manifest))

    ray.init()
    try:
        with Timer() as t:
            checked = validate(manifest, args.raw_dir, params)
    finally:
        ray.shutdown()

    valid = checked[checked["reason"] == "ok"].drop(columns="reason")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    valid.to_csv(out_dir / "manifest.csv", index=False)

    dropped = checked.loc[checked["reason"] != "ok", "reason"].value_counts().to_dict()
    rate = len(checked) / t.seconds
    log.info(
        "%d of %d valid in %.1fs (%.0f images/s), dropped %s",
        len(valid),
        len(checked),
        t.seconds,
        rate,
        dropped,
    )
    write_json(
        {
            "num_checked": len(checked),
            "num_valid": len(valid),
            "dropped": dropped,
            "class_counts": valid["label_name"].value_counts().to_dict(),
            "seconds": round(t.seconds, 2),
            "images_per_sec": round(rate, 1),
        },
        args.metrics,
    )


if __name__ == "__main__":
    main()
