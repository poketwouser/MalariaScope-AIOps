"""Split stage: fixed, stratified train/val/test split of the validated images.

The split is seeded, so the same params give the same split every time, and
stratified, so every split keeps the dataset's class balance.

Output:
    data/split/manifest.csv   validated manifest plus a "split" column
    results/split.json        rows per split and class

Run: PYTHONPATH=src python -m malariascope.data.split
"""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from malariascope.common import get_logger, load_params, write_json

log = get_logger("split")


def make_split(manifest, fractions, seed):
    """Return a copy of manifest with a "split" column of train/val/test."""
    total = fractions["train"] + fractions["val"] + fractions["test"]
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"split fractions must add up to 1, got {total}")

    train, rest = train_test_split(
        manifest,
        train_size=fractions["train"],
        stratify=manifest["label"],
        random_state=seed,
    )
    val_share = fractions["val"] / (fractions["val"] + fractions["test"])
    val, test = train_test_split(
        rest, train_size=val_share, stratify=rest["label"], random_state=seed
    )
    parts = [
        df.assign(split=name)
        for name, df in (("train", train), ("val", val), ("test", test))
    ]
    return pd.concat(parts).sort_values("path").reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--in-manifest", default="data/validated/manifest.csv")
    parser.add_argument("--out-dir", default="data/split")
    parser.add_argument("--metrics", default="results/split.json")
    args = parser.parse_args()

    params = load_params("data")
    manifest = pd.read_csv(args.in_manifest)
    split = make_split(manifest, params["split"], params["seed"])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    split.to_csv(out_dir / "manifest.csv", index=False)

    counts = split.groupby(["split", "label_name"]).size().unstack(fill_value=0)
    log.info("split sizes:\n%s", counts)
    write_json(
        {
            "sizes": split["split"].value_counts().to_dict(),
            "class_counts": {s: row.to_dict() for s, row in counts.iterrows()},
            "seed": params["seed"],
        },
        args.metrics,
    )


if __name__ == "__main__":
    main()
