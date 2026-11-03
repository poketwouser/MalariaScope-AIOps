"""Split stage: fixed, stratified train/val/test split of the validated images.

The split is seeded, so the same params give the same split every time, and
stratified, so every split keeps the dataset's class balance.

User feedback (data/feedback, exported by the retraining DAG) is added to the
train split only. Val and test stay fixed, so a retrained challenger is compared
with the champion on exactly the same images. Feedback whose pixels match a val
or test image is dropped: the UI's sample images come from the test set, and
training on them would leak test data into the comparison.

Output:
    data/split/manifest.csv   validated manifest plus "split" and "source" columns
    results/split.json        rows per split and class, feedback counts

Run: PYTHONPATH=src python -m malariascope.data.split
"""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from PIL import Image

from malariascope.common import get_logger, load_params, write_json
from malariascope.image import pixel_hash

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


def add_feedback(split, feedback, raw_root):
    """Append feedback rows to the train split; return (manifest, counts).

    Drops feedback that duplicates a val/test image (leakage) and keeps only the
    latest label when the same cell was sent more than once.
    """
    split = split.assign(source="raw")
    counts = {"received": len(feedback), "leaked_dropped": 0, "duplicates_dropped": 0}
    if feedback.empty:
        counts["added"] = 0
        return split, counts

    held_out = split.loc[split["split"] != "train", "path"]
    held_out_hashes = set()
    for rel_path in held_out:
        with Image.open(Path(raw_root) / rel_path) as img:
            held_out_hashes.add(pixel_hash(img))
    leaked = feedback["pixel_sha1"].isin(held_out_hashes)
    counts["leaked_dropped"] = int(leaked.sum())
    feedback = feedback[~leaked]

    deduped = feedback.drop_duplicates("pixel_sha1", keep="last")
    counts["duplicates_dropped"] = len(feedback) - len(deduped)
    counts["added"] = len(deduped)

    rows = deduped[["path", "label", "label_name"]].assign(
        split="train", source="feedback"
    )
    # appended after the raw rows, so the raw order (and train.npz) is unchanged
    return pd.concat([split, rows], ignore_index=True), counts


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--in-manifest", default="data/validated/manifest.csv")
    parser.add_argument("--feedback-manifest", default="data/feedback/manifest.csv")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--out-dir", default="data/split")
    parser.add_argument("--metrics", default="results/split.json")
    args = parser.parse_args()

    params = load_params("data")
    manifest = pd.read_csv(args.in_manifest)
    split = make_split(manifest, params["split"], params["seed"])
    feedback_path = Path(args.feedback_manifest)
    feedback = pd.read_csv(feedback_path) if feedback_path.exists() else pd.DataFrame()
    split, fb_counts = add_feedback(split, feedback, args.raw_dir)
    log.info("feedback: %s", fb_counts)

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
            "feedback": fb_counts,
        },
        args.metrics,
    )


if __name__ == "__main__":
    main()
