"""Ingest stage: download the TFDS malaria dataset and export it as PNG files.

Output:
    data/raw/images/<class>/<index>.png
    data/raw/manifest.csv      columns: path, label, label_name
    results/ingest.json        counts and timing (DVC metrics)

Run: PYTHONPATH=src python -m malariascope.data.ingest
"""

import argparse
from pathlib import Path

import pandas as pd

from malariascope.common import Timer, get_logger, load_params, write_json

log = get_logger("ingest")


def export_dataset(name, tfds_dir, out_dir):
    """Write every image of the TFDS dataset to out_dir and return the manifest."""
    # imported here so the other stages don't pay TensorFlow's import time
    import tensorflow_datasets as tfds

    # progress bars redraw thousands of times and flood the DVC/Airflow logs
    tfds.disable_progress_bar()
    ds, info = tfds.load(
        name,
        split="train",  # the malaria dataset ships as a single split
        data_dir=str(tfds_dir),
        with_info=True,
        # keep the stored PNG bytes, no need to decode and re-encode
        decoders={"image": tfds.decode.SkipDecoding()},
    )
    label_names = info.features["label"].names
    log.info(
        "loaded %s: %d examples, labels %s",
        name,
        info.splits["train"].num_examples,
        label_names,
    )

    for label_name in label_names:
        (out_dir / "images" / label_name).mkdir(parents=True, exist_ok=True)

    rows = []
    for i, example in enumerate(tfds.as_numpy(ds)):
        label = int(example["label"])
        rel_path = Path("images") / label_name_of(label, label_names) / f"{i:05d}.png"
        (out_dir / rel_path).write_bytes(example["image"])
        rows.append(
            {"path": str(rel_path), "label": label, "label_name": label_names[label]}
        )
        if (i + 1) % 5000 == 0:
            log.info("exported %d images", i + 1)

    return pd.DataFrame(rows)


def label_name_of(label, label_names):
    if not 0 <= label < len(label_names):
        raise ValueError(
            f"unexpected label {label}, expected 0..{len(label_names) - 1}"
        )
    return label_names[label]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tfds-dir", default="data/tfds", help="TFDS download cache")
    parser.add_argument("--out-dir", default="data/raw")
    parser.add_argument("--metrics", default="results/ingest.json")
    args = parser.parse_args()

    params = load_params("data")
    out_dir = Path(args.out_dir)

    with Timer() as t:
        manifest = export_dataset(params["dataset"], Path(args.tfds_dir), out_dir)
    manifest.to_csv(out_dir / "manifest.csv", index=False)

    counts = manifest["label_name"].value_counts().to_dict()
    log.info("exported %d images in %.1fs: %s", len(manifest), t.seconds, counts)
    write_json(
        {
            "num_images": len(manifest),
            "class_counts": counts,
            "seconds": round(t.seconds, 2),
        },
        args.metrics,
    )


if __name__ == "__main__":
    main()
