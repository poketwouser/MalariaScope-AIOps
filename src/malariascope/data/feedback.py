"""Export user feedback into the DVC-tracked training snapshot (LLD section 6).

Copies every feedback item not yet exported from the backend's feedback store
into data/feedback/, appends it to data/feedback/manifest.csv and marks it
exported. The snapshot is cumulative: it holds all feedback ever exported, so
`dvc add data/feedback` versions exactly what each retraining run saw.

Output:
    data/feedback/images/<label>/fb_<id>.png
    data/feedback/manifest.csv     path, label, label_name, pixel_sha1, ...
    results/feedback_export.json   items exported now and in the snapshot

Run: PYTHONPATH=src python -m malariascope.data.feedback
"""

import argparse
import shutil
import sqlite3
from pathlib import Path

import pandas as pd
from PIL import Image

from malariascope.common import get_logger, write_json
from malariascope.image import pixel_hash
from malariascope.labels import CLASS_NAMES

log = get_logger("feedback")

COLUMNS = [
    "path",
    "label",
    "label_name",
    "pixel_sha1",
    "feedback_id",
    "predicted_label",
    "model_version",
    "ts",
]


def read_snapshot(out_dir):
    path = Path(out_dir) / "manifest.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame(columns=COLUMNS)


def export(feedback_dir, out_dir):
    """Copy unexported feedback into out_dir; return (n_new, n_in_snapshot)."""
    feedback_dir, out_dir = Path(feedback_dir), Path(out_dir)
    snapshot = read_snapshot(out_dir)
    # the backend writes to the same SQLite file; wait for its lock, never fail on it
    conn = sqlite3.connect(feedback_dir / "feedback.db", timeout=30)
    try:
        rows = conn.execute(
            "SELECT id, ts, predicted_label, correct_label, model_version, image_path"
            " FROM feedback WHERE exported = 0 ORDER BY id"
        ).fetchall()
        new = []
        for fid, ts, predicted, correct, version, image_path in rows:
            if correct not in CLASS_NAMES:
                log.warning("feedback %d has unknown label %r, skipped", fid, correct)
                continue
            rel = Path("images") / correct / f"fb_{fid:06d}.png"
            (out_dir / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(feedback_dir / image_path, out_dir / rel)
            with Image.open(out_dir / rel) as img:
                digest = pixel_hash(img)
            new.append(
                {
                    "path": str(rel),
                    "label": CLASS_NAMES.index(correct),
                    "label_name": correct,
                    "pixel_sha1": digest,
                    "feedback_id": fid,
                    "predicted_label": predicted,
                    "model_version": version,
                    "ts": ts,
                }
            )
        if new and snapshot.empty:
            snapshot = pd.DataFrame(new, columns=COLUMNS)
        elif new:
            snapshot = pd.concat([snapshot, pd.DataFrame(new)], ignore_index=True)
        out_dir.mkdir(parents=True, exist_ok=True)
        snapshot.to_csv(out_dir / "manifest.csv", index=False)
        # mark as exported only after the snapshot is written
        conn.executemany(
            "UPDATE feedback SET exported = 1 WHERE id = ?",
            [(r[0],) for r in rows],
        )
        conn.commit()
    finally:
        conn.close()
    return len(new), len(snapshot)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--feedback-dir", default="feedback")
    parser.add_argument("--out-dir", default="data/feedback")
    parser.add_argument("--metrics", default="results/feedback_export.json")
    args = parser.parse_args()

    n_new, n_total = export(args.feedback_dir, args.out_dir)
    log.info("exported %d new feedback items; snapshot now holds %d", n_new, n_total)
    write_json({"exported": n_new, "snapshot_size": n_total}, args.metrics)


if __name__ == "__main__":
    main()
