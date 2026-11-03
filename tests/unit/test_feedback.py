"""Unit tests for the feedback export and how feedback joins the train split."""

import io

import numpy as np
import pandas as pd
from PIL import Image

from app.feedback_store import FeedbackStore
from malariascope.data.feedback import export, read_snapshot
from malariascope.data.split import add_feedback
from malariascope.image import pixel_hash


def cell(seed):
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (40, 40, 3), dtype=np.uint8))


def add(store, img, label="parasitized", predicted="uninfected"):
    store.add(img, "req", predicted, label, 0.4, "1")


# ---- pixel_hash --------------------------------------------------------------


def test_pixel_hash_ignores_file_encoding():
    img = cell(0)
    png, bmp = io.BytesIO(), io.BytesIO()
    img.save(png, format="PNG")
    img.save(bmp, format="BMP")
    assert pixel_hash(Image.open(png)) == pixel_hash(Image.open(bmp))
    assert pixel_hash(img) != pixel_hash(cell(1))


# ---- export ------------------------------------------------------------------


def test_export_copies_new_items_once(tmp_path):
    store = FeedbackStore(tmp_path / "fb")
    add(store, cell(0))
    add(store, cell(1), label="uninfected")
    out = tmp_path / "snapshot"

    assert export(tmp_path / "fb", out) == (2, 2)
    snap = read_snapshot(out)
    assert list(snap["label_name"]) == ["parasitized", "uninfected"]
    assert list(snap["label"]) == [0, 1]
    assert all((out / p).exists() for p in snap["path"])
    assert store.stats()["since_last_export"] == 0

    # nothing new: the snapshot is unchanged
    assert export(tmp_path / "fb", out) == (0, 2)
    # later feedback is appended to the cumulative snapshot
    add(store, cell(2))
    assert export(tmp_path / "fb", out) == (1, 3)


# ---- add_feedback ------------------------------------------------------------


def make_raw(tmp_path):
    """Three raw images: one per split."""
    rows = []
    for i, name in enumerate(("train", "val", "test")):
        path = f"images/{i}.png"
        (tmp_path / "images").mkdir(exist_ok=True)
        cell(10 + i).save(tmp_path / path)
        rows.append(
            {"path": path, "label": 0, "label_name": "parasitized", "split": name}
        )
    return pd.DataFrame(rows)


def feedback_rows(*images):
    return pd.DataFrame(
        {
            "path": [f"images/parasitized/fb_{i:06d}.png" for i in range(len(images))],
            "label": 0,
            "label_name": "parasitized",
            "pixel_sha1": [pixel_hash(img) for img in images],
        }
    )


def test_feedback_goes_to_train_only(tmp_path):
    split = make_raw(tmp_path)
    out, counts = add_feedback(split, feedback_rows(cell(0), cell(1)), tmp_path)
    added = out[out["source"] == "feedback"]
    assert len(added) == 2 and set(added["split"]) == {"train"}
    # raw rows keep their order and split
    assert out.iloc[:3]["path"].tolist() == split["path"].tolist()
    assert counts["added"] == 2


def test_feedback_matching_val_or_test_is_dropped(tmp_path):
    split = make_raw(tmp_path)
    # cell(11) and cell(12) are the val and test images; cell(10) is a train image
    fb = feedback_rows(cell(11), cell(12), cell(10), cell(0))
    out, counts = add_feedback(split, fb, tmp_path)
    assert counts["leaked_dropped"] == 2
    assert counts["added"] == 2


def test_duplicate_feedback_keeps_latest_label(tmp_path):
    split = make_raw(tmp_path)
    fb = feedback_rows(cell(0), cell(0))
    fb.loc[1, ["label", "label_name"]] = [1, "uninfected"]
    out, counts = add_feedback(split, fb, tmp_path)
    added = out[out["source"] == "feedback"]
    assert counts["duplicates_dropped"] == 1
    assert added["label_name"].tolist() == ["uninfected"]


def test_no_feedback_leaves_split_unchanged(tmp_path):
    split = make_raw(tmp_path)
    out, counts = add_feedback(split, pd.DataFrame(), tmp_path)
    assert out.drop(columns="source").equals(split)
    assert counts["added"] == 0
