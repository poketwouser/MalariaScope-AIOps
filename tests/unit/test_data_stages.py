"""Unit tests for the data stages, using tiny generated images."""

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from malariascope.data.split import make_split
from malariascope.data.validate import check_image
from malariascope.image import to_model_input

LIMITS = {"min_side": 20, "max_side": 400, "min_std": 5.0}


def save_png(path, array, mode="RGB"):
    Image.fromarray(array.astype(np.uint8), mode=mode).save(path)
    return path


def noisy(h, w, channels=3, seed=0):
    rng = np.random.default_rng(seed)
    shape = (h, w, channels) if channels > 1 else (h, w)
    return rng.integers(0, 256, size=shape)


# ---- check_image -------------------------------------------------------------


def test_check_image_accepts_normal_cell(tmp_path):
    path = save_png(tmp_path / "ok.png", noisy(120, 100))
    reason, stats = check_image(path, **LIMITS)
    assert reason == "ok"
    assert (stats["width"], stats["height"]) == (100, 120)
    assert len(stats["sha1"]) == 40


def test_check_image_rejects_blank(tmp_path):
    path = save_png(tmp_path / "blank.png", np.full((80, 80, 3), 200))
    reason, _ = check_image(path, **LIMITS)
    assert reason == "blank"


def test_check_image_rejects_tiny(tmp_path):
    path = save_png(tmp_path / "tiny.png", noisy(10, 10))
    reason, _ = check_image(path, **LIMITS)
    assert reason == "bad_size"


def test_check_image_rejects_grayscale(tmp_path):
    path = save_png(tmp_path / "gray.png", noisy(80, 80, channels=1), mode="L")
    reason, _ = check_image(path, **LIMITS)
    assert reason == "not_rgb"


def test_check_image_rejects_corrupt_file(tmp_path):
    path = tmp_path / "broken.png"
    path.write_bytes(b"not a png at all")
    reason, _ = check_image(path, **LIMITS)
    assert reason == "unreadable"


# ---- make_split --------------------------------------------------------------


@pytest.fixture
def manifest():
    labels = [0] * 500 + [1] * 500
    return pd.DataFrame(
        {"path": [f"img_{i:04d}.png" for i in range(1000)], "label": labels}
    )


FRACTIONS = {"train": 0.8, "val": 0.1, "test": 0.1}


def test_split_sizes_and_no_overlap(manifest):
    split = make_split(manifest, FRACTIONS, seed=42)
    sizes = split["split"].value_counts().to_dict()
    assert sizes == {"train": 800, "val": 100, "test": 100}
    assert split["path"].is_unique


def test_split_is_stratified(manifest):
    split = make_split(manifest, FRACTIONS, seed=42)
    for _, part in split.groupby("split"):
        assert part["label"].mean() == pytest.approx(0.5, abs=0.02)


def test_split_is_reproducible(manifest):
    a = make_split(manifest, FRACTIONS, seed=42)
    b = make_split(manifest, FRACTIONS, seed=42)
    pd.testing.assert_frame_equal(a, b)


def test_split_rejects_bad_fractions(manifest):
    with pytest.raises(ValueError):
        make_split(manifest, {"train": 0.8, "val": 0.2, "test": 0.2}, seed=42)


# ---- to_model_input ----------------------------------------------------------


def test_to_model_input_shape_and_dtype():
    img = Image.fromarray(noisy(130, 90).astype(np.uint8))
    x = to_model_input(img, 64)
    assert x.shape == (64, 64, 3)
    assert x.dtype == np.uint8


def test_to_model_input_converts_rgba_to_rgb():
    img = Image.fromarray(noisy(50, 50, channels=4).astype(np.uint8), mode="RGBA")
    assert to_model_input(img, 32).shape == (32, 32, 3)
