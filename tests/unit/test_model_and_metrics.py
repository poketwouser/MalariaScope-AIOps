"""Unit tests for the model contract, the evaluation metrics and threshold tuning."""

import numpy as np
import pytest

from malariascope.evaluate import compute_metrics, pick_threshold, predict_labels
from malariascope.labels import CLASS_NAMES
from malariascope.model import build_model

PARASITIZED = CLASS_NAMES.index("parasitized")
UNINFECTED = CLASS_NAMES.index("uninfected")


def p_parasitized(pred, confidence=0.9):
    """p(parasitized) for a list of predicted labels at a fixed confidence."""
    pred = np.asarray(pred)
    return np.where(pred == PARASITIZED, confidence, 1 - confidence)


def test_perfect_predictions():
    y = np.array([PARASITIZED, PARASITIZED, UNINFECTED, UNINFECTED])
    m = compute_metrics(y, p_parasitized(y))
    assert m["accuracy"] == 1.0
    assert m["recall_parasitized"] == 1.0
    assert m["false_negatives"] == 0
    assert m["roc_auc"] == 1.0


def test_missed_infection_counts_as_false_negative():
    y = np.array([PARASITIZED, PARASITIZED, UNINFECTED, UNINFECTED])
    pred = [UNINFECTED, PARASITIZED, UNINFECTED, UNINFECTED]
    m = compute_metrics(y, p_parasitized(pred))
    assert m["false_negatives"] == 1
    assert m["false_positives"] == 0
    assert m["recall_parasitized"] == pytest.approx(0.5)
    assert m["precision_parasitized"] == 1.0
    assert m["specificity"] == 1.0


def test_lower_threshold_flags_more_cells_as_parasitized():
    p = np.array([0.2, 0.4, 0.6, 0.8])
    assert (predict_labels(p, 0.5) == PARASITIZED).sum() == 2
    assert (predict_labels(p, 0.3) == PARASITIZED).sum() == 3


def test_pick_threshold_reaches_target_recall():
    rng = np.random.default_rng(0)
    y = np.array([PARASITIZED] * 100 + [UNINFECTED] * 100)
    p = np.where(y == PARASITIZED, rng.uniform(0.3, 1.0, 200), rng.uniform(0, 0.7, 200))
    t = pick_threshold(y, p, target_recall=0.97)
    recall = compute_metrics(y, p, t)["recall_parasitized"]
    assert recall >= 0.97
    # the highest such threshold: raising it any further drops below the target
    assert compute_metrics(y, p, t + 1e-6)["recall_parasitized"] < 0.97


def test_model_input_output_contract():
    model = build_model(32, [8], 8, 0.1, augment=True, learning_rate=1e-3)
    x = np.random.default_rng(0).integers(0, 256, (3, 32, 32, 3), dtype=np.uint8)
    probs = model.predict(x, verbose=0)
    assert probs.shape == (3, len(CLASS_NAMES))
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=1e-5)
