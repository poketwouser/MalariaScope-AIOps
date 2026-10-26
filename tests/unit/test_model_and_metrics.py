"""Unit tests for the model contract and the evaluation metrics."""

import numpy as np
import pytest

from malariascope.evaluate import compute_metrics
from malariascope.model import CLASS_NAMES, build_model

PARASITIZED, UNINFECTED = CLASS_NAMES.index("parasitized"), CLASS_NAMES.index(
    "uninfected"
)


def one_hot_probs(pred, confidence=0.9):
    probs = np.full((len(pred), 2), 1 - confidence)
    probs[np.arange(len(pred)), pred] = confidence
    return probs


def test_perfect_predictions():
    y = np.array([PARASITIZED, PARASITIZED, UNINFECTED, UNINFECTED])
    m = compute_metrics(y, one_hot_probs(y))
    assert m["accuracy"] == 1.0
    assert m["recall_parasitized"] == 1.0
    assert m["false_negatives"] == 0
    assert m["roc_auc"] == 1.0


def test_missed_infection_counts_as_false_negative():
    y = np.array([PARASITIZED, PARASITIZED, UNINFECTED, UNINFECTED])
    pred = np.array([UNINFECTED, PARASITIZED, UNINFECTED, UNINFECTED])
    m = compute_metrics(y, one_hot_probs(pred))
    assert m["false_negatives"] == 1
    assert m["false_positives"] == 0
    assert m["recall_parasitized"] == pytest.approx(0.5)
    assert m["precision_parasitized"] == 1.0
    assert m["specificity"] == 1.0


def test_model_input_output_contract():
    model = build_model(32, [8], 8, 0.1, augment=True, learning_rate=1e-3)
    x = np.random.default_rng(0).integers(0, 256, (3, 32, 32, 3), dtype=np.uint8)
    probs = model.predict(x, verbose=0)
    assert probs.shape == (3, len(CLASS_NAMES))
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=1e-5)
