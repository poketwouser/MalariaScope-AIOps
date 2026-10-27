"""Unit tests for the promotion gate."""

from malariascope.promote import decide


def metrics(accuracy, recall):
    return {"accuracy": accuracy, "recall_parasitized": recall}


def test_first_model_is_promoted_when_recall_is_fine():
    promote, reason = decide(metrics(0.96, 0.97), None, min_gain=0.005, min_recall=0.95)
    assert promote
    assert "no champion" in reason


def test_low_recall_is_never_promoted():
    promote, _ = decide(metrics(0.99, 0.90), None, min_gain=0.005, min_recall=0.95)
    assert not promote


def test_promoted_only_with_enough_gain():
    champion = metrics(0.960, 0.97)
    assert decide(metrics(0.966, 0.97), champion, 0.005, 0.95)[0]
    assert not decide(metrics(0.962, 0.97), champion, 0.005, 0.95)[0]


def test_worse_challenger_is_rejected():
    promote, reason = decide(metrics(0.950, 0.97), metrics(0.960, 0.97), 0.005, 0.95)
    assert not promote
    assert "-0.0100" in reason
