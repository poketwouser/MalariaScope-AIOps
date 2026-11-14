"""Streamlit AppTest tests (UI-xx in docs/test_plan.md), with `api` stubbed out."""

from pathlib import Path

import api
import pytest
from streamlit.testing.v1 import AppTest

FRONTEND = Path(__file__).resolve().parents[2] / "frontend"

SAMPLE = "parasitized_1.png"
INFO = {"version": "1", "test_accuracy": 0.963, "test_recall_parasitized": 0.967}


def result(label="parasitized", confidence=0.95):
    p = confidence if label == "parasitized" else 1 - confidence
    return {
        "request_id": "req-1",
        "label": label,
        "is_parasitized": label == "parasitized",
        "p_parasitized": p,
        "confidence": confidence,
        "threshold": 0.24,
        "model_version": "1",
        "latency_ms": 42.0,
    }


@pytest.fixture
def backend(monkeypatch):
    """A stub backend; tests change its fields to force each path."""

    class Stub:
        up = True
        prediction = result()
        predict_error = None
        feedback_error = None
        feedback_calls = []

    stub = Stub()

    def predict(name, data, mime):
        if stub.predict_error:
            raise api.BackendError(stub.predict_error)
        return stub.prediction

    def send_feedback(name, data, mime, res, correct_label):
        if stub.feedback_error:
            raise api.BackendError(stub.feedback_error)
        stub.feedback_calls.append((res["request_id"], correct_label))

    monkeypatch.setattr(api, "ready", lambda: stub.up)
    monkeypatch.setattr(api, "model_info", lambda: INFO)
    monkeypatch.setattr(api, "predict", predict)
    monkeypatch.setattr(api, "send_feedback", send_feedback)
    return stub


def predict_page():
    at = AppTest.from_file(str(FRONTEND / "predict.py"), default_timeout=30)
    return at.run()


def button(at, text):
    return next(b for b in at.button if text in b.label)


def analyse_sample(at):
    at.selectbox[0].select(SAMPLE).run()
    button(at, "Analyse").click().run()
    return at


# UI-01
def test_sidebar_shows_ready_and_model_version(backend):
    at = predict_page()
    assert not at.exception
    assert at.sidebar.success[0].value == "Ready"
    assert "Model version 1" in at.sidebar.caption[0].value


# UI-02
def test_backend_down_is_shown_in_the_sidebar(backend):
    backend.up = False
    at = predict_page()
    assert not at.exception
    assert at.sidebar.error[0].value == "Analysis service unavailable"


# UI-03
def test_analyse_is_disabled_until_an_image_is_chosen(backend):
    at = predict_page()
    assert button(at, "Analyse").disabled
    at.selectbox[0].select(SAMPLE).run()
    assert not button(at, "Analyse").disabled


# UI-04
def test_sample_prediction_shows_the_result(backend):
    at = analyse_sample(predict_page())
    assert not at.exception
    assert "Parasite detected" in at.error[0].value
    assert not at.warning  # confidence 0.95 is not flagged


# UI-05
def test_low_confidence_result_is_flagged(backend):
    backend.prediction = result("uninfected", confidence=0.55)
    at = analyse_sample(predict_page())
    assert "No parasite detected" in at.success[0].value
    assert "Low confidence" in at.warning[0].value


# UI-06
def test_prediction_error_is_shown_without_a_result(backend):
    backend.predict_error = "The analysis service took too long to answer."
    at = analyse_sample(predict_page())
    assert not at.exception
    assert at.error[0].value == backend.predict_error
    assert not [b for b in at.button if "Yes, correct" in b.label]


# UI-07
def test_feedback_is_sent_once_and_acknowledged(backend):
    at = analyse_sample(predict_page())
    button(at, "No, it is").click().run()
    assert backend.feedback_calls == [("req-1", "uninfected")]
    assert "feedback was saved" in at.info[0].value
    assert not [b for b in at.button if "Yes, correct" in b.label]


# UI-08
def test_feedback_error_is_shown(backend):
    backend.feedback_error = "could not store the feedback (error 500)"
    at = analyse_sample(predict_page())
    button(at, "Yes, correct").click().run()
    assert not at.exception
    assert "Feedback could not be saved" in at.error[-1].value


# UI-09
def test_pipeline_page_survives_every_service_being_down(monkeypatch):
    def down():
        raise api.BackendError("The analysis service can't be reached.")

    monkeypatch.setattr(api, "airflow_get", lambda *a, **k: None)
    monkeypatch.setattr(api, "mlflow_get", lambda *a, **k: {})
    monkeypatch.setattr(api, "mlflow_post", lambda *a, **k: {})
    monkeypatch.setattr(api, "feedback_stats", down)
    at = AppTest.from_file(str(FRONTEND / "pipeline.py"), default_timeout=30).run()
    assert not at.exception
    warnings = " ".join(w.value for w in at.warning)
    assert "Airflow can't be reached" in warnings
    assert "analysis service can't be reached" in warnings
