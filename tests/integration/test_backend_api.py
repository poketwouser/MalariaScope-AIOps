"""Backend API tests with a fake model client instead of the model server."""

import io
import os
import tempfile

import numpy as np
import pytest
from PIL import Image

# app.main opens the feedback store on import, keep it away from the real one
os.environ.setdefault("FEEDBACK_DIR", tempfile.mkdtemp())

from fastapi.testclient import TestClient  # noqa: E402

from app import config, main  # noqa: E402
from app.feedback_store import FeedbackStore  # noqa: E402
from app.model_client import ModelUnavailable  # noqa: E402

THRESHOLD = 0.3


class FakeModel:
    def __init__(self, p=0.8, up=True):
        self.p, self.up = p, up

    def info(self):
        if not self.up:
            raise ModelUnavailable("registry down")
        return {
            "name": "malariascope",
            "alias": "champion",
            "version": "7",
            "threshold": THRESHOLD,
            "image_size": 64,
            "test_accuracy": 0.96,
            "test_recall_parasitized": 0.97,
            "git_commit": "abc",
            "data_md5": "def",
            "class_names": ["parasitized", "uninfected"],
        }

    def ready(self):
        return self.up

    def p_parasitized(self, batch):
        if not self.up:
            raise ModelUnavailable("model server down")
        assert batch.shape == (1, 64, 64, 3) and batch.dtype == np.uint8
        return np.array([self.p])


def png_bytes(fmt="PNG", size=(120, 110)):
    rng = np.random.default_rng(0)
    img = Image.fromarray(rng.integers(0, 256, (*size, 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


@pytest.fixture
def make_client(tmp_path, monkeypatch):
    def make(model=None, max_upload_bytes=None):
        monkeypatch.setattr(main, "model", model or FakeModel())
        monkeypatch.setattr(main, "store", FeedbackStore(tmp_path / "feedback"))
        if max_upload_bytes is not None:
            monkeypatch.setattr(config, "MAX_UPLOAD_BYTES", max_upload_bytes)
        return TestClient(main.app)

    return make


def upload(data, name="cell.png", mime="image/png"):
    return {"file": (name, data, mime)}


# ---- health ------------------------------------------------------------------


def test_health(make_client):
    r = make_client().get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok"}


def test_ready_reflects_model_server(make_client):
    assert make_client().get("/ready").status_code == 200
    r = make_client(FakeModel(up=False)).get("/ready")
    assert r.status_code == 503 and r.json()["model_server"] == "down"


# ---- predict -----------------------------------------------------------------


def test_predict_parasitized(make_client):
    r = make_client(FakeModel(p=0.8)).post("/predict", files=upload(png_bytes()))
    assert r.status_code == 200
    body = r.json()
    assert body["label"] == "parasitized" and body["is_parasitized"]
    assert body["confidence"] == pytest.approx(0.8)
    assert body["threshold"] == THRESHOLD and body["model_version"] == "7"
    assert r.headers["X-Request-ID"] == body["request_id"]


def test_predict_uninfected(make_client):
    body = (
        make_client(FakeModel(p=0.1)).post("/predict", files=upload(png_bytes())).json()
    )
    assert body["label"] == "uninfected"
    assert body["confidence"] == pytest.approx(0.9)


def test_predict_uses_tuned_threshold_not_half(make_client):
    # 0.35 is below 0.5 but above the tuned 0.3, so it must be flagged
    body = (
        make_client(FakeModel(p=0.35))
        .post("/predict", files=upload(png_bytes()))
        .json()
    )
    assert body["label"] == "parasitized"
    assert body["confidence"] == pytest.approx(0.35)


def test_predict_accepts_jpeg(make_client):
    r = make_client().post(
        "/predict", files=upload(png_bytes("JPEG"), "c.jpg", "image/jpeg")
    )
    assert r.status_code == 200


def test_predict_rejects_non_image(make_client):
    r = make_client().post(
        "/predict", files=upload(b"hello", "notes.txt", "text/plain")
    )
    assert r.status_code == 400
    body = r.json()
    assert body["error"] == "invalid_image" and body["request_id"]


def test_predict_rejects_other_image_formats(make_client):
    r = make_client().post(
        "/predict", files=upload(png_bytes("GIF"), "c.gif", "image/gif")
    )
    assert r.status_code == 400 and "GIF" in r.json()["detail"]


def test_predict_rejects_large_file(make_client):
    r = make_client(max_upload_bytes=1000).post("/predict", files=upload(png_bytes()))
    assert r.status_code == 413 and r.json()["error"] == "file_too_large"


def test_predict_missing_file(make_client):
    r = make_client().post("/predict")
    assert r.status_code == 422 and r.json()["error"] == "validation_error"


def test_predict_model_down(make_client):
    r = make_client(FakeModel(up=False)).post("/predict", files=upload(png_bytes()))
    assert r.status_code == 503 and r.json()["error"] == "model_unavailable"


# ---- feedback ----------------------------------------------------------------


def feedback_form(correct="uninfected", predicted="parasitized"):
    return {
        "request_id": "r1",
        "predicted_label": predicted,
        "correct_label": correct,
        "p_parasitized": "0.8",
        "model_version": "7",
    }


def test_feedback_is_stored_and_counted(make_client, tmp_path):
    client = make_client()
    r = client.post("/feedback", data=feedback_form(), files=upload(png_bytes()))
    assert r.status_code == 201
    body = r.json()
    assert body["stored"] and not body["agreed"] and body["feedback_count"] == 1
    assert (tmp_path / "feedback/images/uninfected/000001.png").exists()

    client.post(
        "/feedback", data=feedback_form("parasitized"), files=upload(png_bytes())
    )
    stats = client.get("/feedback/stats").json()
    assert stats == {"total": 2, "agreed": 1, "disagreed": 1, "since_last_export": 2}


def test_feedback_rejects_unknown_label(make_client):
    r = make_client().post(
        "/feedback", data=feedback_form("sick"), files=upload(png_bytes())
    )
    assert r.status_code == 400 and r.json()["error"] == "invalid_label"


# ---- model info and metrics ----------------------------------------------------


def test_model_info(make_client):
    r = make_client().get("/model/info")
    assert r.status_code == 200
    assert r.json()["version"] == "7" and r.json()["image_size"] == 64


def test_metrics_count_predictions(make_client):
    client = make_client(FakeModel(p=0.9))
    client.post("/predict", files=upload(png_bytes()))
    text = client.get("/metrics").text
    assert 'malariascope_predictions_total{label="parasitized"}' in text
    assert "malariascope_inference_seconds_bucket" in text
    assert 'path="/predict"' in text


def test_metrics_report_serving_state_without_traffic(make_client):
    text = make_client().get("/metrics").text
    assert "malariascope_model_server_up 1.0" in text
    assert 'malariascope_model_version{alias="champion"} 7.0' in text
    assert "malariascope_feedback_pending 0.0" in text
    # a model outage still returns metrics, with the server marked down
    down = make_client(FakeModel(up=False)).get("/metrics")
    assert down.status_code == 200
    assert "malariascope_model_server_up 0.0" in down.text
