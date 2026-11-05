"""End-to-end tests against the running Docker Compose stack.

These exercise the real containers (images, env vars, network, mounted stores),
which the unit and integration tests can't: the Day 7 missing-pandas bug in the
backend image passed every other test. Test IDs match docs/test_plan.md.

Start the stack first (`docker compose up -d`), then:
    pytest tests/e2e -m e2e
Every test is skipped when the backend isn't reachable. The tests never store
feedback, so they don't change the retraining data.
"""

import os
import time
from pathlib import Path

import numpy as np
import pytest
import requests
from PIL import Image

from malariascope.image import to_model_input

pytestmark = pytest.mark.e2e

HOST = os.getenv("E2E_HOST", "http://127.0.0.1")
BACKEND = f"{HOST}:8000"
MLFLOW = f"{HOST}:5000"
MODEL_SERVER = f"{HOST}:5001"
PROMETHEUS = f"{HOST}:9090"
SAMPLES = sorted(
    (Path(__file__).resolve().parents[2] / "frontend" / "samples").glob("*.png")
)
LATENCY_REQUESTS = 50
P95_LIMIT_MS = 500  # same limit as the PredictLatencyHigh alert


def up(url):
    try:
        return requests.get(url, timeout=3).ok
    except requests.RequestException:
        return False


if not up(f"{BACKEND}/health"):
    pytest.skip("compose stack is not running", allow_module_level=True)


def predict(path):
    return requests.post(
        f"{BACKEND}/predict",
        files={"file": (path.name, path.read_bytes(), "image/png")},
        timeout=30,
    )


def prom_query(expr):
    r = requests.get(f"{PROMETHEUS}/api/v1/query", params={"query": expr}, timeout=5)
    r.raise_for_status()
    return r.json()["data"]["result"]


# E2E-01
@pytest.mark.parametrize(
    "name, url",
    [
        ("mlflow", f"{MLFLOW}/health"),
        ("model-server", f"{MODEL_SERVER}/ping"),
        ("backend", f"{BACKEND}/ready"),
        ("frontend", f"{HOST}:8501/_stcore/health"),
        ("airflow", f"{HOST}:8080/health"),
        ("prometheus", f"{PROMETHEUS}/-/ready"),
        ("pushgateway", f"{HOST}:9091/-/ready"),
        ("alertmanager", f"{HOST}:9093/-/ready"),
        ("grafana", f"{HOST}:3000/api/health"),
    ],
)
def test_service_is_healthy(name, url):
    assert up(url), f"{name} not healthy at {url}"


# E2E-02
def test_backend_serves_the_registry_champion():
    info = requests.get(f"{BACKEND}/model/info", timeout=10)
    assert info.status_code == 200, info.text
    info = info.json()
    alias = requests.get(
        f"{MLFLOW}/api/2.0/mlflow/registered-models/alias",
        params={"name": "malariascope", "alias": "champion"},
        timeout=5,
    ).json()["model_version"]
    assert info["version"] == alias["version"]
    tags = {t["key"]: t["value"] for t in alias.get("tags", [])}
    assert info["threshold"] == pytest.approx(float(tags["decision_threshold"]))
    assert info["git_commit"] == tags["git_commit"]
    assert info["image_size"] == 64


# E2E-03
def test_model_server_returns_two_class_probabilities():
    x = to_model_input(Image.open(SAMPLES[0]), 64)[None]
    r = requests.post(
        f"{MODEL_SERVER}/invocations", json={"instances": x.tolist()}, timeout=30
    )
    assert r.status_code == 200, r.text
    p = np.asarray(r.json()["predictions"])
    assert p.shape == (1, 2)
    assert p.sum() == pytest.approx(1.0, abs=1e-4)


# E2E-04
@pytest.mark.parametrize("path", SAMPLES, ids=[p.stem for p in SAMPLES])
def test_predict_labels_the_sample_cells_correctly(path):
    r = predict(path)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["label"] == path.stem.rsplit("_", 1)[0]
    assert body["is_parasitized"] == (body["p_parasitized"] >= body["threshold"])
    assert r.headers["X-Request-ID"] == body["request_id"]


# E2E-05
@pytest.mark.parametrize(
    "files, status, error",
    [
        ({"file": ("notes.txt", b"not an image", "text/plain")}, 400, "invalid_image"),
        ({"file": ("empty.png", b"", "image/png")}, 400, "invalid_image"),
        (
            {"file": ("big.png", b"\0" * (5 * 1024 * 1024 + 1), "image/png")},
            413,
            "file_too_large",
        ),
        ({}, 422, "validation_error"),
    ],
    ids=["non-image", "empty", "too-large", "no-file"],
)
def test_predict_rejects_bad_uploads(files, status, error):
    r = requests.post(f"{BACKEND}/predict", files=files or None, timeout=30)
    assert r.status_code == status
    assert r.json()["error"] == error


# E2E-06: the label check runs before anything is stored
def test_feedback_with_an_unknown_label_is_rejected_and_not_stored():
    before = requests.get(f"{BACKEND}/feedback/stats", timeout=5).json()["total"]
    r = requests.post(
        f"{BACKEND}/feedback",
        data={
            "request_id": "e2e",
            "predicted_label": "parasitized",
            "correct_label": "sick",
            "p_parasitized": "0.9",
            "model_version": "1",
        },
        files={"file": (SAMPLES[0].name, SAMPLES[0].read_bytes(), "image/png")},
        timeout=10,
    )
    assert r.status_code == 400
    assert r.json()["error"] == "invalid_label"
    after = requests.get(f"{BACKEND}/feedback/stats", timeout=5).json()["total"]
    assert after == before


# E2E-07
def test_prometheus_scrapes_the_backend_predictions():
    targets = requests.get(f"{PROMETHEUS}/api/v1/targets", timeout=5).json()
    health = {t["labels"]["job"]: t["health"] for t in targets["data"]["activeTargets"]}
    assert health.get("backend") == "up", health

    def count():
        res = prom_query("sum(malariascope_predictions_total)")
        return float(res[0]["value"][1]) if res else 0.0

    before = count()
    assert predict(SAMPLES[0]).status_code == 200
    deadline = time.time() + 20  # scrape interval is 5 s
    while count() <= before and time.time() < deadline:
        time.sleep(1)
    assert count() > before


# E2E-08
def test_alert_rules_and_grafana_dashboards_are_provisioned():
    rules = requests.get(f"{PROMETHEUS}/api/v1/rules", timeout=5).json()
    alerts = {
        r["name"]
        for g in rules["data"]["groups"]
        for r in g["rules"]
        if r["type"] == "alerting"
    }
    assert {"BackendDown", "ModelServerDown", "InputDrift"} <= alerts
    dashboards = requests.get(
        f"{HOST}:3000/api/search", params={"type": "dash-db"}, timeout=5
    ).json()
    uids = {d["uid"] for d in dashboards}
    assert {"malariascope-service", "malariascope-pipeline"} <= uids


# E2E-09
def test_airflow_dags_load_without_import_errors():
    auth = ("admin", "admin")
    errors = requests.get(f"{HOST}:8080/api/v1/importErrors", auth=auth, timeout=5)
    assert errors.json()["total_entries"] == 0
    dags = requests.get(f"{HOST}:8080/api/v1/dags", auth=auth, timeout=5).json()
    assert {"training_pipeline", "feedback_retrain"} <= {
        d["dag_id"] for d in dags["dags"]
    }


# E2E-10: acceptance criterion AC-3
def test_predict_p95_latency_is_under_the_limit():
    times = []
    for i in range(LATENCY_REQUESTS):
        start = time.perf_counter()
        assert predict(SAMPLES[i % len(SAMPLES)]).status_code == 200
        times.append((time.perf_counter() - start) * 1000)
    p50, p95 = np.percentile(times, [50, 95])
    print(
        f"\n/predict over {LATENCY_REQUESTS} calls: p50 {p50:.1f} ms, p95 {p95:.1f} ms"
    )
    assert p95 < P95_LIMIT_MS
