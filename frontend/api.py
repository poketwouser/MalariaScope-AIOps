"""HTTP clients for the backend and the MLflow REST API.

The UI talks to the backend only through BACKEND_URL (loose coupling, rubric:
"independent software blocks connected only via configurable REST API calls").
"""

import os

import requests

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")
MLFLOW_URL = os.getenv("MLFLOW_URL", "http://localhost:5000").rstrip("/")
AIRFLOW_API_URL = os.getenv("AIRFLOW_API_URL", "http://localhost:8080").rstrip("/")
AIRFLOW_AUTH = (
    os.getenv("AIRFLOW_USER", "admin"),
    os.getenv("AIRFLOW_PASSWORD", "admin"),
)
TIMEOUT = float(os.getenv("BACKEND_TIMEOUT", "30"))


class BackendError(Exception):
    """A request to the backend failed; message is safe to show to users."""


def _call(method, path, **kwargs):
    try:
        r = requests.request(method, f"{BACKEND_URL}{path}", timeout=TIMEOUT, **kwargs)
    except requests.ConnectionError:
        raise BackendError("The analysis service can't be reached. Is it running?")
    except requests.Timeout:
        raise BackendError("The analysis service took too long to answer.")
    if r.ok:
        return r.json()
    try:
        detail = r.json().get("detail", r.text)
    except ValueError:
        detail = r.text
    raise BackendError(f"{detail} (error {r.status_code})")


def ready():
    try:
        return requests.get(f"{BACKEND_URL}/ready", timeout=3).status_code == 200
    except requests.RequestException:
        return False


def model_info():
    return _call("GET", "/model/info")


def predict(name, data, mime):
    return _call("POST", "/predict", files={"file": (name, data, mime)})


def send_feedback(name, data, mime, result, correct_label):
    form = {
        "request_id": result["request_id"],
        "predicted_label": result["label"],
        "correct_label": correct_label,
        "p_parasitized": str(result["p_parasitized"]),
        "model_version": result["model_version"],
    }
    return _call("POST", "/feedback", data=form, files={"file": (name, data, mime)})


def feedback_stats():
    return _call("GET", "/feedback/stats")


def mlflow_get(path, **params):
    """GET on the MLflow REST API; returns {} if MLflow is unreachable."""
    try:
        r = requests.get(
            f"{MLFLOW_URL}/api/2.0/mlflow/{path}", params=params, timeout=5
        )
        return r.json() if r.ok else {}
    except requests.RequestException:
        return {}


def mlflow_post(path, body):
    try:
        r = requests.post(f"{MLFLOW_URL}/api/2.0/mlflow/{path}", json=body, timeout=5)
        return r.json() if r.ok else {}
    except requests.RequestException:
        return {}


def airflow_get(path, **params):
    """GET on the Airflow REST API (v1); returns None if Airflow is unreachable."""
    try:
        r = requests.get(
            f"{AIRFLOW_API_URL}/api/v1/{path}",
            params=params,
            auth=AIRFLOW_AUTH,
            timeout=5,
        )
        return r.json() if r.ok else None
    except requests.RequestException:
        return None


def airflow_trigger(dag_id, conf=None):
    """Start a DAG run; returns the run id, or raises BackendError."""
    try:
        r = requests.post(
            f"{AIRFLOW_API_URL}/api/v1/dags/{dag_id}/dagRuns",
            json={"conf": conf or {}},
            auth=AIRFLOW_AUTH,
            timeout=10,
        )
    except requests.RequestException:
        raise BackendError("Airflow can't be reached. Is it running?")
    if not r.ok:
        raise BackendError(f"Airflow refused the run (error {r.status_code})")
    return r.json()["dag_run_id"]
