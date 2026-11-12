"""Pipeline page: pipeline runs, models, feedback and links to each tool."""

import json
import os
from datetime import datetime, timezone
from urllib.parse import quote

import pandas as pd
import streamlit as st

import api

MODEL = "malariascope"
EXPERIMENT = "malariascope"
LINKS = {
    "Airflow (pipeline runs)": os.getenv("AIRFLOW_URL", "http://localhost:8080"),
    "MLflow (experiments and models)": os.getenv(
        "MLFLOW_PUBLIC_URL", "http://localhost:5000"
    ),
    "Grafana (live monitoring)": os.getenv("GRAFANA_URL", "http://localhost:3000")
    + "/d/malariascope-service",
    "Prometheus (raw metrics)": os.getenv("PROMETHEUS_URL", "http://localhost:9090"),
    "Alertmanager (alerts)": os.getenv("ALERTMANAGER_URL", "http://localhost:9093"),
}


DAGS = {
    "training_pipeline": "Training pipeline (DVC stages)",
    "feedback_retrain": "Feedback retraining (checks every 30 min)",
}
STATE_ICON = {
    "success": "✅ success",
    "failed": "❌ failed",
    "running": "⏳ running",
    "queued": "🕓 queued",
    "skipped": "⏭️ skipped",
    "upstream_failed": "⛔ upstream failed",
}


def when(ms):
    return datetime.fromtimestamp(int(ms) / 1000).strftime("%Y-%m-%d %H:%M")


def parse_iso(ts):
    return datetime.fromisoformat(ts) if ts else None


def seconds(start, end):
    start, end = parse_iso(start), parse_iso(end)
    if start is None:
        return None
    return round(((end or datetime.now(timezone.utc)) - start).total_seconds(), 1)


def local_time(ts):
    t = parse_iso(ts)
    return t.astimezone().strftime("%Y-%m-%d %H:%M") if t else ""


def xcom_json(dag_id, run_id, task_id):
    # each training task prints its metrics json last, airflow keeps it as the xcom
    path = (
        f"dags/{dag_id}/dagRuns/{quote(run_id, safe='')}/taskInstances/"
        f"{task_id}/xcomEntries/return_value"
    )
    entry = api.airflow_get(path) or {}
    try:
        return json.loads(entry.get("value") or "{}")
    except ValueError:
        return {}


def dag_runs(dag_id):
    st.markdown(f"**{DAGS[dag_id]}**")
    found = api.airflow_get(
        f"dags/{dag_id}/dagRuns", order_by="-execution_date", limit=5
    )
    failed = api.airflow_get(f"dags/{dag_id}/dagRuns", state="failed", limit=1)
    runs = (found or {}).get("dag_runs", [])
    left, right = st.columns([3, 1])
    left.caption(
        f"{(found or {}).get('total_entries', 0)} runs in total, "
        f"{(failed or {}).get('total_entries', 0)} failed"
    )
    if right.button("Run now", key=f"run_{dag_id}", width="stretch"):
        try:
            run_id = api.airflow_trigger(dag_id)
            st.success(f"Started {run_id}")
        except api.BackendError as e:
            st.error(str(e))
    if runs:
        rows = [
            {
                "started": local_time(r["start_date"]),
                "state": STATE_ICON.get(r["state"], r["state"]),
                "trigger": r["run_type"],
                "duration (s)": seconds(r["start_date"], r["end_date"]),
            }
            for r in runs
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    return runs


def airflow_section():
    st.subheader("Pipeline runs (Airflow)")
    if api.airflow_get("health") is None:
        st.warning("Airflow can't be reached, so pipeline runs are not shown.")
        return
    training_runs = dag_runs("training_pipeline")
    dag_runs("feedback_retrain")
    if not training_runs:
        return

    run = training_runs[0]
    st.markdown(
        f"**Stages of the latest training run** ({local_time(run['start_date'])})"
    )
    tasks = api.airflow_get(
        f"dags/training_pipeline/dagRuns/{quote(run['dag_run_id'], safe='')}"
        "/taskInstances"
    )
    rows = [
        {
            "stage": t["task_id"],
            "state": STATE_ICON.get(t["state"], t["state"] or "not started"),
            "duration (s)": round(t["duration"], 1) if t["duration"] else None,
        }
        for t in sorted(
            (tasks or {}).get("task_instances", []),
            key=lambda t: t["start_date"] or "9999",
        )
    ]
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    cols = st.columns(2)
    for col, stage, label in (
        (cols[0], "validate", "Ray validation"),
        (cols[1], "preprocess", "Ray preprocessing"),
    ):
        rate = xcom_json("training_pipeline", run["dag_run_id"], stage).get(
            "images_per_sec"
        )
        col.metric(f"{label} throughput", f"{rate:,.0f} images/s" if rate else "–")


def champion_section():
    st.subheader("Model in use")
    found = api.mlflow_get("registered-models/alias", name=MODEL, alias="champion")
    version = found.get("model_version")
    if not version:
        st.info("No champion model registered yet.")
        return
    tags = {t["key"]: t["value"] for t in version.get("tags", [])}
    cols = st.columns(4)
    cols[0].metric("Version", version["version"])
    cols[1].metric("Test accuracy", f"{float(tags.get('test_accuracy', 0)):.1%}")
    cols[2].metric(
        "Infected cells caught", f"{float(tags.get('test_recall_parasitized', 0)):.1%}"
    )
    cols[3].metric(
        "Decision threshold", f"{float(tags.get('decision_threshold', 0)):.3f}"
    )
    st.caption(
        f"Registered {when(version['creation_timestamp'])} · "
        f"code {tags.get('git_commit', '?')[:8]} · data {tags.get('data_md5', '?')[:8]}"
    )

    versions = api.mlflow_get("model-versions/search", filter=f"name='{MODEL}'").get(
        "model_versions", []
    )
    if versions:
        rows = []
        for v in versions:
            t = {x["key"]: x["value"] for x in v.get("tags", [])}
            rows.append(
                {
                    "version": int(v["version"]),
                    "aliases": ", ".join(v.get("aliases", [])),
                    "test accuracy": float(t.get("test_accuracy", "nan")),
                    "recall (parasitized)": float(
                        t.get("test_recall_parasitized", "nan")
                    ),
                    "registered": when(v["creation_timestamp"]),
                }
            )
        st.dataframe(
            pd.DataFrame(rows).sort_values("version", ascending=False),
            hide_index=True,
            width="stretch",
        )


def runs_section():
    st.subheader("Recent training runs")
    exp = api.mlflow_get("experiments/get-by-name", experiment_name=EXPERIMENT)
    exp_id = exp.get("experiment", {}).get("experiment_id")
    if not exp_id:
        st.info("MLflow is not reachable, or no runs yet.")
        return
    runs = api.mlflow_post(
        "runs/search",
        {
            "experiment_ids": [exp_id],
            "max_results": 10,
            "order_by": ["attributes.start_time DESC"],
        },
    ).get("runs", [])
    rows = []
    for r in runs:
        m = {x["key"]: x["value"] for x in r["data"].get("metrics", [])}
        t = {x["key"]: x["value"] for x in r["data"].get("tags", [])}
        info = r["info"]
        rows.append(
            {
                "started": when(info["start_time"]),
                "status": info["status"],
                "experiment": t.get("dvc_exp", ""),
                "test accuracy": m.get("test_tuned_accuracy"),
                "recall (parasitized)": m.get("test_tuned_recall_parasitized"),
                "train time (s)": m.get("best_train_seconds"),
                "run": info["run_id"][:8],
            }
        )
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def feedback_section():
    st.subheader("User feedback")
    try:
        s = api.feedback_stats()
    except api.BackendError as e:
        st.warning(str(e))
        return
    cols = st.columns(3)
    cols[0].metric("Feedback received", s["total"])
    cols[1].metric("Model was wrong", s["disagreed"])
    cols[2].metric("Waiting for retraining", s["since_last_export"])


st.title("⚙️ Pipeline")
st.markdown(
    "Data ingestion (Ray) → validation → split → preprocessing (Ray) → training → "
    "evaluation → promotion, versioned with DVC and tracked in MLflow."
)
st.markdown(" · ".join(f"[{name}]({url})" for name, url in LINKS.items()))
airflow_section()
champion_section()
runs_section()
feedback_section()
