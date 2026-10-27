"""Pipeline page: one place to see the ML pipeline and jump to each tool's UI."""

import os
from datetime import datetime

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
    "Grafana (live monitoring)": os.getenv("GRAFANA_URL", "http://localhost:3000"),
    "Prometheus (raw metrics)": os.getenv("PROMETHEUS_URL", "http://localhost:9090"),
}


def when(ms):
    return datetime.fromtimestamp(int(ms) / 1000).strftime("%Y-%m-%d %H:%M")


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
champion_section()
runs_section()
feedback_section()
