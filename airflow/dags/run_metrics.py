# dag success/failure callback: pushes the run result so prometheus can alert on
# a failed run

import logging
import os

from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
from pendulum import now

log = logging.getLogger(__name__)


def push_run_result(context):
    gateway = os.getenv("PUSHGATEWAY_URL")
    run = context["dag_run"]
    if not gateway:
        return
    finished = run.end_date or now("UTC")
    registry = CollectorRegistry()
    gauges = {
        "malariascope_dag_run_success": (
            "1 if the latest run of the DAG succeeded, 0 if it failed",
            1 if run.get_state() == "success" else 0,
        ),
        "malariascope_dag_run_duration_seconds": (
            "Duration of the latest run of the DAG",
            (finished - run.start_date).total_seconds() if run.start_date else 0,
        ),
        "malariascope_dag_run_finished_timestamp_seconds": (
            "When the latest run of the DAG finished",
            finished.timestamp(),
        ),
    }
    for name, (help_text, value) in gauges.items():
        Gauge(name, help_text, registry=registry).set(value)
    try:
        push_to_gateway(
            gateway,
            job="malariascope_airflow",
            grouping_key={"dag_id": run.dag_id},
            registry=registry,
            timeout=10,
        )
    except Exception as e:  # monitoring must not break the scheduler callback
        log.warning("could not push run metrics for %s: %s", run.dag_id, e)
