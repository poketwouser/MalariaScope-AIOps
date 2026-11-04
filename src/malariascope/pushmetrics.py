"""Push a pipeline stage's metrics to the Prometheus Pushgateway.

Prometheus pulls metrics (Lecture 9), which suits the long-running backend. A
pipeline stage is a batch job that exits before a scrape could reach it, so the
Airflow task pushes the stage's metrics JSON to the Pushgateway instead, and
Prometheus scrapes them from there. Every numeric field becomes a gauge, e.g.
results/preprocess.json {"images_per_sec": 5502.4} becomes the gauge
malariascope_pipeline_images_per_sec{stage="preprocess"}.

Monitoring must never fail the pipeline: without PUSHGATEWAY_URL this is a
no-op, and a failed push is logged and ignored.

Run: PYTHONPATH=src python -m malariascope.pushmetrics STAGE [METRICS_JSON]
"""

import argparse
import json
import os
import re
from pathlib import Path

from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

from malariascope.common import get_logger

log = get_logger("pushmetrics")

JOB = "malariascope_pipeline"
PREFIX = "malariascope_pipeline_"


def flatten(metrics, prefix=""):
    """Yield (name, value) for every number or bool in a nested dict."""
    for key, value in metrics.items():
        name = re.sub(r"[^a-zA-Z0-9_]", "_", f"{prefix}{key}")
        if isinstance(value, dict):
            yield from flatten(value, name + "_")
        elif isinstance(value, (bool, int, float)):
            yield name, float(value)


def build_registry(metrics):
    registry = CollectorRegistry()
    for name, value in flatten(metrics):
        # the help text depends only on the name: the Pushgateway rejects the same
        # metric pushed by two stages with different help strings
        Gauge(PREFIX + name, f"Pipeline metric {name}", registry=registry).set(value)
    Gauge(
        PREFIX + "last_success_timestamp_seconds",
        "When the stage last finished successfully",
        registry=registry,
    ).set_to_current_time()
    return registry


def push(stage, metrics, gateway):
    # PUT replaces the stage's whole group, so no stale fields from an older run stay
    push_to_gateway(
        gateway,
        job=JOB,
        grouping_key={"stage": stage},
        registry=build_registry(metrics),
        timeout=10,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage")
    parser.add_argument("metrics", nargs="?", help="stage metrics JSON (optional)")
    args = parser.parse_args()

    gateway = os.getenv("PUSHGATEWAY_URL")
    if not gateway:
        log.info("PUSHGATEWAY_URL not set, not pushing %s metrics", args.stage)
        return
    metrics = json.loads(Path(args.metrics).read_text()) if args.metrics else {}
    try:
        push(args.stage, metrics, gateway)
        log.info("pushed %s metrics to %s", args.stage, gateway)
    except Exception as e:
        log.warning("could not push %s metrics to %s: %s", args.stage, gateway, e)


if __name__ == "__main__":
    main()
