"""Push a stage's metrics json to the Pushgateway.

a stage finishes before prometheus would scrape it, so it pushes instead.
every number in the json becomes a gauge. if the push fails we only log it,
monitoring should never fail the pipeline.
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
