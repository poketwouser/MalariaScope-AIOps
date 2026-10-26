"""Shared helpers for the pipeline stages: params, logging, timing and JSON output."""

import json
import logging
import time
from pathlib import Path

import yaml

PARAMS_FILE = Path("params.yaml")
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def get_logger(name):
    """Return a logger that writes timestamped lines to stderr."""
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    return logging.getLogger(name)


def load_params(section=None, path=PARAMS_FILE):
    """Read params.yaml, or one top-level section of it."""
    with open(path) as f:
        params = yaml.safe_load(f)
    if section is None:
        return params
    if section not in params:
        raise KeyError(f"section '{section}' missing from {path}")
    return params[section]


def write_json(data, path):
    """Write a dict as pretty JSON, creating parent folders."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


class Timer:
    """Context manager that measures wall-clock seconds.

    with Timer() as t:
        work()
    t.seconds
    """

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self._start
        return False
