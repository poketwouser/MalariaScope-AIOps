"""Unit tests for pushing pipeline stage metrics to the Pushgateway."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from prometheus_client import generate_latest

from malariascope import pushmetrics
from malariascope.pushmetrics import build_registry, flatten, push

PROMOTE = {
    "promoted": False,
    "reason": "accuracy gain +0.0010 < 0.005",
    "challenger": {"accuracy": 0.964, "version": 2},
    "champion_before": None,
    "champion_now": 1,
}


def test_flatten_keeps_numbers_and_bools_only():
    assert dict(flatten(PROMOTE)) == {
        "promoted": 0.0,
        "challenger_accuracy": 0.964,
        "challenger_version": 2.0,
        "champion_now": 1.0,
    }


def test_registry_has_a_gauge_per_field_and_a_timestamp():
    text = generate_latest(build_registry({"sizes": {"train": 22046}})).decode()
    assert "malariascope_pipeline_sizes_train 22046.0" in text
    assert "malariascope_pipeline_last_success_timestamp_seconds" in text


def test_push_puts_the_stage_group(tmp_path):
    received = {}

    class Gateway(BaseHTTPRequestHandler):
        def do_PUT(self):
            received["path"] = self.path
            received["body"] = self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.end_headers()

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Gateway)
    threading.Thread(target=server.handle_request, daemon=True).start()
    push("promote", PROMOTE, f"127.0.0.1:{server.server_port}")
    server.server_close()
    assert received["path"] == "/metrics/job/malariascope_pipeline/stage/promote"
    assert b"malariascope_pipeline_champion_now 1.0" in received["body"]


def test_main_without_gateway_is_a_no_op(tmp_path, monkeypatch):
    path = tmp_path / "m.json"
    path.write_text(json.dumps(PROMOTE))
    monkeypatch.delenv("PUSHGATEWAY_URL", raising=False)
    monkeypatch.setattr("sys.argv", ["pushmetrics", "promote", str(path)])
    pushmetrics.main()  # no exception, nothing pushed


def test_main_ignores_an_unreachable_gateway(tmp_path, monkeypatch):
    monkeypatch.setenv("PUSHGATEWAY_URL", "127.0.0.1:1")  # nothing listens here
    monkeypatch.setattr("sys.argv", ["pushmetrics", "eda"])
    pushmetrics.main()  # a monitoring outage must not fail the stage
