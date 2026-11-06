"""Webhook that Alertmanager posts to. Logs each alert as one line.

Same idea as Assignment 4's alert_receiver.py: the course sample sends alerts to
webhook.site, but this deployment stays on-prem, so notifications are written to
a log file instead (ALERT_LOG, default /results/alerts.log). Standard library only.
"""

import json
import os
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

LOG = Path(os.getenv("ALERT_LOG", "/results/alerts.log"))
PORT = int(os.getenv("PORT", "9095"))


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with LOG.open("a") as f:
            for a in body.get("alerts", []):
                line = (
                    f"{now}  {a['status'].upper():8} {a['labels'].get('alertname')}"
                    f"  severity={a['labels'].get('severity')}"
                    f"  {a['annotations'].get('summary', '')}"
                )
                print(line, flush=True)
                f.write(line + "\n")
        self.send_response(200)
        self.end_headers()

    def do_GET(self):  # healthcheck
        self.send_response(200 if self.path == "/health" else 404)
        self.end_headers()

    def log_message(self, *args):  # keep the console to the alert lines only
        pass


if __name__ == "__main__":
    LOG.parent.mkdir(parents=True, exist_ok=True)
    print(f"alert receiver on :{PORT}/alert, logging to {LOG}", flush=True)
    HTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
