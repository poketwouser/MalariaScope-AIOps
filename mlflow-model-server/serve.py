# `mlflow models serve` only looks up the alias when it starts, so after a promotion
# it would keep serving the old version. this script checks the alias every
# POLL_SECONDS and restarts the server when it moved.

import logging
import os
import signal
import subprocess
import sys
import time

from mlflow import MlflowClient

NAME = os.getenv("MODEL_NAME", "malariascope")
ALIAS = os.getenv("MODEL_ALIAS", "champion")
PORT = os.getenv("PORT", "5001")
POLL = float(os.getenv("POLL_SECONDS", "15"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("serve")


def alias_version(client):
    return client.get_model_version_by_alias(NAME, ALIAS).version


def start(version):
    log.info("serving %s version %s (@%s) on port %s", NAME, version, ALIAS, PORT)
    cmd = ["mlflow", "models", "serve", "-m", f"models:/{NAME}/{version}"]
    cmd += ["-h", "0.0.0.0", "-p", PORT, "--env-manager", "local"]
    # own process group, so stopping it also stops the uvicorn workers it spawns
    return subprocess.Popen(cmd, start_new_session=True)


def stop(proc):
    if proc.poll() is not None:
        return
    os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()


def main():
    client = MlflowClient()
    while True:
        try:
            version = alias_version(client)
            break
        except Exception as e:  # registry not up yet, or no champion registered
            log.warning("waiting for %s@%s: %s", NAME, ALIAS, e)
            time.sleep(POLL)

    proc = start(version)

    def shutdown(signum, frame):
        stop(proc)
        sys.exit(0)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    while True:
        time.sleep(POLL)
        if proc.poll() is not None:
            log.error("model server exited with code %s", proc.returncode)
            sys.exit(proc.returncode or 1)  # the compose restart policy takes over
        try:
            latest = alias_version(client)
        except Exception as e:
            log.warning("registry check failed, still serving v%s: %s", version, e)
            continue
        if latest != version:
            log.info("@%s moved from v%s to v%s, restarting", ALIAS, version, latest)
            stop(proc)
            version = latest
            proc = start(version)


if __name__ == "__main__":
    main()
