"""Talks to the MLflow model server (predictions) and the MLflow registry (model info).

The backend never loads the model itself: the MLflow model server serves
models:/malariascope@champion over REST (Lecture 10), so the backend image stays
free of TensorFlow and a promotion only needs the model server restarted.
"""

import threading
import time

import mlflow
import numpy as np
import requests
from mlflow import MlflowClient

from malariascope.labels import CLASS_NAMES, POSITIVE


class ModelUnavailable(Exception):
    """The model server or the registry could not answer."""


class ModelClient:
    def __init__(self, settings, session=None):
        self.settings = settings
        self.session = session or requests.Session()
        self._info = None
        self._info_time = 0.0
        self._lock = threading.Lock()

    # ---- registry ------------------------------------------------------------

    def _fetch_info(self):
        s = self.settings
        client = MlflowClient(tracking_uri=s.tracking_uri)
        version = client.get_model_version_by_alias(s.model_name, s.model_alias)
        mlflow.set_tracking_uri(s.tracking_uri)
        signature = mlflow.models.get_model_info(
            f"models:/{s.model_name}/{version.version}"
        ).signature
        # input tensor shape is (-1, size, size, 3)
        image_size = int(signature.inputs.inputs[0].shape[1])
        tags = version.tags
        return {
            "name": s.model_name,
            "alias": s.model_alias,
            "version": str(version.version),
            "threshold": float(tags.get("decision_threshold", 0.5)),
            "image_size": image_size,
            "test_accuracy": float(tags.get("test_accuracy", "nan")),
            "test_recall_parasitized": float(
                tags.get("test_recall_parasitized", "nan")
            ),
            "git_commit": tags.get("git_commit", ""),
            "data_md5": tags.get("data_md5", ""),
            "class_names": list(CLASS_NAMES),
        }

    def info(self):
        """Champion version metadata, cached for model_info_ttl seconds."""
        with self._lock:
            fresh = time.monotonic() - self._info_time < self.settings.model_info_ttl
            if self._info is not None and fresh:
                return self._info
            try:
                self._info = self._fetch_info()
            except Exception as e:  # registry down, alias missing, bad signature
                if self._info is not None:
                    return self._info  # serve the last known info rather than fail
                raise ModelUnavailable(f"model registry lookup failed: {e}") from e
            self._info_time = time.monotonic()
            return self._info

    # ---- model server --------------------------------------------------------

    def ready(self):
        try:
            r = self.session.get(f"{self.settings.model_server_url}/ping", timeout=2)
            return r.status_code == 200
        except requests.RequestException:
            return False

    def p_parasitized(self, batch):
        """p(parasitized) for a (N, size, size, 3) uint8 batch."""
        try:
            r = self.session.post(
                f"{self.settings.model_server_url}/invocations",
                json={"instances": batch.tolist()},
                timeout=self.settings.request_timeout,
            )
            r.raise_for_status()
            probs = np.asarray(r.json()["predictions"], dtype=np.float32)
        except (requests.RequestException, KeyError, ValueError) as e:
            raise ModelUnavailable(f"model server request failed: {e}") from e
        if probs.ndim != 2 or probs.shape[1] != len(CLASS_NAMES):
            raise ModelUnavailable(f"unexpected prediction shape {probs.shape}")
        return probs[:, POSITIVE]
