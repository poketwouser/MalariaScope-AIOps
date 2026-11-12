"""Client for the MLflow model server (predictions) and the registry (model info).

the backend never loads the model, so its image has no tensorflow. the model
server follows @champion on its own after a promotion.
"""

import threading
import time

import mlflow
import numpy as np
import requests
from mlflow import MlflowClient

from app import config
from malariascope.labels import CLASS_NAMES, POSITIVE


class ModelUnavailable(Exception):
    pass


class ModelClient:
    def __init__(self, session=None):
        self.session = session or requests.Session()
        self._info = None
        self._info_time = 0.0
        self._lock = threading.Lock()

    def _fetch_info(self):
        client = MlflowClient(tracking_uri=config.TRACKING_URI)
        version = client.get_model_version_by_alias(
            config.MODEL_NAME, config.MODEL_ALIAS
        )
        mlflow.set_tracking_uri(config.TRACKING_URI)
        signature = mlflow.models.get_model_info(
            f"models:/{config.MODEL_NAME}/{version.version}"
        ).signature
        # input tensor shape is (-1, size, size, 3)
        image_size = int(signature.inputs.inputs[0].shape[1])
        tags = version.tags
        return {
            "name": config.MODEL_NAME,
            "alias": config.MODEL_ALIAS,
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
        # cached for model_info_ttl seconds
        with self._lock:
            fresh = time.monotonic() - self._info_time < config.MODEL_INFO_TTL
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

    def ready(self):
        try:
            r = self.session.get(f"{config.MODEL_SERVER_URL}/ping", timeout=2)
            return r.status_code == 200
        except requests.RequestException:
            return False

    def p_parasitized(self, batch):
        """p(parasitized) for a (N, size, size, 3) uint8 batch."""
        try:
            r = self.session.post(
                f"{config.MODEL_SERVER_URL}/invocations",
                json={"instances": batch.tolist()},
                timeout=config.REQUEST_TIMEOUT,
            )
            r.raise_for_status()
            probs = np.asarray(r.json()["predictions"], dtype=np.float32)
        except (requests.RequestException, KeyError, ValueError) as e:
            raise ModelUnavailable(f"model server request failed: {e}") from e
        if probs.ndim != 2 or probs.shape[1] != len(CLASS_NAMES):
            raise ModelUnavailable(f"unexpected prediction shape {probs.shape}")
        return probs[:, POSITIVE]
