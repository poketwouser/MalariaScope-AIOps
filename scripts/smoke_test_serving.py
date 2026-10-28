"""Smoke test of the serving chain: MLflow model server, then the backend.

Sends the frontend sample images to both and prints label, p(parasitized) and
latency. Exits non-zero if any call fails.

Usage: PYTHONPATH=src python scripts/smoke_test_serving.py \
         [--model-server http://127.0.0.1:5001] [--backend http://127.0.0.1:8000]
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import requests
from PIL import Image

from malariascope.image import to_model_input
from malariascope.labels import POSITIVE

SAMPLES = Path(__file__).resolve().parent.parent / "frontend" / "samples"


def check_model_server(url, image_size):
    print(f"MLflow model server {url}")
    assert requests.get(f"{url}/ping", timeout=5).status_code == 200, "ping failed"
    for path in sorted(SAMPLES.glob("*.png")):
        x = to_model_input(Image.open(path), image_size)[None]
        start = time.perf_counter()
        r = requests.post(
            f"{url}/invocations", json={"instances": x.tolist()}, timeout=30
        )
        r.raise_for_status()
        p = np.asarray(r.json()["predictions"])[0, POSITIVE]
        ms = (time.perf_counter() - start) * 1000
        print(f"  {path.name:20s} p(parasitized)={p:.4f}  {ms:6.1f} ms")


def check_backend(url):
    print(f"Backend {url}")
    info = requests.get(f"{url}/model/info", timeout=10).json()
    print(
        f"  champion v{info['version']} threshold={info['threshold']:.4f} "
        f"image_size={info['image_size']}"
    )
    assert requests.get(f"{url}/ready", timeout=5).status_code == 200, "not ready"
    for path in sorted(SAMPLES.glob("*.png")):
        r = requests.post(
            f"{url}/predict",
            files={"file": (path.name, path.read_bytes(), "image/png")},
            timeout=30,
        )
        r.raise_for_status()
        b = r.json()
        print(
            f"  {path.name:20s} -> {b['label']:11s} confidence={b['confidence']:.3f}"
            f"  {b['latency_ms']:6.1f} ms"
        )
    return info["image_size"]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-server", default="http://127.0.0.1:5001")
    parser.add_argument("--backend", default="http://127.0.0.1:8000")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--skip-backend", action="store_true")
    args = parser.parse_args()
    try:
        check_model_server(args.model_server, args.image_size)
        if not args.skip_backend:
            check_backend(args.backend)
    except (requests.RequestException, AssertionError, KeyError) as e:
        print(f"FAILED: {e}")
        sys.exit(1)
    print("OK")


if __name__ == "__main__":
    main()
