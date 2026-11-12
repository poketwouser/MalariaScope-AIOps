"""Send train-split cell images to the backend, optionally with feedback or brightened.

only train images are used, so feedback never touches val/test. brightened
images (--brightness, the drift demo) are never sent as feedback.

  PYTHONPATH=src python scripts/simulate_traffic.py -n 60 --feedback
  PYTHONPATH=src python scripts/simulate_traffic.py -n 40 --brightness 1.3
"""

import argparse
import io
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from PIL import Image, ImageEnhance

ROOT = Path(__file__).resolve().parent.parent


def pick_train_images(manifest, n, seed):
    """n train-split raw images, balanced over the two labels."""
    train = manifest[(manifest["split"] == "train") & (manifest["source"] == "raw")]
    per_label = -(-n // 2)
    picked = pd.concat(
        g.sample(per_label, random_state=seed) for _, g in train.groupby("label_name")
    )
    return picked.sample(frac=1, random_state=seed).head(n)


def png_bytes(path, brightness):
    img = Image.open(path).convert("RGB")
    if brightness != 1.0:
        img = ImageEnhance.Brightness(img).enhance(brightness)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backend", default="http://127.0.0.1:8000")
    parser.add_argument("-n", type=int, default=40, help="number of images")
    parser.add_argument("--feedback", action="store_true")
    parser.add_argument("--brightness", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--delay", type=float, default=0.2, help="seconds between")
    parser.add_argument("--manifest", default=str(ROOT / "data/split/manifest.csv"))
    parser.add_argument("--raw-dir", default=str(ROOT / "data/raw"))
    args = parser.parse_args()
    if args.feedback and args.brightness != 1.0:
        sys.exit("brightened images are a drift demo and must not become feedback")

    manifest = pd.read_csv(args.manifest)
    if "source" not in manifest:
        manifest["source"] = "raw"
    rows = pick_train_images(manifest, args.n, args.seed)
    correct = 0
    for i, row in enumerate(rows.itertuples(), 1):
        data = png_bytes(Path(args.raw_dir) / row.path, args.brightness)
        files = {"file": ("cell.png", data, "image/png")}
        r = requests.post(f"{args.backend}/predict", files=files, timeout=30)
        r.raise_for_status()
        pred = r.json()
        correct += pred["label"] == row.label_name
        line = f"{i:3d} {row.path:32s} true={row.label_name:11s} pred={pred['label']}"
        if args.feedback:
            fb = requests.post(
                f"{args.backend}/feedback",
                files={"file": ("cell.png", data, "image/png")},
                data={
                    "request_id": pred["request_id"],
                    "predicted_label": pred["label"],
                    "correct_label": row.label_name,
                    "p_parasitized": pred["p_parasitized"],
                    "model_version": pred["model_version"],
                },
                timeout=30,
            )
            fb.raise_for_status()
            line += f"  feedback #{fb.json()['feedback_id']}"
        print(line, flush=True)
        time.sleep(args.delay)
    print(f"{correct}/{len(rows)} predictions matched the true label")


if __name__ == "__main__":
    main()
