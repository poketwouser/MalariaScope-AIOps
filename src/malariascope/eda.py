"""EDA stage: four figures about the validated dataset, saved in docs/figures."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from malariascope.common import get_logger

log = get_logger("eda")

CLASSES = ["parasitized", "uninfected"]


def class_balance(df, out):
    counts = df["label_name"].value_counts().reindex(CLASSES)
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(counts.index, counts.values)
    ax.bar_label(bars)
    ax.set_xlabel("Class")
    ax.set_ylabel("Images")
    ax.set_title("Class balance")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def image_sizes(df, out):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, dim in zip(axes, ["width", "height"]):
        for name in CLASSES:
            ax.hist(
                df.loc[df["label_name"] == name, dim], bins=40, alpha=0.6, label=name
            )
        ax.set_xlabel(f"Image {dim} (pixels)")
        ax.set_title(f"Image {dim}")
        ax.legend()
    axes[0].set_ylabel("Images")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def intensity(df, out):
    fig, ax = plt.subplots(figsize=(6, 4))
    for name in CLASSES:
        ax.hist(
            df.loc[df["label_name"] == name, "mean"], bins=50, alpha=0.6, label=name
        )
    ax.set_xlabel("Mean pixel intensity (0-255)")
    ax.set_ylabel("Images")
    ax.set_title("Mean intensity per image")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def samples(df, raw_dir, out, per_class=8, seed=0):
    fig, axes = plt.subplots(2, per_class, figsize=(per_class * 1.4, 3.4))
    for row, name in enumerate(CLASSES):
        picks = df[df["label_name"] == name].sample(per_class, random_state=seed)
        for ax, rel_path in zip(axes[row], picks["path"]):
            with Image.open(Path(raw_dir) / rel_path) as img:
                ax.imshow(img.convert("RGB"))
            ax.set_xticks([])
            ax.set_yticks([])
        axes[row][0].set_ylabel(name)
    fig.suptitle("Sample cells per class")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", default="data/validated/manifest.csv")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--out-dir", default="docs/figures")
    args = parser.parse_args()

    df = pd.read_csv(args.manifest)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    class_balance(df, out / "class_balance.png")
    image_sizes(df, out / "image_sizes.png")
    intensity(df, out / "intensity.png")
    samples(df, args.raw_dir, out / "samples.png")
    log.info("wrote 4 figures to %s", out)


if __name__ == "__main__":
    main()
