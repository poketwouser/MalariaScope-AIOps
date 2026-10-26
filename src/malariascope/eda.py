"""EDA stage: figures describing the validated dataset.

Figures follow the course's seven plotting commandments: colour-blind-safe
palette (Okabe-Ito), named axes with units, legends, titles and a short story
line under each title.

Output (docs/figures/): class_balance.png, image_sizes.png, intensity.png, samples.png

Run: PYTHONPATH=src python -m malariascope.eda
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from malariascope.common import get_logger  # noqa: E402

log = get_logger("eda")

# Okabe-Ito colours, distinguishable with the common types of colour blindness
COLOURS = {"parasitized": "#D55E00", "uninfected": "#0072B2"}


def titled(ax, title, story):
    ax.set_title(f"{title}\n{story}", fontsize=10, loc="left")


def class_balance(df, out):
    counts = df["label_name"].value_counts().reindex(list(COLOURS))
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(counts.index, counts.values, color=[COLOURS[c] for c in counts.index])
    ax.bar_label(bars)
    ax.set_xlabel("Class")
    ax.set_ylabel("Images (count, linear scale)")
    titled(ax, "Class balance", "The two classes are almost exactly balanced.")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def image_sizes(df, out):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, dim in zip(axes, ("width", "height")):
        for name, colour in COLOURS.items():
            ax.hist(
                df.loc[df["label_name"] == name, dim],
                bins=40,
                alpha=0.6,
                color=colour,
                label=name,
            )
        ax.set_xlabel(f"Image {dim} (pixels)")
        ax.legend(title="Class")
    axes[0].set_ylabel("Images (count, linear scale)")
    titled(axes[0], "Image width", "Cell crops vary in size, so they are resized.")
    titled(axes[1], "Image height", "Both classes have similar size ranges.")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def intensity(df, out):
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, colour in COLOURS.items():
        ax.hist(
            df.loc[df["label_name"] == name, "mean"],
            bins=50,
            alpha=0.6,
            color=colour,
            label=name,
        )
    ax.set_xlabel("Mean pixel intensity (0-255)")
    ax.set_ylabel("Images (count, linear scale)")
    ax.legend(title="Class")
    titled(
        ax,
        "Mean intensity per image",
        "Baseline for the drift metric watched in production.",
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def samples(df, raw_dir, out, per_class=8, seed=0):
    fig, axes = plt.subplots(2, per_class, figsize=(per_class * 1.4, 3.4))
    for row, name in enumerate(COLOURS):
        picks = df[df["label_name"] == name].sample(per_class, random_state=seed)
        for ax, rel_path in zip(axes[row], picks["path"]):
            with Image.open(Path(raw_dir) / rel_path) as img:
                ax.imshow(img.convert("RGB"))
            ax.set_xticks([])
            ax.set_yticks([])
        axes[row][0].set_ylabel(name, color=COLOURS[name], fontsize=10)
    fig.suptitle(
        "Sample cells per class\n"
        "Parasitized cells show purple-stained Plasmodium inside.",
        fontsize=10,
        x=0.01,
        ha="left",
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150)
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
