"""Desenează un patch cu RGB, pantă, DEM și mască.

Rulare, din rădăcina repo-ului:  python scripts/plot_patch.py --split train --id 241
"""
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from l4s.data import DEM, SLOPE, load_image, load_mask  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--split", default="train", choices=["train"])
    parser.add_argument("--id", type=int, default=241)
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "figures")
    args = parser.parse_args()

    x = load_image(args.data, args.split, args.id)
    mask = load_mask(args.data, args.split, args.id)
    rgb = x[..., [3, 2, 1]]
    low, high = np.percentile(rgb, [1, 99])
    rgb = np.clip((rgb - low) / (high - low), 0, 1)

    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6), constrained_layout=True)
    axes[0].imshow(rgb)
    axes[0].set_title("Sentinel-2 RGB (B4, B3, B2)")
    for ax, band, title in [(axes[1], SLOPE, "Pantă (B13)"), (axes[2], DEM, "DEM (B14)")]:
        image = ax.imshow(x[..., band], cmap="viridis")
        ax.set_title(title)
        fig.colorbar(image, ax=ax, shrink=0.85, label="valoare din fișier")
    axes[3].imshow(mask, cmap="magma", vmin=0, vmax=1, interpolation="nearest")
    axes[3].set_title(f"Mască (alunecare: {mask.mean() * 100:.1f}%)")
    for ax in axes:
        ax.set_xlabel("pixeli (~10 m/pixel)")
        ax.set_ylabel("pixeli")
    fig.suptitle(f"Landslide4Sense, {args.split}, image_{args.id}.h5, formă {x.shape}")

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"patch_{args.split}_{args.id}.png"
    fig.savefig(path, dpi=150)
    print(path)


if __name__ == "__main__":
    main()
