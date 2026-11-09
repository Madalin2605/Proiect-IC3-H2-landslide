"""Verifică dacă măștile corespund imaginilor locale.

Alunecările au vegetație mai slabă decât terenul din jur. Pentru fiecare patch cu
alunecări se calculează NDVI(în afara măștii) - NDVI(în mască), o dată pe perechea
corectă și o dată pe perechi amestecate. Dacă măștile corespund, diferența este
clar pozitivă pe perechile corecte și apropiată de zero pe cele amestecate.

Rulare, din rădăcina repo-ului:  python scripts/check_mask_match.py
Rezultat: reports/mask_match.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from l4s.data import SPLITS, image_ids, load_image, load_mask  # noqa: E402

RED, NIR = 3, 7


def ndvi(x: np.ndarray) -> np.ndarray:
    return (x[..., NIR] - x[..., RED]) / np.maximum(x[..., NIR] + x[..., RED], 1e-6)


def contrast(v: np.ndarray, mask: np.ndarray) -> float:
    inside = mask == 1
    if inside.all() or not inside.any():
        return float("nan")
    return float(v[~inside].mean() - v[inside].mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--out", type=Path, default=ROOT / "reports")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    result = {"seed": args.seed, "splits": {}}
    for split in SPLITS:
        ids = image_ids(args.data, split)
        ndvis = np.stack([ndvi(load_image(args.data, split, i)) for i in ids])
        masks = np.stack([load_mask(args.data, split, i) for i in ids])
        positive = np.flatnonzero(masks.any(axis=(1, 2)))
        shuffled = rng.permutation(positive)
        true_pairs = np.array([contrast(ndvis[i], masks[i]) for i in positive])
        shuffled_pairs = np.array([contrast(ndvis[i], masks[j]) for i, j in zip(positive, shuffled)])
        result["splits"][split] = {
            "patches": len(ids),
            "patches_with_landslide": int(len(positive)),
            "landslide_pixels": int(masks.sum()),
            "total_pixels": int(masks.size),
            "landslide_pixel_fraction": float(masks.mean()),
            "all_negative_accuracy": float(1 - masks.mean()),
            "ndvi_contrast_true_median": float(np.nanmedian(true_pairs)),
            "ndvi_contrast_true_share_positive": float(np.nanmean(true_pairs > 0)),
            "ndvi_contrast_shuffled_median": float(np.nanmedian(shuffled_pairs)),
            "ndvi_contrast_shuffled_share_positive": float(np.nanmean(shuffled_pairs > 0)),
        }

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "mask_match.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
