"""Verifică patch-urile fără relief, pixelii lipsă și vecinătatea dintre ID-uri.

Întrebarea despre vecinătate: dacă marginea DEM a unui patch continuă marginea
patch-ului cu ID-ul următor mult mai des decât la perechi aleatoare, atunci
ID-urile apropiate sunt zone apropiate, iar un split aleator ar produce leakage.

Rulare, din rădăcina repo-ului:  python scripts/check_structure.py
Rezultat: reports/structure_summary.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from l4s.data import DEM, SLOPE, SPLITS, image_ids, load_image, load_mask, mask_ids  # noqa: E402


def edge_gap(a: np.ndarray, b: np.ndarray) -> float:
    """Diferența medie dintre marginea dreaptă a lui a și cea stângă a lui b."""
    return float(np.abs(a[:, -1] - b[:, 0]).mean())


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
        has_masks = bool(mask_ids(args.data, split))
        dems, flat, flat_values, flat_positive, zero_pixels = {}, [], [], 0, []
        for patch_id in ids:
            x = load_image(args.data, split, patch_id)
            dems[patch_id] = x[..., DEM]
            if x[..., DEM].min() == x[..., DEM].max():
                flat.append(patch_id)
                flat_values.append(float(x[..., DEM][0, 0]))
                if has_masks:
                    flat_positive += int(load_mask(args.data, split, patch_id).any())
            zeros = int((x[..., :DEM + 1] == 0).all(axis=-1).sum())
            if zeros:
                zero_pixels.append((patch_id, zeros))

        usable = [i for i in ids if i not in set(flat)]
        consecutive = [
            edge_gap(dems[a], dems[b]) for a, b in zip(usable, usable[1:]) if b == a + 1
        ]
        random_pairs = [
            edge_gap(dems[a], dems[b])
            for a, b in rng.choice(usable, size=(len(usable) * 2, 2))
            if a != b
        ]
        threshold = float(np.percentile(random_pairs, 5))
        result["splits"][split] = {
            "patches": len(ids),
            "flat_dem_patches": len(flat),
            "flat_dem_values": sorted(set(round(v, 6) for v in flat_values)),
            "flat_dem_patches_with_landslide": flat_positive if has_masks else None,
            "patches_with_all_zero_pixels": len(zero_pixels),
            "all_zero_pixels": sum(z for _, z in zero_pixels),
            "consecutive_pairs": len(consecutive),
            "edge_gap_consecutive_median": float(np.median(consecutive)),
            "edge_gap_random_median": float(np.median(random_pairs)),
            "random_pair_5th_percentile": threshold,
            "consecutive_share_below_threshold": float(np.mean(np.array(consecutive) < threshold)),
        }

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "structure_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
