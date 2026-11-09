"""Citește toate patch-urile și scrie statisticile folosite în data card.

Rulare, din rădăcina repo-ului:  python scripts/check_data.py
Rezultate: reports/data_summary.json și reports/band_stats.csv
"""
import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from l4s.data import BANDS, DEM, PATCH_SHAPE, SPLITS, image_ids, load_image, load_mask, mask_ids  # noqa: E402

SAMPLED_PIXELS_PER_PATCH = 200


def digest(array: np.ndarray) -> str:
    return hashlib.sha1(array.tobytes()).hexdigest()


def duplicates(hashes: dict[str, list[tuple[str, int]]]) -> dict:
    groups = [members for members in hashes.values() if len(members) > 1]
    cross = [g for g in groups if len({split for split, _ in g}) > 1]
    return {
        "groups": len(groups),
        "extra_patches": sum(len(g) - 1 for g in groups),
        "groups_across_splits": len(cross),
        "examples": [[f"{s}:{i}" for s, i in g] for g in groups[:5]],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--out", type=Path, default=ROOT / "reports")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    image_hashes, dem_hashes = defaultdict(list), defaultdict(list)
    summary = {"seed": args.seed, "expected_shape": PATCH_SHAPE, "splits": {}}
    band_rows = []

    for split in SPLITS:
        ids = image_ids(args.data, split)
        masks = set(mask_ids(args.data, split))
        count = len(BANDS)
        sums, squares = np.zeros(count), np.zeros(count)
        mins, maxs = np.full(count, np.inf), np.full(count, -np.inf)
        zeros, samples = np.zeros(count), []
        bad_shape, non_finite, constant_bands = [], [], defaultdict(int)
        dtypes = set()

        for patch_id in ids:
            x = load_image(args.data, split, patch_id)
            dtypes.add(str(x.dtype))
            if x.shape != PATCH_SHAPE:
                bad_shape.append(patch_id)
                continue
            if not np.isfinite(x).all():
                non_finite.append(patch_id)
            flat = x.reshape(-1, count)
            sums += flat.sum(0)
            squares += (flat**2).sum(0)
            mins, maxs = np.minimum(mins, flat.min(0)), np.maximum(maxs, flat.max(0))
            zeros += (flat == 0).sum(0)
            samples.append(flat[rng.integers(0, len(flat), SAMPLED_PIXELS_PER_PATCH)])
            for band in np.flatnonzero(flat.min(0) == flat.max(0)):
                constant_bands[BANDS[band]] += 1
            image_hashes[digest(x)].append((split, patch_id))
            dem_hashes[digest(x[..., DEM])].append((split, patch_id))

        pixels = (len(ids) - len(bad_shape)) * PATCH_SHAPE[0] * PATCH_SHAPE[1]
        means = sums / pixels
        stds = np.sqrt(np.maximum(squares / pixels - means**2, 0))
        p1, p50, p99 = np.percentile(np.vstack(samples), [1, 50, 99], axis=0)
        for b, name in enumerate(BANDS):
            band_rows.append([split, name, mins[b], p1[b], p50[b], p99[b], maxs[b], means[b], stds[b], zeros[b] / pixels])

        info = {
            "image_patches": len(ids),
            "mask_patches": len(masks),
            "image_dtypes": sorted(dtypes),
            "bad_shape": bad_shape,
            "non_finite_patches": non_finite,
            "constant_band_occurrences": dict(constant_bands),
        }
        if masks:
            positive = positive_patches = total = 0
            values, bad_masks = set(), []
            for patch_id in ids:
                if patch_id not in masks:
                    continue
                m = load_mask(args.data, split, patch_id)
                if m.shape != PATCH_SHAPE[:2]:
                    bad_masks.append(patch_id)
                    continue
                values |= set(np.unique(m).tolist())
                c = int((m == 1).sum())
                positive += c
                positive_patches += c > 0
                total += m.size
            info.update(
                images_without_mask=len(set(ids) - masks),
                masks_without_image=len(masks - set(ids)),
                mask_values=sorted(values),
                bad_mask_shape=bad_masks,
                patches_with_landslide=positive_patches,
                landslide_pixels=positive,
                total_pixels=total,
                landslide_pixel_fraction=positive / total,
                all_negative_accuracy=1 - positive / total,
            )
        summary["splits"][split] = info

    summary["exact_duplicate_images"] = duplicates(image_hashes)
    summary["identical_dem_patches"] = duplicates(dem_hashes)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "data_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with open(args.out / "band_stats.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "band", "min", "p1", "p50", "p99", "max", "mean", "std", "zero_fraction"])
        for row in band_rows:
            writer.writerow([*row[:2], *(f"{v:.6g}" for v in row[2:])])
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
