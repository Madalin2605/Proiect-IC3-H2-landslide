"""CP3, pasul 1: construiește cache-ul pentru rețele și verifică datele care intră în model.

Pentru train, validare și test scrie în data/cache/ trăsăturile reprezentării A (18 canale,
float16), măștile, pixelii valizi și sumele per patch pentru normalizare (src/l4s/dataset.py).

Verificări:
  1. numărul de pixeli de alunecare și de pixeli invalizi coincide cu data card-ul;
  2. eroarea de rotunjire float16 față de fișierele originale, pe câteva patch-uri;
  3. normalizarea pe fold: statisticile se calculează doar pe train-ul fold-ului, deci
     diferă între fold-uri (fold 1 vs. fold 5).

Rulare, din rădăcina repo-ului:  python scripts/build_cache.py
Refacerea cache-ului:            python scripts/build_cache.py --force
Durată: ~3-5 minute la prima rulare; cache-ul ocupă ~2,9 GB în data/cache/ (ignorat de Git).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from l4s.data import load_image  # noqa: E402
from l4s.dataset import FEATURES, build_cache, channel_stats, load_cache  # noqa: E402
from l4s.features import feature_cube  # noqa: E402
from l4s.splits import load_folds  # noqa: E402

SPLITS = ["train", "validation", "test"]
# Pixeli zero în toate benzile, raportați în methods_H2.md (secțiunea 1).
EXPECTED_INVALID = {"train": 7423, "validation": 256, "test": 512}


def check_counts(cache, summary: dict) -> None:
    landslide = int(cache.y.sum(dtype=np.int64))
    invalid = int((~cache.valid).sum())
    expected = summary["splits"][cache.split]["landslide_pixels"]
    ok = landslide == expected and invalid == EXPECTED_INVALID[cache.split]
    print(f"  {cache.split:<10} {len(cache.ids):>5} patch-uri | alunecare {landslide:>9} (data card {expected:>9}) "
          f"| invalizi {invalid:>5} (așteptat {EXPECTED_INVALID[cache.split]}) | {'OK' if ok else 'DIFERENȚĂ'}")


def check_rounding(cache, data: Path, rng: np.random.Generator) -> None:
    rows = rng.choice(len(cache.ids), size=5, replace=False)
    worst = 0.0
    for r in rows:
        exact = feature_cube(load_image(data, cache.split, int(cache.ids[r])), "s2_topo").transpose(2, 0, 1)
        rel = np.abs(cache.x[r].astype(np.float32) - exact) / np.maximum(np.abs(exact), 1e-3)
        worst = max(worst, float(rel.max()))
    print(f"  {cache.split:<10} eroare relativă maximă float16 pe 5 patch-uri: {worst:.2e}")


def check_normalization(train, folds: dict) -> None:
    for k in (1, 5):
        mean, std = channel_stats(train, folds[k]["train_ids"], "s2_topo")
        print(f"  fold {k}: medie pe train   " + "  ".join(f"{n} {m:.3f}" for n, m in zip(FEATURES, mean) if n in ("B4", "NDVI", "SLOPE", "DEM")))
        print(f"          abatere pe train " + "  ".join(f"{n} {s:.3f}" for n, s in zip(FEATURES, std) if n in ("B4", "NDVI", "SLOPE", "DEM")))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--cache", type=Path, default=ROOT / "data" / "cache")
    parser.add_argument("--splits", type=Path, default=ROOT / "reports" / "CP2" / "splits.json")
    parser.add_argument("--force", action="store_true", help="reconstruiește cache-ul chiar dacă există")
    args = parser.parse_args()

    start = time.time()
    for split in SPLITS:
        if args.force or not (args.cache / f"{split}_stats.npz").exists():
            t = time.time()
            print(f"Construiesc cache-ul pentru {split}...")
            build_cache(args.data, split, args.cache)
            print(f"  gata în {time.time() - t:.0f} s")
        else:
            print(f"Cache existent pentru {split} (folosește --force ca să-l refaci)")

    size = sum(p.stat().st_size for p in args.cache.glob("*")) / 2**30
    print(f"Cache: {args.cache} ({size:.2f} GB)")

    caches = {s: load_cache(args.cache, s) for s in SPLITS}
    summary = json.loads((ROOT / "reports" / "CP1" / "data_summary.json").read_text(encoding="utf-8"))
    folds = {f["fold"]: f for f in load_folds(args.splits)}
    rng = np.random.default_rng(42)

    print("\n1. Pixeli de alunecare și pixeli invalizi")
    for cache in caches.values():
        check_counts(cache, summary)

    print("\n2. Rotunjirea la float16")
    for cache in caches.values():
        check_rounding(cache, args.data, rng)

    print("\n3. Normalizarea, calculată doar pe train-ul fold-ului")
    check_normalization(caches["train"], folds)

    print(f"\nGata în {time.time() - start:.0f} s.")


if __name__ == "__main__":
    main()
