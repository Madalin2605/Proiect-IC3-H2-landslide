"""Descarcă măștile de validare și test de pe Hugging Face.

Fiecare fișier este verificat înainte să fie păstrat: cheia `mask`, forma
128x128 și valorile 0/1.

Rulare, din rădăcina repo-ului:  python scripts/download_masks.py
"""
import argparse
import io
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from l4s.data import SPLITS, image_ids  # noqa: E402

BASE = "https://huggingface.co/datasets/ibm-nasa-geospatial/Landslide4sense/resolve/main/annotations"


def fetch(url: str, attempts: int = 4) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "ic3-h2-landslide"})
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read()
        except OSError:
            if attempt == attempts - 1:
                raise
            time.sleep(2**attempt)
    raise RuntimeError(url)


def is_valid(content: bytes) -> bool:
    with h5py.File(io.BytesIO(content), "r") as f:
        if list(f.keys()) != ["mask"]:
            return False
        mask = f["mask"][()]
    return mask.shape == (128, 128) and set(np.unique(mask).tolist()) <= {0, 1}


def download(job: tuple[str, Path]) -> str:
    url, path = job
    if path.exists():
        return "skipped"
    content = fetch(url)
    if not is_valid(content):
        return f"invalid:{path.name}"
    temporary = path.with_suffix(".part")
    temporary.write_bytes(content)
    temporary.replace(path)
    return "saved"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    for split in ("validation", "test"):
        folder = args.data / SPLITS[split] / "mask"
        folder.mkdir(parents=True, exist_ok=True)
        jobs = [(f"{BASE}/{split}/mask_{i}.h5", folder / f"mask_{i}.h5") for i in image_ids(args.data, split)]
        with ThreadPoolExecutor(args.workers) as pool:
            results = list(pool.map(download, jobs))
        counts = {r: results.count(r) for r in sorted(set(results))}
        print(split, len(jobs), counts)


if __name__ == "__main__":
    main()
