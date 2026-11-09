"""Citirea patch-urilor Landslide4Sense din fișierele HDF5."""
import re
from pathlib import Path

import h5py
import numpy as np

SPLITS = {"train": "TrainData", "validation": "ValidData", "test": "TestData"}
BANDS = [f"B{i}" for i in range(1, 13)] + ["SLOPE", "DEM"]
SLOPE, DEM = 12, 13
PATCH_SHAPE = (128, 128, len(BANDS))


def _ids(folder: Path, prefix: str) -> list[int]:
    pattern = re.compile(rf"{prefix}_(\d+)\.h5$")
    matches = (pattern.match(p.name) for p in folder.glob(f"{prefix}_*.h5"))
    return sorted(int(m.group(1)) for m in matches if m)


def image_ids(root: Path, split: str) -> list[int]:
    return _ids(Path(root) / SPLITS[split] / "img", "image")


def mask_ids(root: Path, split: str) -> list[int]:
    return _ids(Path(root) / SPLITS[split] / "mask", "mask")


def load_image(root: Path, split: str, patch_id: int) -> np.ndarray:
    """Întoarce un tablou H x W x 14: B1-B12, pantă, DEM."""
    path = Path(root) / SPLITS[split] / "img" / f"image_{patch_id}.h5"
    with h5py.File(path, "r") as f:
        return f["img"][()]


def load_mask(root: Path, split: str, patch_id: int) -> np.ndarray:
    """Întoarce masca H x W: 1 = alunecare, 0 = restul."""
    path = Path(root) / SPLITS[split] / "mask" / f"mask_{patch_id}.h5"
    with h5py.File(path, "r") as f:
        return f["mask"][()]
