"""Datele pentru rețelele de la CP3: cache pe disc, normalizare pe fold și batch-uri pe GPU.

Citirea celor 3.799 de fișiere HDF5 la fiecare epocă ar fi prea lentă, așa că fiecare split
se transformă o singură dată într-un cache NumPy (scripts/build_cache.py), în data/cache/:

    {split}_x.npy      N x 18 x 128 x 128, float16: trăsăturile reprezentării A (features.py)
    {split}_y.npy      N x 128 x 128, uint8: masca (1 = alunecare)
    {split}_valid.npy  N x 128 x 128, bool: False pentru pixelii zero în toate benzile
    {split}_ids.npy    N, int: ID-ul patch-ului
    {split}_stats.npz  sume per patch și canal, pe pixelii valizi (pentru normalizare)

Valorile din fișiere sunt mici (sub ~60) și indicii sunt în [-1, 1], deci float16 ajunge.
Train-ul ocupă ~2,2 GB și încape în memorie.

Normalizarea: media și abaterea standard per canal se calculează doar pe patch-urile de
antrenare ale fold-ului, din sumele per patch, deci exact și fără o nouă trecere prin date.
Reprezentarea B folosește doar canalele pantă și DEM din același cache.

Batch-urile se pregătesc direct pe GPU (GpuBatches): normalizarea și augmentarea pe procesor
erau gâtul de sticlă al antrenării (~17 s pe epocă doar pentru date).
"""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from l4s.data import image_ids, load_image, load_mask
from l4s.features import feature_cube, feature_names, valid_pixels

FEATURES = feature_names("s2_topo")
CHANNELS = {
    "s2_topo": list(range(len(FEATURES))),
    "topo": [FEATURES.index("SLOPE"), FEATURES.index("DEM")],
}
CLIP = 10.0  # valorile normalizate se taie la ±10 abateri standard (cozi lungi, ex. B10 pe validare)


# ---------- construirea cache-ului ----------

def build_cache(root: Path, split: str, out_dir: Path, ids: list[int] | None = None) -> None:
    """Citește patch-urile unui split și scrie cache-ul descris mai sus."""
    ids = image_ids(root, split) if ids is None else list(ids)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n, c, (h, w) = len(ids), len(FEATURES), (128, 128)

    x_out = np.lib.format.open_memmap(out_dir / f"{split}_x.npy", mode="w+", dtype=np.float16, shape=(n, c, h, w))
    y_out = np.lib.format.open_memmap(out_dir / f"{split}_y.npy", mode="w+", dtype=np.uint8, shape=(n, h, w))
    v_out = np.lib.format.open_memmap(out_dir / f"{split}_valid.npy", mode="w+", dtype=bool, shape=(n, h, w))
    sums = np.zeros((n, c))
    sumsq = np.zeros((n, c))
    counts = np.zeros(n, dtype=np.int64)

    for row, patch_id in enumerate(ids):
        raw = load_image(root, split, patch_id)
        valid = valid_pixels(raw)
        feats = feature_cube(raw, "s2_topo")  # H x W x 18, float32
        x_out[row] = feats.transpose(2, 0, 1).astype(np.float16)
        y_out[row] = load_mask(root, split, patch_id).astype(np.uint8)
        v_out[row] = valid
        pix = feats[valid].astype(np.float64)
        sums[row], sumsq[row], counts[row] = pix.sum(axis=0), (pix ** 2).sum(axis=0), len(pix)

    for arr in (x_out, y_out, v_out):
        arr.flush()
    np.save(out_dir / f"{split}_ids.npy", np.asarray(ids, dtype=np.int64))
    np.savez(out_dir / f"{split}_stats.npz", sums=sums, sumsq=sumsq, counts=counts)


# ---------- citirea cache-ului ----------

@dataclass
class PatchCache:
    split: str
    x: np.ndarray        # N x 18 x H x W, float16
    y: np.ndarray        # N x H x W, uint8
    valid: np.ndarray    # N x H x W, bool
    ids: np.ndarray      # N
    sums: np.ndarray     # N x 18
    sumsq: np.ndarray    # N x 18
    counts: np.ndarray   # N
    row_of: dict = field(init=False)

    def __post_init__(self):
        self.row_of = {int(i): r for r, i in enumerate(self.ids)}

    def rows(self, ids) -> np.ndarray:
        return np.array([self.row_of[int(i)] for i in ids], dtype=np.int64)


def load_cache(cache_dir: Path, split: str, in_memory: bool = True) -> PatchCache:
    """Încarcă cache-ul unui split; cu in_memory=False rămâne pe disc (memmap)."""
    cache_dir = Path(cache_dir)
    mode = None if in_memory else "r"
    stats = np.load(cache_dir / f"{split}_stats.npz")
    return PatchCache(
        split=split,
        x=np.load(cache_dir / f"{split}_x.npy", mmap_mode=mode),
        y=np.load(cache_dir / f"{split}_y.npy", mmap_mode=mode),
        valid=np.load(cache_dir / f"{split}_valid.npy", mmap_mode=mode),
        ids=np.load(cache_dir / f"{split}_ids.npy"),
        sums=stats["sums"], sumsq=stats["sumsq"], counts=stats["counts"],
    )


def channel_stats(cache: PatchCache, ids, representation: str) -> tuple[np.ndarray, np.ndarray]:
    """Media și abaterea standard per canal, pe pixelii valizi ai patch-urilor date (doar train-ul fold-ului)."""
    rows = cache.rows(ids)
    ch = CHANNELS[representation]
    n = cache.counts[rows].sum()
    mean = cache.sums[rows][:, ch].sum(axis=0) / n
    var = cache.sumsq[rows][:, ch].sum(axis=0) / n - mean ** 2
    std = np.sqrt(np.maximum(var, 1e-12))
    return mean.astype(np.float32), std.astype(np.float32)


# ---------- batch-uri pentru antrenare ----------

try:
    import torch
except ImportError:  # cache-ul se poate construi și fără torch
    torch = None


class GpuBatches:
    """Batch-uri pregătite pe GPU, pentru antrenare și evaluare.

    Procesorul doar copiază din cache batch-ul brut în float16. Normalizarea, tăierea la
    ±CLIP, mascarea pixelilor invalizi și augmentarea (rotație cu k·90° și flip, aleasă
    separat pentru fiecare patch) se fac pe GPU. Ordinea și augmentarea depind doar de seed.
    Iterarea întoarce (x, y, valid, ids), cu x: B x C x H x W, y și valid: B x 1 x H x W.
    """

    def __init__(self, cache: PatchCache, ids, representation: str, mean: np.ndarray, std: np.ndarray,
                 batch_size: int = 16, shuffle: bool = False, augment: bool = False, seed: int = 0,
                 device: str = "cuda"):
        self.cache = cache
        self.ids = np.asarray([int(i) for i in ids], dtype=np.int64)
        self.rows = cache.rows(self.ids)
        self.channels = CHANNELS[representation]
        self.all_channels = self.channels == list(range(cache.x.shape[1]))
        self.device = torch.device(device)
        self.mean = torch.as_tensor(mean, device=self.device).view(1, -1, 1, 1)
        self.std = torch.as_tensor(std, device=self.device).view(1, -1, 1, 1)
        self.batch_size, self.shuffle, self.augment = batch_size, shuffle, augment
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return (len(self.rows) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        order = self.rng.permutation(len(self.rows)) if self.shuffle else np.arange(len(self.rows))
        for start in range(0, len(order), self.batch_size):
            pick = order[start:start + self.batch_size]
            rows = self.rows[pick]
            x = self.cache.x[rows] if self.all_channels else self.cache.x[rows][:, self.channels]
            x = torch.from_numpy(np.ascontiguousarray(x)).to(self.device, non_blocking=True).float()
            y = torch.from_numpy(self.cache.y[rows]).to(self.device).float().unsqueeze(1)
            v = torch.from_numpy(self.cache.valid[rows]).to(self.device).float().unsqueeze(1)
            x = ((x - self.mean) / self.std).clamp_(-CLIP, CLIP) * v
            if self.augment:
                x, y, v = self._augment(x, y, v)
            yield x, y, v, self.ids[pick]

    def _augment(self, x, y, v):
        ks = self.rng.integers(4, size=len(x))
        flips = self.rng.integers(2, size=len(x)).astype(bool)
        out = [[], [], []]
        for i, (k, flip) in enumerate(zip(ks, flips)):
            for j, a in enumerate((x[i], y[i], v[i])):
                a = torch.rot90(a, int(k), dims=(-2, -1))
                out[j].append(a.flip(-1) if flip else a)
        return [torch.stack(o) for o in out]
