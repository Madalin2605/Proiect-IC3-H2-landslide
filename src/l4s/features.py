"""Reprezentările pe pixel folosite la CP2.

Reprezentarea A (s2_topo): cele 12 benzi Sentinel-2, panta, DEM și 4 indici derivați.
Reprezentarea B (topo): doar panta și DEM, proxy-ul topografic de susceptibilitate.

Benzile din fișiere sunt scalate cu constante nedocumentate (vezi data card), deci
indicii de mai jos nu sunt valorile fizice standard. Sunt rapoarte normalizate între
benzi scalate: păstrează ordinea relativă a pixelilor într-o bandă, nu valoarea absolută.
"""
import numpy as np

from l4s.data import BANDS, DEM, SLOPE

# Indexuri în tabloul H x W x 14 (B8A lipsește din set).
B2, B3, B4, B8, B12 = 1, 2, 3, 7, 11
EPS = 1e-6


def _normalized_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a - b) / np.maximum(a + b, EPS)


def indices(x: np.ndarray) -> dict[str, np.ndarray]:
    """Indicii spectrali, fiecare H x W."""
    return {
        # vegetație: scade pe cicatricea alunecării
        "NDVI": _normalized_difference(x[..., B8], x[..., B4]),
        # apă / umiditate (McFeeters)
        "NDWI": _normalized_difference(x[..., B3], x[..., B8]),
        # NIR vs SWIR2: sol gol și vegetație distrusă
        "NBR": _normalized_difference(x[..., B8], x[..., B12]),
        # luminozitate în vizibil: solul proaspăt expus e mai luminos
        "BRIGHT": x[..., [B2, B3, B4]].mean(axis=-1),
    }


INDEX_NAMES = ["NDVI", "NDWI", "NBR", "BRIGHT"]

REPRESENTATIONS = {
    "s2_topo": BANDS + INDEX_NAMES,
    "topo": ["SLOPE", "DEM"],
}


def feature_names(representation: str) -> list[str]:
    return list(REPRESENTATIONS[representation])


def feature_cube(x: np.ndarray, representation: str) -> np.ndarray:
    """Întoarce H x W x F, cu trăsăturile în ordinea din feature_names."""
    if representation == "topo":
        return np.stack([x[..., SLOPE], x[..., DEM]], axis=-1).astype(np.float32)
    if representation == "s2_topo":
        idx = indices(x)
        extra = np.stack([idx[name] for name in INDEX_NAMES], axis=-1)
        return np.concatenate([x, extra], axis=-1).astype(np.float32)
    raise ValueError(f"Reprezentare necunoscută: {representation}")


def valid_pixels(x: np.ndarray) -> np.ndarray:
    """H x W boolean: False pentru pixelii zero în toate benzile (goluri, vezi data card)."""
    return np.any(x != 0, axis=-1)


def has_relief(x: np.ndarray) -> bool:
    """False pentru patch-urile cu DEM constant (115 în train, fără alunecări)."""
    dem = x[..., DEM]
    return bool(dem.max() > dem.min())
