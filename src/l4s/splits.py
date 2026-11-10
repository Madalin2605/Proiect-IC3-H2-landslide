"""Protocolul de evaluare din CP2: fold-uri spațiale pe train, cu buffer.

Patch-urile de train sunt așezate pe grile, rând cu rând (verificat din continuitatea
marginilor DEM în scripts/make_splits.py):

    G1: ID 1-3034,    41 coloane x 74 rânduri
    G2: ID 3035-3154, 15 x 8
    G3: ID 3155-3195, 41 x 1
    G4: ID 3196-3305, 11 x 10
    G5: ID 3306-3799, 26 x 19

Vecinul din dreapta lui i este i+1, iar cel de dedesubt este i+lățime.
Fold-urile 1-4 sunt benzi orizontale de rânduri din G1 (blocuri contigue de ID-uri),
iar fold-ul 5 conține grilele G2-G5. Când un fold e test, patch-urile din aceeași grilă
aflate la cel mult `buffer` patch-uri de el (inclusiv pe diagonală) nu intră la antrenare.

Validarea (245) și testul oficial (800) vin din alte situri și rămân separate:
validarea pentru verificarea alegerilor, testul o singură dată, la final.
"""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Grid:
    name: str
    first_id: int
    last_id: int
    width: int

    @property
    def rows(self) -> int:
        return (self.last_id - self.first_id + 1) // self.width

    def contains(self, patch_id: int) -> bool:
        return self.first_id <= patch_id <= self.last_id

    def position(self, patch_id: int) -> tuple[int, int]:
        offset = patch_id - self.first_id
        return offset // self.width, offset % self.width


GRIDS = [
    Grid("G1", 1, 3034, 41),
    Grid("G2", 3035, 3154, 15),
    Grid("G3", 3155, 3195, 41),
    Grid("G4", 3196, 3305, 11),
    Grid("G5", 3306, 3799, 26),
]
G1_BANDS = 4
DEFAULT_BUFFER = 1


def grid_of(patch_id: int) -> Grid:
    for grid in GRIDS:
        if grid.contains(patch_id):
            return grid
    raise ValueError(f"ID {patch_id} nu aparține niciunei grile de train")


def fold_of(patch_id: int) -> int:
    """Fold-ul 1-4 pentru benzile din G1, 5 pentru celelalte grile."""
    grid = grid_of(patch_id)
    if grid.name != "G1":
        return G1_BANDS + 1
    row, _ = grid.position(patch_id)
    bands = np.array_split(np.arange(grid.rows), G1_BANDS)
    return next(k + 1 for k, band in enumerate(bands) if row in band)


def neighbours(patch_id: int, distance: int = 1) -> list[int]:
    """ID-urile din aceeași grilă aflate la cel mult `distance` patch-uri (pe rând, coloană sau diagonală)."""
    grid = grid_of(patch_id)
    row, col = grid.position(patch_id)
    found = []
    for dr in range(-distance, distance + 1):
        for dc in range(-distance, distance + 1):
            r, c = row + dr, col + dc
            if (dr or dc) and 0 <= r < grid.rows and 0 <= c < grid.width:
                found.append(grid.first_id + r * grid.width + c)
    return found


def spatial_folds(ids: list[int], excluded: set[int] = frozenset(), buffer: int = DEFAULT_BUFFER) -> list[dict]:
    """Pentru fiecare fold întoarce ID-urile de test, de antrenare și cele scoase ca buffer."""
    usable = [i for i in ids if i not in excluded]
    by_fold = {}
    for i in usable:
        by_fold.setdefault(fold_of(i), []).append(i)

    folds = []
    for k in sorted(by_fold):
        test = set(by_fold[k])
        near = {n for i in test for n in neighbours(i, buffer)} - test if buffer else set()
        train = [i for i in usable if i not in test and i not in near]
        folds.append({
            "fold": k,
            "test_ids": sorted(test),
            "train_ids": train,
            "buffer_ids": sorted(near & set(usable)),
        })
    return folds


def load_folds(path: Path) -> list[dict]:
    """Citește fold-urile salvate de scripts/make_splits.py."""
    return json.loads(Path(path).read_text(encoding="utf-8"))["folds"]
