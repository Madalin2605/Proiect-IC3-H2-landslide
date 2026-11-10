"""CP2, etapa 2: construiește și verifică fold-urile spațiale de pe train.

Pași:
1. Verifică din date așezarea pe grile din l4s.splits: marginea DEM a lui i trebuie să
   continue marginea vecinului din dreapta (i+1) și a celui de dedesubt (i+lățime).
2. Verifică dacă grilele mici G2-G5 se lipesc de G1 (potriviri stricte de margine).
3. Construiește fold-urile (benzi din G1 + grilele mici), cu buffer, fără patch-urile cu DEM constant.
4. Măsoară leakage-ul: ce parte din patch-urile de test are un vecin confirmat în antrenare,
   pentru split aleator, pentru blocuri fără buffer și pentru protocolul ales.

Rulare, din rădăcina repo-ului:  python scripts/make_splits.py
Rezultate: reports/splits.json, reports/figures/spatial_folds.png
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from l4s.data import DEM, image_ids, load_image, load_mask  # noqa: E402
from l4s.features import has_relief, valid_pixels  # noqa: E402
from l4s.splits import DEFAULT_BUFFER, GRIDS, grid_of, spatial_folds  # noqa: E402

STRICT_GAP = 0.01  # potrivire strictă de margine, pentru legătura dintre grile
MIN_CONTINUITY = 0.95
FOLD_COLORS = ["#3b6ea8", "#d98b2b", "#5a9e5a", "#b04a5a", "#8a6bb1"]


def load_patches(data: Path) -> tuple[list[int], dict]:
    ids = image_ids(data, "train")
    info = {k: [] for k in ["top", "bottom", "left", "right", "flat", "landslide", "valid"]}
    for patch_id in ids:
        x = load_image(data, "train", patch_id)
        dem = x[..., DEM]
        info["top"].append(dem[0])
        info["bottom"].append(dem[-1])
        info["left"].append(dem[:, 0])
        info["right"].append(dem[:, -1])
        info["flat"].append(not has_relief(x))
        info["landslide"].append(int(load_mask(data, "train", patch_id).sum()))
        info["valid"].append(int(valid_pixels(x).sum()))
    return ids, {k: np.array(v) for k, v in info.items()}


def gap(info: dict, edge_a: str, a: int, edge_b: str, b: int) -> float:
    """Diferența medie dintre două margini DEM (ID-urile încep de la 1)."""
    return float(np.abs(info[edge_a][a - 1] - info[edge_b][b - 1]).mean())


def random_threshold(info: dict, ids: list[int], rng: np.random.Generator) -> float:
    """Percentila 5 a diferenței de margine la perechi aleatoare (ca în check_structure.py)."""
    usable = [i for i in ids if not info["flat"][i - 1]]
    pairs = rng.choice(usable, size=(len(usable) * 2, 2))
    return float(np.percentile([gap(info, "right", a, "left", b) for a, b in pairs if a != b], 5))


def confirmed_neighbours(info: dict, threshold: float) -> list[tuple[int, int]]:
    """Perechile vecine pe grilă (dreapta, dedesubt) a căror margine DEM chiar continuă."""
    pairs = []
    for grid in GRIDS:
        for i in range(grid.first_id, grid.last_id + 1):
            row, col = grid.position(i)
            candidates = []
            if col + 1 < grid.width:
                candidates.append((i + 1, "right", "left"))
            if row + 1 < grid.rows:
                candidates.append((i + grid.width, "bottom", "top"))
            for j, ea, eb in candidates:
                if not (info["flat"][i - 1] or info["flat"][j - 1]) and gap(info, ea, i, eb, j) < threshold:
                    pairs.append((i, j))
    return pairs


def grid_continuity(info: dict, threshold: float) -> dict:
    result = {}
    for grid in GRIDS:
        horizontal, vertical = [], []
        for i in range(grid.first_id, grid.last_id + 1):
            row, col = grid.position(i)
            if col + 1 < grid.width and not (info["flat"][i - 1] or info["flat"][i]):
                horizontal.append(gap(info, "right", i, "left", i + 1) < threshold)
            j = i + grid.width
            if row + 1 < grid.rows and not (info["flat"][i - 1] or info["flat"][j - 1]):
                vertical.append(gap(info, "bottom", i, "top", j) < threshold)
        result[grid.name] = {
            "ids": [grid.first_id, grid.last_id],
            "width": grid.width,
            "rows": grid.rows,
            "horizontal_continuity": float(np.mean(horizontal)) if horizontal else None,
            "vertical_continuity": float(np.mean(vertical)) if vertical else None,
        }
    return result


def cross_grid_matches(info: dict) -> dict:
    """Câte patch-uri din G2-G5 au vreo margine aproape identică cu o margine din G1."""
    g1 = GRIDS[0]
    g1_idx = np.array([i - 1 for i in range(g1.first_id, g1.last_id + 1) if not info["flat"][i - 1]])
    small_idx = np.arange(g1.last_id, GRIDS[-1].last_id)
    best = np.full(len(small_idx), np.inf)
    for ea, eb in [("bottom", "top"), ("top", "bottom"), ("right", "left"), ("left", "right")]:
        a, b = info[ea][small_idx], info[eb][g1_idx]
        for s in range(0, len(a), 100):
            best[s:s + 100] = np.minimum(best[s:s + 100], np.abs(a[s:s + 100, None] - b[None]).mean(2).min(1))
    return {"strict_gap": STRICT_GAP, "small_grid_patches": len(small_idx),
            "patches_with_match_in_G1": int((best < STRICT_GAP).sum())}


def leakage_share(folds: list[dict], pairs: list[tuple[int, int]]) -> list[float]:
    """Pe fold: ce parte din patch-urile de test are un vecin confirmat în setul de antrenare."""
    adjacency = {}
    for a, b in pairs:
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)
    shares = []
    for fold in folds:
        train = set(fold["train_ids"])
        test = fold["test_ids"]
        shares.append(float(np.mean([bool(adjacency.get(i, set()) & train) for i in test])))
    return shares


def random_folds(usable: list[int], sizes: list[int], rng: np.random.Generator) -> list[dict]:
    order = rng.permutation(usable)
    folds, start = [], 0
    for size in sizes:
        test = sorted(int(i) for i in order[start:start + size])
        test_set = set(test)
        folds.append({"test_ids": test, "train_ids": [i for i in usable if i not in test_set]})
        start += size
    return folds


def plot_folds(folds: list[dict], info: dict, out: Path, show_buffer_for: int = 2) -> Path:
    # Canvas: G1 în stânga, grilele mici în dreapta.
    origins = {"G1": (0, 0), "G5": (0, 43), "G2": (21, 43), "G4": (31, 43), "G3": (43, 43)}
    shape = (GRIDS[0].rows, 43 + 41)
    fold_map, fraction_map = np.full(shape, np.nan), np.full(shape, np.nan)
    pixels = 128 * 128
    fold_of_id = {i: f["fold"] for f in folds for i in f["test_ids"]}

    def cell(i):
        grid = grid_of(i)
        r, c = grid.position(i)
        r0, c0 = origins[grid.name]
        return r0 + r, c0 + c

    # Categorii pe hartă: 0 = DEM constant (exclus), 1-5 = fold, 6 = buffer pentru fold-ul ilustrat.
    buffer_ids = set(next(f for f in folds if f["fold"] == show_buffer_for)["buffer_ids"])
    for grid in GRIDS:
        for i in range(grid.first_id, grid.last_id + 1):
            r, c = cell(i)
            fraction_map[r, c] = info["landslide"][i - 1] / pixels * 100
            if info["flat"][i - 1]:
                fold_map[r, c] = 0
            elif i in buffer_ids:
                fold_map[r, c] = 6
            else:
                fold_map[r, c] = fold_of_id[i]

    colors = ["black", *FOLD_COLORS, "#c8c8c8"]
    labels = ["DEM constant (exclus)", *(f"fold {k}" for k in range(1, 6)),
              f"buffer când fold {show_buffer_for} e test"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    axes[0].imshow(fold_map, cmap=ListedColormap(colors), vmin=-0.5, vmax=6.5, interpolation="nearest")
    handles = [Patch(color=c, label=lab) for c, lab in zip(colors, labels)]
    axes[0].legend(handles=handles, fontsize=7, loc="lower right", framealpha=0.9)
    axes[0].set_title("Fold-uri spațiale pe train (un pătrat = un patch de 1,28 km)", fontsize=9)

    shown = axes[1].imshow(fraction_map, cmap="magma", interpolation="nearest")
    fig.colorbar(shown, ax=axes[1], shrink=0.8, label="pixeli alunecare în patch (%)")
    axes[1].set_title("Unde sunt alunecările pe aceleași grile", fontsize=9)
    for ax in axes:
        for name, (r0, c0) in origins.items():
            ax.text(c0, r0 - 0.8, name, fontsize=7, va="bottom")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylim(shape[0] - 0.5, -2.5)
    path = out / "spatial_folds.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--out", type=Path, default=ROOT / "reports")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--buffer", type=int, default=DEFAULT_BUFFER)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    ids, info = load_patches(args.data)
    threshold = random_threshold(info, ids, rng)

    continuity = grid_continuity(info, threshold)
    for name, c in continuity.items():
        for key in ["horizontal_continuity", "vertical_continuity"]:
            if c[key] is not None and c[key] < MIN_CONTINUITY:
                sys.exit(f"Grila {name} nu se confirmă: {key} = {c[key]:.2f}. Verifică l4s/splits.py.")

    excluded = {i for i in ids if info["flat"][i - 1]}
    folds = spatial_folds(ids, excluded=excluded, buffer=args.buffer)
    no_buffer = spatial_folds(ids, excluded=excluded, buffer=0)
    usable = [i for i in ids if i not in excluded]
    shuffled = random_folds(usable, [len(f["test_ids"]) for f in folds], rng)
    pairs = confirmed_neighbours(info, threshold)

    leakage = {
        "definition": "parte din patch-urile de test cu cel puțin un vecin confirmat (margine DEM continuă) în antrenare",
        "random_split": leakage_share(shuffled, pairs),
        "blocks_without_buffer": leakage_share(no_buffer, pairs),
        "blocks_with_buffer": leakage_share(folds, pairs),
    }

    pixels = 128 * 128
    fold_rows = []
    for f in folds:
        test = np.array(f["test_ids"]) - 1
        fold_rows.append({
            "fold": f["fold"],
            "region": "G1, rândurile " + "-".join(str(grid_of(i).position(i)[0] + 1) for i in (f["test_ids"][0], f["test_ids"][-1]))
                      if f["fold"] <= 4 else "G2-G5",
            "test_patches": len(f["test_ids"]),
            "train_patches": len(f["train_ids"]),
            "buffer_patches": len(f["buffer_ids"]),
            "test_patches_with_landslide": int((info["landslide"][test] > 0).sum()),
            "test_landslide_pixel_fraction": float(info["landslide"][test].sum() / (len(test) * pixels)),
        })

    result = {
        "seed": args.seed,
        "protocol": {
            "cv": "5 fold-uri spațiale pe train: benzile 1-4 din G1 și grilele G2-G5 ca fold 5",
            "buffer_patches": args.buffer,
            "excluded": "patch-urile cu DEM constant (fără relief, fără alunecări), din toate fold-urile",
            "validation": "245 de patch-uri din alte situri: verificarea alegerilor (prag, hiperparametri)",
            "test": "800 de patch-uri din alte situri: o singură evaluare, la final",
            "metrics": "F1, precizie, recall, IoU pe clasa alunecare din confuzia cumulată pe pixeli; AUC și AP fără prag",
        },
        "edge_threshold": threshold,
        "grids": continuity,
        "cross_grid": cross_grid_matches(info),
        "excluded_ids": sorted(excluded),
        "leakage": leakage,
        "fold_summary": fold_rows,
        "folds": folds,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "splits.json").write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    (args.out / "figures").mkdir(parents=True, exist_ok=True)
    print(plot_folds(folds, info, args.out / "figures"))

    print(json.dumps({k: result[k] for k in ["grids", "cross_grid", "leakage", "fold_summary"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
