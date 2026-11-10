"""CP2, etapa 1: descrie cele două reprezentări și cât separă fiecare trăsătură clasele.

Pentru fiecare split cu măști se eșantionează pixeli valizi (până la 100 de alunecare și
300 de fundal per patch, seed fix). Pe eșantion se calculează, pe trăsătură, media în și în
afara măștii și AUC-ul trăsăturii singure (Mann-Whitney). AUC = 0,5 înseamnă nicio
separare; sub 0,5 înseamnă că valoarea scade pe alunecări (de ex. NDVI).
Eșantionarea stratificată schimbă prevalența, dar nu și AUC-ul.

Rulare, din rădăcina repo-ului:  python scripts/describe_representations.py
Rezultate: reports/representation_stats.csv, reports/representation_summary.json,
           reports/figures/representation_auc.png, reports/figures/features_train_241.png
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from l4s.data import image_ids, load_image, load_mask  # noqa: E402
from l4s.features import REPRESENTATIONS, feature_cube, has_relief, indices, valid_pixels  # noqa: E402
from l4s.metrics import auc  # noqa: E402

POS_PER_PATCH, NEG_PER_PATCH = 100, 300
EVAL_SPLITS = ["train", "validation"]  # testul nu se folosește la alegerea reprezentării


def sample_split(data: Path, split: str, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, dict]:
    """Întoarce trăsăturile s2_topo eșantionate (N x 18), etichetele și câteva numărători."""
    features, labels = [], []
    counts = {"patches": 0, "no_relief_patches": 0, "invalid_pixels": 0}
    for patch_id in image_ids(data, split):
        x = load_image(data, split, patch_id)
        mask = load_mask(data, split, patch_id)
        counts["patches"] += 1
        counts["no_relief_patches"] += not has_relief(x)
        valid = valid_pixels(x)
        counts["invalid_pixels"] += int((~valid).sum())
        cube = feature_cube(x, "s2_topo").reshape(-1, len(REPRESENTATIONS["s2_topo"]))
        flat_mask, flat_valid = mask.reshape(-1), valid.reshape(-1)
        for cls, k in [(1, POS_PER_PATCH), (0, NEG_PER_PATCH)]:
            idx = np.flatnonzero((flat_mask == cls) & flat_valid)
            if len(idx) > k:
                idx = rng.choice(idx, k, replace=False)
            features.append(cube[idx])
            labels.append(np.full(len(idx), cls, dtype=np.uint8))
    return np.vstack(features), np.concatenate(labels), counts


def plot_patch_features(data: Path, patch_id: int, out: Path) -> Path:
    x = load_image(data, "train", patch_id)
    mask = load_mask(data, "train", patch_id)
    rgb = x[..., [3, 2, 1]]
    low, high = np.percentile(rgb, [1, 99])
    panels = [("Sentinel-2 RGB", np.clip((rgb - low) / (high - low), 0, 1), None)]
    panels += [(name, values, "RdYlGn" if name != "BRIGHT" else "gray") for name, values in indices(x).items()]
    panels += [("Pantă", x[..., 12], "viridis"), ("DEM", x[..., 13], "viridis"), ("Mască", mask, "magma")]

    fig, axes = plt.subplots(2, 4, figsize=(10.4, 5.6), constrained_layout=True)
    for ax, (title, image, cmap) in zip(axes.flat, panels):
        shown = ax.imshow(image, cmap=cmap, interpolation="nearest")
        if cmap and title != "Mască":
            fig.colorbar(shown, ax=ax, shrink=0.8)
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"Reprezentarea A pe train, image_{patch_id}.h5 (indici din benzi scalate; B = doar pantă și DEM)", fontsize=10)
    path = out / f"features_train_{patch_id}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return path


def plot_auc(rows: list[dict], out: Path) -> Path:
    names = REPRESENTATIONS["s2_topo"]
    fig, ax = plt.subplots(figsize=(10.4, 3.6), constrained_layout=True)
    width = 0.4
    xs = np.arange(len(names))
    for offset, split, color in [(-width / 2, "train", "#3b6ea8"), (width / 2, "validation", "#d98b2b")]:
        values = [next(r["auc"] for r in rows if r["split"] == split and r["feature"] == n) for n in names]
        ax.bar(xs + offset, values, width, label=split, color=color)
    ax.axhline(0.5, color="black", lw=0.8, ls="--", label="fără separare (0,5)")
    topo_start = names.index("SLOPE")
    ax.axvspan(topo_start - 0.5, topo_start + 1.5, color="#7a9a5b", alpha=0.15, label="reprezentarea B (topo)")
    ax.set_xticks(xs, names, rotation=45, ha="right")
    ax.set_ylim(0, 1.15)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_ylabel("AUC trăsătură singură")
    ax.set_title("Cât separă fiecare trăsătură pixelii de alunecare de fundal (pixeli eșantionați, seed 42)", fontsize=10)
    ax.legend(fontsize=8, ncol=4, loc="upper center", frameon=False)
    path = out / "representation_auc.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--out", type=Path, default=ROOT / "reports")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patch", type=int, default=241)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    names = REPRESENTATIONS["s2_topo"]
    rows, summary = [], {"seed": args.seed, "representations": REPRESENTATIONS,
                         "pixels_per_patch": {"landslide": POS_PER_PATCH, "background": NEG_PER_PATCH}, "splits": {}}

    for split in EVAL_SPLITS:
        X, y, counts = sample_split(args.data, split, rng)
        summary["splits"][split] = {**counts, "sampled_landslide": int(y.sum()), "sampled_background": int((y == 0).sum())}
        for j, name in enumerate(names):
            v = X[:, j].astype(np.float64)
            rows.append({
                "split": split,
                "feature": name,
                "representation": "A+B" if name in REPRESENTATIONS["topo"] else "A",
                "mean_landslide": float(v[y == 1].mean()),
                "mean_background": float(v[y == 0].mean()),
                "auc": auc(v, y),
            })
        print(f"{split}: {counts['patches']} patch-uri, {len(y)} pixeli eșantionați")

    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "representation_stats.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        for r in rows:
            writer.writerow({k: f"{v:.4f}" if isinstance(v, float) else v for k, v in r.items()})
    summary["top_features_by_separation"] = {
        split: sorted(({"feature": r["feature"], "auc": round(r["auc"], 4)} for r in rows if r["split"] == split),
                      key=lambda d: -abs(d["auc"] - 0.5))[:6]
        for split in EVAL_SPLITS
    }
    (args.out / "representation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    figures = args.out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    print(plot_auc(rows, figures))
    print(plot_patch_features(args.data, args.patch, figures))
    print(json.dumps(summary["top_features_by_separation"], indent=2))


if __name__ == "__main__":
    main()
