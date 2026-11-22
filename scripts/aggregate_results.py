"""CP3, pasul 5: tabelul principal (model v1 vs. baseline, pe seed-uri, cu intervale).

Citește rulările din reports/CP3/runs/{model}/seed{s}/{fold}/ (patches.csv + run.json), scrise
de scripts/train_unet.py și scripts/run_rf_seeds.py, și calculează:

1. F1 (și precizie, recall, IoU, AUC, AP) pe fiecare evaluare, pe fiecare seed și media lor.
     CV G1      media pe fold-urile 1-4 x 3 seed-uri (generalizare în același sit)
     Fold 5     grilele G2-G5, raportat separat (transfer către altă zonă)
     Validare / test oficial: situri noi, modele antrenate pe tot train-ul
2. Interval de încredere 95% pentru F1, prin bootstrap pe clustere (1000 de reeșantionări):
     în train, clusterul e un rând al grilei (patch-urile vecine sunt corelate spațial);
     în validare / test nu se știe așezarea, deci clusterul e patch-ul (interval mai optimist).
   Aceleași reeșantionări se folosesc pentru toate modelele și seed-urile, deci comparațiile
   sunt pereche. Variabilitatea între seed-uri se raportează separat (min-max).
3. ΔF1 pereche între modele (ex. U-Net - RF), cu interval 95% și fracția reeșantionărilor
   în care Δ <= 0 (p unilateral aproximativ).
4. Metrica pentru decizie (clientul verifică patch-uri de ~1,28 km, nu pixeli): patch-urile
   se ordonează după scorul mediu prezis, iar „captura@k” = ce fracție din pixelii de
   alunecare se află în primele k% patch-uri. O ordonare la întâmplare dă în medie k%.
   Plus AUC la nivel de patch (patch cu / fără alunecare).

Ieșiri:
  reports/CP3/results_v1_H2.csv           rânduri pe rulare + rânduri sumar + rânduri Δ (livrabilul CP3)
  reports/CP3/results_v1.md               tabelele, pentru docs/results_H2.md și slide-uri
  reports/CP3/figures/results_v1_f1.png   F1 cu intervale + captura@20%

Rulare, din rădăcina repo-ului:  python scripts/aggregate_results.py      (~1-2 minute, procesor)
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
from l4s.splits import grid_of  # noqa: E402

MODELS = [  # (cheie, etichetă, reprezentare)
    ("unet_s2_topo", "U-Net (A)", "A"),
    ("rf_s2_topo", "RF pe pixeli (A)", "A"),
    ("rule_ndvi", "Prag NDVI (A)", "A"),
    ("unet_topo", "U-Net (B)", "B"),
    ("rf_topo", "RF pe pixeli (B)", "B"),
    ("rule_slope", "Prag pantă (B)", "B"),
]
LABEL = {k: lab for k, lab, _ in MODELS}
REP = {k: r for k, _, r in MODELS}
COMPARISONS = [  # (a, b): Δ = a - b
    ("unet_s2_topo", "rf_s2_topo"),
    ("unet_s2_topo", "rule_ndvi"),
    ("unet_topo", "rf_topo"),
    ("unet_topo", "rule_slope"),
    ("unet_s2_topo", "unet_topo"),
]
SUMMARIES = {  # evaluare sumar -> evaluările din care se compune
    "cv_g1": ["cv_fold_1", "cv_fold_2", "cv_fold_3", "cv_fold_4"],
    "cv_fold_5": ["cv_fold_5"],
    "validation": ["validation"],
    "test": ["test"],
}
EVAL_LABEL = {"cv_g1": "CV G1 (fold 1-4)", "cv_fold_5": "Fold 5 (G2-G5)", "validation": "Validare", "test": "Test oficial"}
TOP_K = (0.10, 0.20)
LITERATURE_TEST_F1 = 0.5992  # U-Net, Ghorbanzadeh et al. (2022b)
METRICS = ["f1", "precision", "recall", "iou", "auc", "ap"]


# ---------- citire ----------

def load_runs(runs_dir: Path) -> dict:
    """runs[model][seed][evaluation] = {"metrics", "threshold", "best_epoch", "patches": {coloană: array}}."""
    runs = {}
    for model, _, _ in MODELS:
        for run_json in sorted((runs_dir / model).glob("seed*/*/run.json")):
            info = json.loads(run_json.read_text(encoding="utf-8"))
            with open(run_json.parent / "patches.csv", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            for evaluation, metrics in info["results"].items():
                sel = [r for r in rows if r["split"] == evaluation]
                sel.sort(key=lambda r: int(r["patch_id"]))
                patches = {c: np.array([float(r[c]) for r in sel]) for c in
                           ("patch_id", "tp", "fp", "fn", "n_valid", "n_pos", "prob_mean")}
                runs.setdefault(model, {}).setdefault(int(info["seed"]), {})[evaluation] = {
                    "metrics": metrics, "threshold": info["threshold"], "best_epoch": info.get("best_epoch"),
                    "patches": patches,
                }
    return runs


def cluster_ids(evaluation: str, patch_ids: np.ndarray) -> np.ndarray:
    if not evaluation.startswith("cv_fold_"):
        return np.arange(len(patch_ids))
    keys = []
    for pid in patch_ids.astype(int):
        g = grid_of(int(pid))
        keys.append((g.name, g.position(int(pid))[0]))
    _, inverse = np.unique(np.array([f"{a}-{b:03d}" for a, b in keys]), return_inverse=True)
    return inverse


def bootstrap_weights(clusters: np.ndarray, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """n_boot x n_patches: de câte ori apare fiecare patch într-o reeșantionare pe clustere."""
    n_clusters = clusters.max() + 1
    draws = rng.integers(n_clusters, size=(n_boot, n_clusters))
    per_cluster = np.stack([np.bincount(d, minlength=n_clusters) for d in draws])
    return per_cluster[:, clusters].astype(np.float64)


# ---------- metrici pe reeșantionări ----------

def f1_from(p: dict, W: np.ndarray | None = None) -> np.ndarray | float:
    tp, fp, fn = p["tp"], p["fp"], p["fn"]
    if W is None:
        return 2 * tp.sum() / max(2 * tp.sum() + fp.sum() + fn.sum(), 1)
    tp, fp, fn = W @ tp, W @ fp, W @ fn
    return 2 * tp / np.maximum(2 * tp + fp + fn, 1)


def capture_at(p: dict, k: float, W: np.ndarray | None = None) -> np.ndarray | float:
    """Fracția din pixelii de alunecare aflați în primele k% patch-uri, ordonate după scorul mediu."""
    order = np.argsort(-p["prob_mean"], kind="mergesort")
    pos = p["n_pos"][order]
    if W is None:
        n_top = int(round(k * len(pos)))
        return pos[:n_top].sum() / max(pos.sum(), 1)
    w = W[:, order]
    cum = np.cumsum(w, axis=1)
    inside = cum <= k * w.sum(axis=1, keepdims=True)
    return (w * inside * pos).sum(axis=1) / np.maximum((w * pos).sum(axis=1), 1)


def patch_auc(p: dict) -> float:
    """AUC la nivel de patch: scorul mediu separă patch-urile cu alunecare de cele fără?"""
    y = p["n_pos"] > 0
    s = p["prob_mean"]
    if y.all() or not y.any():
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    for v in np.unique(s):  # ranguri medii la egalitate
        tie = s == v
        if tie.sum() > 1:
            ranks[tie] = ranks[tie].mean()
    n1, n0 = y.sum(), (~y).sum()
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# ---------- agregare ----------

def aggregate(runs: dict, n_boot: int, seed: int) -> tuple[list[dict], dict, dict]:
    rng = np.random.default_rng(seed)
    seeds = sorted(next(iter(runs.values())))
    evaluations = sorted({e for m in runs.values() for s in m.values() for e in s})

    # aceleași reeșantionări pentru toate modelele și seed-urile
    weights = {}
    for evaluation in evaluations:
        ref = runs[MODELS[0][0]][seeds[0]][evaluation]["patches"]
        for model in runs:
            for s in seeds:
                assert np.array_equal(runs[model][s][evaluation]["patches"]["patch_id"], ref["patch_id"]), \
                    f"patch-uri diferite: {model} seed {s} {evaluation}"
        weights[evaluation] = bootstrap_weights(cluster_ids(evaluation, ref["patch_id"]), n_boot, rng)

    run_rows, boot, summary = [], {}, {}
    for model, _, _ in MODELS:
        for s in seeds:
            for evaluation in evaluations:
                r = runs[model][s][evaluation]
                p, W = r["patches"], weights[evaluation]
                run_rows.append({
                    "row_type": "run", "model": model, "label": LABEL[model], "representation": REP[model],
                    "evaluation": evaluation, "seed": s, **{m: r["metrics"][m] for m in METRICS},
                    "threshold": r["threshold"], "best_epoch": r["best_epoch"], "n_patches": len(p["tp"]),
                    **{f"capture_top{int(k * 100)}": capture_at(p, k) for k in TOP_K},
                    "patch_auc": patch_auc(p),
                })
                boot[(model, s, evaluation)] = {"f1": f1_from(p, W),
                                                **{f"capture_top{int(k * 100)}": capture_at(p, k, W) for k in TOP_K}}

    def combine(model: str, parts: list[str], key: str) -> tuple[float, np.ndarray, list[float]]:
        """Media pe seed-uri și evaluările componente: estimarea, reeșantionările și media per seed."""
        point = [[next(r[key] for r in run_rows if r["model"] == model and r["seed"] == s and r["evaluation"] == e)
                  for e in parts] for s in seeds]
        samples = np.mean([[boot[(model, s, e)][key] for e in parts] for s in seeds], axis=(0, 1))
        per_seed = [float(np.mean(v)) for v in point]
        return float(np.mean(point)), samples, per_seed

    summary_rows = []
    for model, _, _ in MODELS:
        for name, parts in SUMMARIES.items():
            rows = [r for r in run_rows if r["model"] == model and r["evaluation"] in parts]
            row = {"row_type": "summary", "model": model, "label": LABEL[model], "representation": REP[model],
                   "evaluation": name, "seed": "mean", **{m: float(np.mean([r[m] for r in rows])) for m in METRICS},
                   "n_runs": len(rows)}
            for key in ["f1"] + [f"capture_top{int(k * 100)}" for k in TOP_K]:
                point, samples, per_seed = combine(model, parts, key)
                summary[(model, name, key)] = samples
                row[key] = point
                row[f"{key}_ci_low"], row[f"{key}_ci_high"] = np.percentile(samples, [2.5, 97.5]).tolist()
                if key == "f1":
                    row["f1_seed_min"], row["f1_seed_max"] = min(per_seed), max(per_seed)
                    row["f1_sd_runs"] = float(np.std([r["f1"] for r in rows], ddof=1))
            row["patch_auc"] = float(np.nanmean([r["patch_auc"] for r in rows]))
            summary_rows.append(row)

    delta_rows = []
    for a, b in COMPARISONS:
        for name in SUMMARIES:
            for key in ("f1", "capture_top20"):
                pa = next(r[key] for r in summary_rows if r["model"] == a and r["evaluation"] == name)
                pb = next(r[key] for r in summary_rows if r["model"] == b and r["evaluation"] == name)
                d = summary[(a, name, key)] - summary[(b, name, key)]
                lo, hi = np.percentile(d, [2.5, 97.5]).tolist()
                delta_rows.append({"row_type": "delta", "model": f"{a} - {b}", "label": f"{LABEL[a]} - {LABEL[b]}",
                                   "evaluation": name, "seed": "mean", "metric": key, "delta": pa - pb,
                                   "delta_ci_low": lo, "delta_ci_high": hi, "p_delta_le_0": float((d <= 0).mean())})

    epochs = {model: [r["best_epoch"] for r in run_rows if r["model"] == model and r["best_epoch"] is not None
                      and r["evaluation"] in ("cv_fold_1", "cv_fold_2", "cv_fold_3", "cv_fold_4", "cv_fold_5", "test")]
              for model in runs if model.startswith("unet")}
    return run_rows + summary_rows + delta_rows, {"seeds": seeds, "best_epochs": epochs}, summary


# ---------- ieșiri ----------

FIELDS = ["row_type", "model", "label", "representation", "evaluation", "seed", "metric",
          "f1", "f1_ci_low", "f1_ci_high", "f1_seed_min", "f1_seed_max", "f1_sd_runs",
          "precision", "recall", "iou", "auc", "ap",
          "capture_top10", "capture_top10_ci_low", "capture_top10_ci_high",
          "capture_top20", "capture_top20_ci_low", "capture_top20_ci_high", "patch_auc",
          "delta", "delta_ci_low", "delta_ci_high", "p_delta_le_0",
          "threshold", "best_epoch", "n_patches", "n_runs"]


def write_csv(rows: list[dict], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8-sig") as f:  # cu BOM, ca Excel să citească diacriticele corect
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})


def pct(v: float, digits: int = 1) -> str:
    return f"{100 * v:.{digits}f}".replace(".", ",")


def dec(v: float) -> str:
    return f"{v:.3f}".replace(".", ",")


def write_markdown(rows: list[dict], meta: dict, path: Path, n_boot: int) -> None:
    summ = {(r["model"], r["evaluation"]): r for r in rows if r["row_type"] == "summary"}
    deltas = [r for r in rows if r["row_type"] == "delta"]
    seeds = ", ".join(str(s) for s in meta["seeds"])

    def cell(model, ev, key="f1"):
        r = summ[(model, ev)]
        return f"{pct(r[key])} [{pct(r[key + '_ci_low'])}, {pct(r[key + '_ci_high'])}]"

    L = ["# Rezultate v1 (CP3): model vs. baseline, pe seed-uri, cu intervale", "",
         f"Generat de `scripts/aggregate_results.py` din `reports/CP3/runs/`. Seed-uri: {seeds}. "
         "F1 pe clasa alunecare, din confuzia cumulată pe pixelii valizi; valoarea = media pe seed-uri "
         "(la CV G1: pe fold-urile 1-4 x seed-uri). În paranteze: interval 95% bootstrap "
         f"({n_boot} reeșantionări; în train pe rânduri de grilă, în validare/test pe patch-uri).", "",
         "## Tabelul principal: F1 (%) [IC 95%]", "",
         "| Model | " + " | ".join(EVAL_LABEL[e] for e in SUMMARIES) + " | AP test |",
         "|---|" + "---:|" * (len(SUMMARIES) + 1)]
    for model, label, _ in MODELS:
        L.append(f"| {label} | " + " | ".join(cell(model, e) for e in SUMMARIES) + f" | {dec(summ[(model, 'test')]['ap'])} |")
    L += ["", f"Referință: U-Net din literatură pe testul oficial, F1 = {pct(LITERATURE_TEST_F1)}%. "
          "„Totul negativ” are F1 = 0.", "",
          "## Variabilitatea între seed-uri: F1 (%) min-max", "",
          "| Model | " + " | ".join(EVAL_LABEL[e] for e in SUMMARIES) + " |", "|---|" + "---:|" * len(SUMMARIES)]
    for model, label, _ in MODELS:
        L.append(f"| {label} | " + " | ".join(
            f"{pct(summ[(model, e)]['f1_seed_min'])}-{pct(summ[(model, e)]['f1_seed_max'])}" for e in SUMMARIES) + " |")
    L += ["", "La CV G1, min-max e între mediile pe fold-uri ale fiecărui seed; abaterea standard pe cele 12 rulări: " +
          "; ".join(f"{label} {pct(summ[(m, 'cv_g1')]['f1_sd_runs'])}" for m, label, _ in MODELS) + ".", "",
          "## Precizie, recall, AUC, AP (medii pe seed-uri)", "",
          "| Model | Evaluare | Precizie | Recall | IoU | AUC | AP |", "|---|---|---:|---:|---:|---:|---:|"]
    for model, label, _ in MODELS:
        for e in SUMMARIES:
            r = summ[(model, e)]
            L.append(f"| {label} | {EVAL_LABEL[e]} | {pct(r['precision'])} | {pct(r['recall'])} | {pct(r['iou'])} | "
                     f"{dec(r['auc'])} | {dec(r['ap'])} |")
    L += ["", "## Comparații pereche: ΔF1 (puncte) [IC 95%], p = fracția reeșantionărilor cu Δ ≤ 0", "",
          "| Comparație | " + " | ".join(EVAL_LABEL[e] for e in SUMMARIES) + " |", "|---|" + "---:|" * len(SUMMARIES)]
    for a, b in COMPARISONS:
        cells = []
        for e in SUMMARIES:
            d = next(r for r in deltas if r["model"] == f"{a} - {b}" and r["evaluation"] == e and r["metric"] == "f1")
            cells.append(f"{100 * d['delta']:+.1f} [{100 * d['delta_ci_low']:+.1f}, {100 * d['delta_ci_high']:+.1f}], "
                         f"p={d['p_delta_le_0']:.3f}".replace(".", ","))
        L.append(f"| {LABEL[a]} − {LABEL[b]} | " + " | ".join(cells) + " |")
    L += ["", "## Metrica pentru decizie: captura alunecărilor verificând primele 10% / 20% patch-uri (%)", "",
          "Patch-urile (~1,28 km) se ordonează după scorul mediu prezis; captura = fracția pixelilor de alunecare "
          "din patch-urile verificate. Ordonarea la întâmplare dă în medie 10%, respectiv 20%. "
          "AUC patch = separarea patch-urilor cu / fără alunecare.", "",
          "| Model | Evaluare | Captura top 10% | Captura top 20% [IC 95%] | AUC patch |", "|---|---|---:|---:|---:|"]
    for model, label, _ in MODELS:
        for e in SUMMARIES:
            r = summ[(model, e)]
            L.append(f"| {label} | {EVAL_LABEL[e]} | {pct(r['capture_top10'])} | {cell(model, e, 'capture_top20')} | "
                     f"{dec(r['patch_auc'])} |")
    L += ["", "ΔCaptura top 20% (puncte) [IC 95%]:", ""]
    for a, b in COMPARISONS:
        parts = []
        for e in SUMMARIES:
            d = next(r for r in deltas if r["model"] == f"{a} - {b}" and r["evaluation"] == e
                     and r["metric"] == "capture_top20")
            parts.append(f"{EVAL_LABEL[e]} {100 * d['delta']:+.1f} [{100 * d['delta_ci_low']:+.1f}, "
                         f"{100 * d['delta_ci_high']:+.1f}]".replace(".", ","))
        L.append(f"- {LABEL[a]} − {LABEL[b]}: " + "; ".join(parts))
    L += ["", "## Epoca aleasă la U-Net (validarea internă)", ""]
    for model, ep in meta["best_epochs"].items():
        values, counts = np.unique(ep, return_counts=True)
        L.append(f"- {LABEL[model]}: " + ", ".join(f"epoca {v}: {c}" for v, c in zip(values, counts)) +
                 f" (din {len(ep)} rulări; maximul e 15)")
    L.append("")
    path.write_text("\n".join(L), encoding="utf-8")


def plot(rows: list[dict], path: Path) -> None:
    summ = {(r["model"], r["evaluation"]): r for r in rows if r["row_type"] == "summary"}
    colors = {"cv_g1": "#2a78d6", "cv_fold_5": "#eb6834", "validation": "#1baf7a", "test": "#eda100"}
    xs = np.arange(len(MODELS))
    width = 0.2
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), constrained_layout=True, gridspec_kw={"width_ratios": [1, 1]})
    for ax, key, title, ylim in [
        (axes[0], "f1", "F1 pe pixeli, clasa alunecare (%)", 100),
        (axes[1], "capture_top20", "Alunecări capturate verificând primele 20% patch-uri (%)", 100),
    ]:
        for j, ev in enumerate(SUMMARIES):
            vals = np.array([100 * summ[(m, ev)][key] for m, _, _ in MODELS])
            lo = np.array([100 * summ[(m, ev)][f"{key}_ci_low"] for m, _, _ in MODELS])
            hi = np.array([100 * summ[(m, ev)][f"{key}_ci_high"] for m, _, _ in MODELS])
            bars = ax.bar(xs + (j - 1.5) * width, vals, width * 0.92, color=colors[ev], label=EVAL_LABEL[ev],
                          yerr=[vals - lo, hi - vals], capsize=2, error_kw={"lw": 0.8, "ecolor": "#333333"})
            ax.bar_label(bars, labels=[f"{v:.0f}" for v in vals], fontsize=6.5, padding=2, color="#333333")
        if key == "f1":
            ax.axhline(100 * LITERATURE_TEST_F1, color="#333333", ls="--", lw=0.8,
                       label="U-Net din literatură, test oficial (59,9)")
        else:
            random_line = ax.axhline(20, color="#333333", ls=":", lw=0.8, label="ordonare la întâmplare (20)")
        ax.axvline(2.5, color="#999999", lw=0.6)
        ax.set_xticks(xs, [lab.replace(" (", "\n(") for _, lab, _ in MODELS], fontsize=8)
        ax.set_ylim(0, ylim)
        ax.set_title(title, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e5e5e5", lw=0.6)
        ax.set_axisbelow(True)
    axes[0].legend(fontsize=7.5, loc="upper right", frameon=False, ncol=2)
    axes[1].legend(handles=[random_line], fontsize=7.5, loc="upper right", frameon=False)
    fig.suptitle("CP3: medii pe 3 seed-uri, bare = IC 95% bootstrap. Stânga liniei: reprezentarea A; dreapta: B",
                 fontsize=9, color="#333333")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, default=ROOT / "reports" / "CP3" / "runs")
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "CP3")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    runs = load_runs(args.runs)
    missing = [m for m, _, _ in MODELS if m not in runs]
    if missing:
        raise SystemExit(f"Lipsesc rulările pentru: {missing}")
    counts = {m: sum(len(v) for v in runs[m].values()) for m in runs}
    print("Evaluări citite pe model (seed x evaluare):", counts)

    rows, meta, _ = aggregate(runs, args.n_boot, args.seed)
    write_csv(rows, args.out / "results_v1_H2.csv")
    write_markdown(rows, meta, args.out / "results_v1.md", args.n_boot)
    plot(rows, args.out / "figures" / "results_v1_f1.png")

    print("\nF1 (%) [IC 95%], medii pe seed-uri:")
    print(f"  {'model':<18}" + "".join(f"{EVAL_LABEL[e]:>24}" for e in SUMMARIES))
    for model, label, _ in MODELS:
        line = ""
        for e in SUMMARIES:
            r = next(x for x in rows if x["row_type"] == "summary" and x["model"] == model and x["evaluation"] == e)
            line += f"{100 * r['f1']:>8.1f} [{100 * r['f1_ci_low']:5.1f},{100 * r['f1_ci_high']:5.1f}]"
        print(f"  {label:<18}{line}")
    print("\nΔF1 (puncte) [IC 95%] p(Δ<=0):")
    for r in rows:
        if r["row_type"] == "delta" and r["metric"] == "f1":
            print(f"  {r['label']:<36} {EVAL_LABEL[r['evaluation']]:<18} {100 * r['delta']:+6.1f} "
                  f"[{100 * r['delta_ci_low']:+6.1f},{100 * r['delta_ci_high']:+6.1f}]  p={r['p_delta_le_0']:.3f}")
    print("\nCaptura top 20% patch-uri (%) [IC 95%] (întâmplare = 20):")
    for model, label, _ in MODELS:
        line = ""
        for e in SUMMARIES:
            r = next(x for x in rows if x["row_type"] == "summary" and x["model"] == model and x["evaluation"] == e)
            line += f"{100 * r['capture_top20']:>8.1f} [{100 * r['capture_top20_ci_low']:5.1f},{100 * r['capture_top20_ci_high']:5.1f}]"
        print(f"  {label:<18}{line}")
    print("\nScris: reports/CP3/results_v1_H2.csv, reports/CP3/results_v1.md, "
          "reports/CP3/figures/results_v1_f1.png")


if __name__ == "__main__":
    main()
