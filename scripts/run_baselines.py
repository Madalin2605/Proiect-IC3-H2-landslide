"""CP2, etapa 3: rulează baseline-urile pe protocolul din reports/splits.json.

Baseline-uri, de la trivial la clasicul puternic:
  all_negative     "fără alunecare" peste tot (F1 = 0, arată de ce acuratețea nu ajută)
  rule / s2_topo   prag pe NDVI: alunecare dacă NDVI < prag
  rule / topo      prag pe pantă: alunecare dacă panta > prag (proxy topografic)
  rf   / s2_topo   Random Forest pe pixeli, reprezentarea A (18 trăsături)
  rf   / topo      Random Forest pe pixeli, reprezentarea B (pantă + DEM)

Antrenare: din fiecare patch de antrenare se eșantionează pixeli valizi (până la 50 de
alunecare și 150 de fundal, seed fix). Pragul fiecărui baseline se alege pe datele de
antrenare (la RF pe predicțiile out-of-bag), maximizând F1 cu ponderi care readuc
eșantionul la prevalența reală. Hiperparametrii RF sunt fixați dinainte, nu se reglează.

Evaluare: pe toți pixelii valizi ai patch-urilor evaluate.
  CV       5 fold-uri spațiale din splits.json (modele reantrenate pe fiecare fold)
  validare 245 de patch-uri din alte situri (model antrenat pe tot train-ul)
  test     800 de patch-uri din alte situri, cu aceeași configurație, rulată o singură dată

Rulare, din rădăcina repo-ului:  python scripts/run_baselines.py
Rapid, pentru verificare:        python scripts/run_baselines.py --folds 1 --skip-test
Rezultate: reports/baseline_results.{csv,json,md}, reports/figures/baseline_f1.png
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from l4s.data import image_ids, load_image, load_mask  # noqa: E402
from l4s.features import feature_cube, feature_names, valid_pixels  # noqa: E402
from l4s.metrics import auc, average_precision, best_threshold, confusion, scores  # noqa: E402
from l4s.splits import load_folds  # noqa: E402

NAMES = feature_names("s2_topo")
COLUMNS = {"s2_topo": list(range(len(NAMES))), "topo": [NAMES.index("SLOPE"), NAMES.index("DEM")]}
SPECS = [
    {"key": "all_negative", "kind": "all_negative", "representation": "-", "label": "Totul negativ"},
    {"key": "rule_ndvi", "kind": "rule", "representation": "s2_topo", "label": "Prag NDVI",
     "feature": "NDVI", "sign": -1.0},
    {"key": "rule_slope", "kind": "rule", "representation": "topo", "label": "Prag pantă",
     "feature": "SLOPE", "sign": 1.0},
    {"key": "rf_s2_topo", "kind": "rf", "representation": "s2_topo", "label": "RF pe pixeli (A)"},
    {"key": "rf_topo", "kind": "rf", "representation": "topo", "label": "RF pe pixeli (B)"},
]
REP_LABEL = {"s2_topo": "A: S2 + indici + pantă + DEM", "topo": "B: pantă + DEM", "-": "-"}
LITERATURE = {"label": "U-Net, test oficial (Ghorbanzadeh et al., 2022b)", "f1": 0.5992}
BATCH_ROWS = 600_000
METRICS = ["f1", "precision", "recall", "iou", "auc", "ap"]


# ---------- date ----------

def sample_train_pixels(data: Path, ids: list[int], pos_k: int, neg_k: int, rng: np.random.Generator) -> dict:
    """Pixeli eșantionați pe patch, plus câți pixeli valizi are fiecare patch pe clasă."""
    X, y, owner, totals = [], [], [], {}
    for patch_id in ids:
        x = load_image(data, "train", patch_id)
        mask = load_mask(data, "train", patch_id).reshape(-1)
        valid = valid_pixels(x).reshape(-1)
        cube = feature_cube(x, "s2_topo").reshape(-1, len(NAMES))
        totals[patch_id] = (int((valid & (mask == 1)).sum()), int((valid & (mask == 0)).sum()))
        for cls, k in [(1, pos_k), (0, neg_k)]:
            idx = np.flatnonzero(valid & (mask == cls))
            if len(idx) > k:
                idx = rng.choice(idx, k, replace=False)
            X.append(cube[idx])
            y.append(np.full(len(idx), cls, dtype=np.uint8))
            owner.append(np.full(len(idx), patch_id, dtype=np.int32))
    return {"X": np.vstack(X), "y": np.concatenate(y), "owner": np.concatenate(owner), "totals": totals}


def training_rows(sample: dict, train_ids: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = np.isin(sample["owner"], train_ids)
    X, y = sample["X"][rows], sample["y"][rows]
    real_pos = sum(sample["totals"][i][0] for i in train_ids)
    real_neg = sum(sample["totals"][i][1] for i in train_ids)
    weights = np.where(y == 1, real_pos / max((y == 1).sum(), 1), real_neg / max((y == 0).sum(), 1))
    return X, y, weights


# ---------- modele ----------

def fit(spec: dict, X: np.ndarray, y: np.ndarray, w: np.ndarray, args) -> dict:
    model = {"spec": spec}
    if spec["kind"] == "all_negative":
        model["threshold"] = np.inf
    elif spec["kind"] == "rule":
        column = NAMES.index(spec["feature"])
        model["column"] = column
        model["threshold"], model["train_f1"] = best_threshold(spec["sign"] * X[:, column], y, w)
    else:
        rf = RandomForestClassifier(
            n_estimators=args.trees, max_depth=args.max_depth, min_samples_leaf=args.min_leaf,
            max_features="sqrt", oob_score=True, n_jobs=-1, random_state=args.seed,
        )
        rf.fit(X[:, COLUMNS[spec["representation"]]], y)
        oob = np.nan_to_num(rf.oob_decision_function_[:, 1])
        model["rf"] = rf
        model["threshold"], model["train_f1"] = best_threshold(oob, y, w)
        model["importances"] = dict(zip([NAMES[c] for c in COLUMNS[spec["representation"]]],
                                        rf.feature_importances_.round(4).tolist()))
    return model


def predict_scores(model: dict, X: np.ndarray) -> np.ndarray:
    spec = model["spec"]
    if spec["kind"] == "all_negative":
        return np.zeros(len(X), dtype=np.float32)
    if spec["kind"] == "rule":
        return (spec["sign"] * X[:, model["column"]]).astype(np.float32)
    return model["rf"].predict_proba(X[:, COLUMNS[spec["representation"]]])[:, 1].astype(np.float32)


def evaluate(models: list[dict], data: Path, split: str, ids: list[int]) -> dict:
    """Aplică toate modelele pe toți pixelii valizi ai patch-urilor date."""
    counts = {m["spec"]["key"]: np.zeros(4, dtype=np.int64) for m in models}
    all_scores = {m["spec"]["key"]: [] for m in models}
    all_labels, batch_X, batch_y = [], [], []

    def flush():
        if not batch_X:
            return
        X, y = np.vstack(batch_X), np.concatenate(batch_y)
        for m in models:
            s = predict_scores(m, X)
            counts[m["spec"]["key"]] += confusion(s >= m["threshold"], y)
            all_scores[m["spec"]["key"]].append(s)
        all_labels.append(y)
        batch_X.clear()
        batch_y.clear()

    rows = 0
    for patch_id in ids:
        x = load_image(data, split, patch_id)
        valid = valid_pixels(x).reshape(-1)
        batch_X.append(feature_cube(x, "s2_topo").reshape(-1, len(NAMES))[valid])
        batch_y.append(load_mask(data, split, patch_id).reshape(-1)[valid])
        rows += int(valid.sum())
        if rows >= BATCH_ROWS:
            flush()
            rows = 0
    flush()

    labels = np.concatenate(all_labels)
    result = {}
    for m in models:
        key = m["spec"]["key"]
        s = np.concatenate(all_scores[key])
        result[key] = {**scores(counts[key]), "auc": auc(s, labels), "ap": average_precision(s, labels),
                       "threshold": float(m["threshold"]), "pixels": int(len(labels)),
                       "confusion_tp_fp_fn_tn": counts[key].tolist()}
    return result


def fit_all(sample: dict, train_ids: list[int], args) -> list[dict]:
    X, y, w = training_rows(sample, train_ids)
    return [fit(spec, X, y, w, args) for spec in SPECS]


# ---------- raportare ----------

def pct(v: float) -> str:
    return f"{v * 100:.2f}".replace(".", ",")


def cv_f1(rows: list[dict], key: str, folds: list[int]) -> list[float]:
    return [r["f1"] for r in rows if r["baseline"] == key and r["evaluation"] in {f"cv_fold_{k}" for k in folds}]


def mean_std(values: list[float]) -> str:
    if not values:
        return "-"
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return f"{pct(float(np.mean(values)))} ± {pct(std)}"


def dec(v: float) -> str:
    return f"{v:.3f}".replace(".", ",")


def write_markdown(rows: list[dict], summary: dict, path: Path, args) -> None:
    def get(key, evaluation):
        return next((r for r in rows if r["baseline"] == key and r["evaluation"] == evaluation), None)

    lines = [
        "# Rezultate baseline (CP2)",
        "",
        "Generat de `scripts/run_baselines.py`. Metricile sunt pe clasa alunecare, din confuzia "
        "cumulată pe toți pixelii valizi. Pragurile și hiperparametrii sunt fixați pe train; "
        "testul a fost rulat o singură dată, cu configurația fixată.",
        "",
        "- **CV G1 (fold 1–4):** benzi din grila G1, cu restul G1 la antrenare: generalizare în același sit "
        "(medie ± abatere standard între fold-uri).",
        "- **Fold 5 (G2–G5):** grilele mici, cu antrenare aproape doar pe G1: transfer către altă zonă.",
        "- **Validare / test:** situri oficiale noi, model antrenat pe tot train-ul.",
        "",
        "## Tabelul principal: F1 (%)",
        "",
        "| Baseline | Reprezentare | CV G1 (fold 1–4) | Fold 5 (G2–G5) | CV 5 fold-uri | Validare | Test | IoU test | AUC test | AP test |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for spec in SPECS:
        key = spec["key"]
        val, test = get(key, "validation"), get(key, "test")
        fold5 = cv_f1(rows, key, [5])
        lines.append(
            f"| {spec['label']} | {REP_LABEL[spec['representation']]} | {mean_std(cv_f1(rows, key, [1, 2, 3, 4]))} | "
            f"{pct(fold5[0]) if fold5 else '-'} | {mean_std(cv_f1(rows, key, [1, 2, 3, 4, 5]))} | "
            f"{pct(val['f1']) if val else '-'} | {pct(test['f1']) if test else '-'} | "
            f"{pct(test['iou']) if test else '-'} | {dec(test['auc']) if test else '-'} | "
            f"{dec(test['ap']) if test else '-'} |"
        )
    lines += ["", f"Referință din literatură: {LITERATURE['label']}, F1 = {pct(LITERATURE['f1'])}%. "
              "U-Net folosește contextul spațial al patch-ului; baseline-urile de aici clasifică fiecare pixel separat.", ""]

    lines += ["## Precizie și recall (%)", "", "| Baseline | Evaluare | Precizie | Recall | Prag |", "|---|---|---:|---:|---:|"]
    for spec in SPECS[1:]:
        for evaluation, name in [("validation", "validare"), ("test", "test")]:
            r = get(spec["key"], evaluation)
            if r:
                threshold = f"{r['threshold']:.4g}".replace(".", ",")
                lines.append(f"| {spec['label']} | {name} | {pct(r['precision'])} | {pct(r['recall'])} | {threshold} |")
    lines += ["", "Pragul se aplică scorului: −NDVI la pragul NDVI, panta la pragul de pantă, probabilitatea la RF."]

    folds = sorted({r["evaluation"] for r in rows if r["evaluation"].startswith("cv_fold_")})
    if folds:
        lines += ["", "## F1 pe fold (%)", "", "| Baseline | " + " | ".join(f.replace("cv_fold_", "Fold ") for f in folds) + " |",
                  "|---|" + "---:|" * len(folds)]
        for spec in SPECS:
            lines.append(f"| {spec['label']} | " + " | ".join(pct(get(spec["key"], f)["f1"]) for f in folds) + " |")
    lines += ["", f"Configurație RF: {args.trees} arbori, adâncime maximă {args.max_depth}, minimum {args.min_leaf} "
              f"pixeli în frunză, seed {args.seed}; eșantion de antrenare: până la {args.pos_per_patch} pixeli de alunecare "
              f"și {args.neg_per_patch} de fundal per patch.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_f1(rows: list[dict], summary: dict, path: Path) -> None:
    specs = SPECS[1:]
    xs = np.arange(len(specs))
    width = 0.2
    fig, ax = plt.subplots(figsize=(10, 3.9), constrained_layout=True)
    groups = [(-1.5 * width, "g1", "#3b6ea8", "CV în G1, fold 1–4 (medie ± abatere)"),
              (-0.5 * width, "fold5", "#8a6bb1", "fold 5: grilele G2–G5"),
              (0.5 * width, "validation", "#d98b2b", "validare (situri noi)"),
              (1.5 * width, "test", "#5a9e5a", "test oficial (situri noi)")]
    for offset, evaluation, color, label in groups:
        values, errors = [], []
        for spec in specs:
            if evaluation == "g1":
                f1s = cv_f1(rows, spec["key"], [1, 2, 3, 4])
                values.append(np.mean(f1s) * 100 if f1s else np.nan)
                errors.append(np.std(f1s, ddof=1) * 100 if len(f1s) > 1 else 0)
            elif evaluation == "fold5":
                f1s = cv_f1(rows, spec["key"], [5])
                values.append(f1s[0] * 100 if f1s else np.nan)
                errors.append(0)
            else:
                r = next((r for r in rows if r["baseline"] == spec["key"] and r["evaluation"] == evaluation), None)
                values.append(r["f1"] * 100 if r else np.nan)
                errors.append(0)
        bars = ax.bar(xs + offset, values, width, yerr=errors if evaluation == "g1" else None,
                      capsize=3, color=color, label=label)
        ax.bar_label(bars, labels=[f"{v:.1f}".replace(".", ",") for v in values], fontsize=7, padding=2)
    ax.axhline(LITERATURE["f1"] * 100, color="black", ls="--", lw=0.9, label=LITERATURE["label"])
    ax.set_xticks(xs, [f"{s['label']}\n{REP_LABEL[s['representation']]}" for s in specs], fontsize=8)
    ax.set_ylabel("F1 pe clasa alunecare (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Baseline-uri pe pixeli: reprezentarea A vs alternativa B (totul negativ: F1 = 0)", fontsize=10)
    ax.legend(fontsize=7, loc="upper left", ncol=2, frameon=False)
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


# ---------- main ----------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--splits", type=Path, default=ROOT / "reports" / "splits.json")
    parser.add_argument("--out", type=Path, default=ROOT / "reports")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=str, default="1,2,3,4,5", help="fold-urile CV de rulat; gol = fără CV")
    parser.add_argument("--skip-test", action="store_true", help="nu evalua pe testul oficial")
    parser.add_argument("--pos-per-patch", type=int, default=50)
    parser.add_argument("--neg-per-patch", type=int, default=150)
    parser.add_argument("--trees", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=20)
    parser.add_argument("--min-leaf", type=int, default=5)
    parser.add_argument("--report-only", action="store_true",
                        help="refă doar tabelul .md și figura din baseline_results.json, fără rulare")
    args = parser.parse_args()

    if args.report_only:
        saved = json.loads((args.out / "baseline_results.json").read_text(encoding="utf-8"))
        for key, value in saved["config"].items():
            if key not in {"out", "report_only"}:
                setattr(args, key, value)
        write_markdown(saved["rows"], saved["summary"], args.out / "baseline_results.md", args)
        plot_f1(saved["rows"], saved["summary"], args.out / "figures" / "baseline_f1.png")
        print("Refăcut: reports/baseline_results.md și reports/figures/baseline_f1.png")
        return

    start = time.time()
    rng = np.random.default_rng(args.seed)
    splits = json.loads(args.splits.read_text(encoding="utf-8"))
    excluded = set(splits["excluded_ids"])
    folds = {f["fold"]: f for f in load_folds(args.splits)}
    usable = [i for i in image_ids(args.data, "train") if i not in excluded]

    print(f"Eșantionez pixeli din {len(usable)} patch-uri de train...")
    sample = sample_train_pixels(args.data, usable, args.pos_per_patch, args.neg_per_patch, rng)
    print(f"  {len(sample['y'])} pixeli ({int(sample['y'].sum())} de alunecare), {time.time() - start:.0f} s")

    rows, extra = [], {"thresholds": {}, "rf_importances": {}, "train_f1_at_threshold": {}}

    def record(evaluation: str, result: dict) -> None:
        for spec in SPECS:
            r = result[spec["key"]]
            rows.append({"baseline": spec["key"], "representation": spec["representation"], "evaluation": evaluation,
                         **{k: r[k] for k in ["f1", "precision", "recall", "iou", "accuracy", "auc", "ap", "threshold", "pixels"]}})
            print(f"  {evaluation:<12} {spec['label']:<18} F1 {r['f1'] * 100:6.2f}  AUC {r['auc']:.3f}")

    selected = [int(k) for k in args.folds.split(",") if k.strip()]
    for k in selected:
        fold = folds[k]
        t = time.time()
        models = fit_all(sample, fold["train_ids"], args)
        result = evaluate(models, args.data, "train", fold["test_ids"])
        print(f"Fold {k}: {len(fold['test_ids'])} patch-uri de test, {time.time() - t:.0f} s")
        record(f"cv_fold_{k}", result)

    t = time.time()
    final = fit_all(sample, usable, args)
    for m in final:
        key = m["spec"]["key"]
        extra["thresholds"][key] = float(m["threshold"])
        if "train_f1" in m:
            extra["train_f1_at_threshold"][key] = m["train_f1"]
        if "importances" in m:
            extra["rf_importances"][key] = m["importances"]
    evaluations = [("validation", image_ids(args.data, "validation"))]
    if not args.skip_test:
        evaluations.append(("test", image_ids(args.data, "test")))
    for split, ids in evaluations:
        result = evaluate(final, args.data, split, ids)
        print(f"{split}: {len(ids)} patch-uri, {time.time() - t:.0f} s")
        record(split, result)

    summary = {"cv": {}}
    for spec in SPECS:
        cv_rows = [r for r in rows if r["baseline"] == spec["key"] and r["evaluation"].startswith("cv_fold_")]
        if cv_rows:
            summary["cv"][spec["key"]] = {
                **{f"{m}_mean": float(np.mean([r[m] for r in cv_rows])) for m in METRICS},
                **{f"{m}_std": float(np.std([r[m] for r in cv_rows], ddof=1)) if len(cv_rows) > 1 else 0.0 for m in METRICS},
                "folds": len(cv_rows),
            }

    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "baseline_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        for r in rows:
            writer.writerow({k: f"{v:.6g}" if isinstance(v, float) else v for k, v in r.items()})
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    (args.out / "baseline_results.json").write_text(json.dumps(
        {"config": config, "sampled_train_pixels": int(len(sample["y"])), "summary": summary, **extra,
         "literature_reference": LITERATURE, "runtime_seconds": round(time.time() - start), "rows": rows},
        indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(rows, summary, args.out / "baseline_results.md", args)
    (args.out / "figures").mkdir(parents=True, exist_ok=True)
    plot_f1(rows, summary, args.out / "figures" / "baseline_f1.png")
    print(f"Gata în {time.time() - start:.0f} s. Vezi reports/baseline_results.md")


if __name__ == "__main__":
    main()
