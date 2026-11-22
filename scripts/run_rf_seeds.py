"""CP3, pasul 4: baseline-urile din CP2 pe aceleași seed-uri și fold-uri ca U-Net-ul.

Pentru comparația de la CP3 (U-Net vs. RF, pe seed-uri, cu intervale), baseline-urile se
rulează pe seed-urile 42, 43 și 44, cu rezultate salvate pe patch, în același format ca
scripts/train_unet.py, ca să se poată face bootstrap pereche pe aceleași patch-uri.

Modele (configurația din CP2, neschimbată; codul de antrenare e cel din run_baselines.py):
  rf_s2_topo   Random Forest pe pixeli, reprezentarea A
  rf_topo      Random Forest pe pixeli, reprezentarea B
  rule_ndvi    prag pe NDVI (A)
  rule_slope   prag pe pantă (B)

Seed-ul controlează eșantionarea pixelilor de antrenare (până la 50 de alunecare și 150 de
fundal per patch) și arborii RF. Regulile nu au parte aleatoare, dar pragul lor se alege pe
pixelii eșantionați, deci variază puțin între seed-uri.

O rulare = (seed, fold): fold 1-5 = antrenare pe train-ul fold-ului, evaluare pe fold-ul de
test; full = antrenare pe tot train-ul, evaluare pe validarea și testul oficial. Pragul se
alege pe predicțiile out-of-bag (RF) sau pe pixelii de antrenare (reguli), ca la CP2.

Datele vin din cache-ul de la CP3 (data/cache, float16). Față de CP2 (float32 din HDF5)
pot apărea diferențe foarte mici, deci seed-ul 42 nu reproduce exact tabelul din CP2.

Rezultate, în reports/CP3/runs/{model}/seed{s}/{fold}/:
  patches.csv  pe patch: tp, fp, fn, tn la pragul ales, pixeli valizi și de alunecare,
               scorul mediu și maxim (prob_mean / prob_max; la reguli e -NDVI, respectiv
               panta), fracția prezisă
  hist.npz     doar pentru RF: histograme ale probabilităților (ca la U-Net)
  run.json     configurația, pragul, metricile (AUC și AP exacte, pe toți pixelii), durata

Rulările terminate nu se refac, deci scriptul poate fi oprit și repornit.
Rulare, din rădăcina repo-ului:
  python scripts/run_rf_seeds.py                              toate: 3 seed-uri x 6 fold-uri
  python scripts/run_rf_seeds.py --seeds 42 --folds 1 --out reports/CP3/runs_smoke   verificare rapidă
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from l4s.dataset import load_cache  # noqa: E402
from l4s.metrics import auc, average_precision, scores  # noqa: E402
from l4s.splits import load_folds  # noqa: E402
from run_baselines import SPECS, fit, predict_scores, training_rows  # noqa: E402

MODELS = [s for s in SPECS if s["kind"] != "all_negative"]
BINS = 1000
BATCH_PATCHES = 32
PATCH_FIELDS = ["split", "patch_id", "tp", "fp", "fn", "tn", "n_valid", "n_pos", "prob_mean", "prob_max", "pred_frac"]


def patch_pixels(cache, row: int) -> tuple[np.ndarray, np.ndarray]:
    """Trăsăturile (n_valid x 18, float32) și eticheta pixelilor valizi ai unui patch."""
    valid = np.asarray(cache.valid[row])
    x = np.asarray(cache.x[row], dtype=np.float32)  # 18 x H x W
    return x[:, valid].T, np.asarray(cache.y[row])[valid]


def sample_train_pixels(cache, ids: list[int], pos_k: int, neg_k: int, rng: np.random.Generator) -> dict:
    """La fel ca run_baselines.sample_train_pixels, dar din cache."""
    X, y, owner, totals = [], [], [], {}
    for patch_id in ids:
        cube, mask = patch_pixels(cache, cache.row_of[patch_id])
        totals[patch_id] = (int((mask == 1).sum()), int((mask == 0).sum()))
        for cls, k in [(1, pos_k), (0, neg_k)]:
            idx = np.flatnonzero(mask == cls)
            if len(idx) > k:
                idx = rng.choice(idx, k, replace=False)
            X.append(cube[idx])
            y.append(np.full(len(idx), cls, dtype=np.uint8))
            owner.append(np.full(len(idx), patch_id, dtype=np.int32))
    return {"X": np.vstack(X), "y": np.concatenate(y), "owner": np.concatenate(owner), "totals": totals}


def evaluate(models: list[dict], cache, ids: list[int], split: str) -> dict:
    """Pentru fiecare model: rândurile pe patch, metricile globale și (la RF) histogramele."""
    out = {m["spec"]["key"]: {"rows": [], "scores": [], "pos": np.zeros(BINS, np.int64), "neg": np.zeros(BINS, np.int64)}
           for m in models}
    labels = []
    for start in range(0, len(ids), BATCH_PATCHES):
        chunk = ids[start:start + BATCH_PATCHES]
        parts = [patch_pixels(cache, cache.row_of[i]) for i in chunk]
        X = np.vstack([p[0] for p in parts])
        y = np.concatenate([p[1] for p in parts])
        bounds = np.cumsum([0] + [len(p[1]) for p in parts])
        labels.append(y)
        for m in models:
            key, o = m["spec"]["key"], out[m["spec"]["key"]]
            s = predict_scores(m, X)
            o["scores"].append(s)
            if m["spec"]["kind"] == "rf":
                b = np.clip((s * BINS).astype(np.int64), 0, BINS - 1)
                o["pos"] += np.bincount(b[y == 1], minlength=BINS)
                o["neg"] += np.bincount(b[y == 0], minlength=BINS)
            pred = s >= m["threshold"]
            for j, pid in enumerate(chunk):
                a, z = bounds[j], bounds[j + 1]
                p, t = pred[a:z], y[a:z] == 1
                n = z - a
                tp, fp, fn = int((p & t).sum()), int((p & ~t).sum()), int((~p & t).sum())
                o["rows"].append({
                    "split": split, "patch_id": int(pid), "tp": tp, "fp": fp, "fn": fn, "tn": int(n - tp - fp - fn),
                    "n_valid": int(n), "n_pos": int(t.sum()),
                    "prob_mean": round(float(s[a:z].mean()), 6) if n else 0.0,
                    "prob_max": round(float(s[a:z].max()), 6) if n else 0.0,
                    "pred_frac": round(float(p.mean()), 6) if n else 0.0,
                })
    labels = np.concatenate(labels)
    for o in out.values():
        s = np.concatenate(o.pop("scores"))
        counts = np.sum([[r["tp"], r["fp"], r["fn"], r["tn"]] for r in o["rows"]], axis=0)
        sc = scores(counts)
        o["result"] = {k: sc[k] for k in ("f1", "precision", "recall", "iou")} | {
            "auc": auc(s, labels), "ap": average_precision(s, labels), "patches": len(o["rows"])}
    return out


def save(model: dict, evals: dict, out_dir: Path, seed: int, fold_name: str, n_fit: int, args, seconds: float) -> None:
    spec, key = model["spec"], model["spec"]["key"]
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, hists, results = [], {}, {}
    for split, per_model in evals.items():
        o = per_model[key]
        rows += o["rows"]
        results[split] = o["result"]
        if spec["kind"] == "rf":
            hists[f"{split}_pos"], hists[f"{split}_neg"] = o["pos"], o["neg"]
    with open(out_dir / "patches.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PATCH_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    if hists:
        np.savez_compressed(out_dir / "hist.npz", **hists)
    config = {"pos_per_patch": args.pos_per_patch, "neg_per_patch": args.neg_per_patch}
    if spec["kind"] == "rf":
        config |= {"trees": args.trees, "max_depth": args.max_depth, "min_leaf": args.min_leaf, "max_features": "sqrt"}
    else:
        config |= {"feature": spec["feature"], "sign": spec["sign"]}
    (out_dir / "run.json").write_text(json.dumps({
        "model": spec["kind"], "key": key, "representation": spec["representation"], "fold": fold_name, "seed": seed,
        "config": config, "n_fit": n_fit, "threshold": float(model["threshold"]),
        "train_f1": model.get("train_f1"), "importances": model.get("importances"),
        "results": results, "runtime_seconds": round(seconds),
    }, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", default="1,2,3,4,5,full")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--pos-per-patch", type=int, default=50)
    parser.add_argument("--neg-per-patch", type=int, default=150)
    parser.add_argument("--trees", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=20)
    parser.add_argument("--min-leaf", type=int, default=5)
    parser.add_argument("--cache", type=Path, default=ROOT / "data" / "cache")
    parser.add_argument("--splits", type=Path, default=ROOT / "reports" / "CP2" / "splits.json")
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "CP3" / "runs")
    args = parser.parse_args()

    splits = json.loads(args.splits.read_text(encoding="utf-8"))
    excluded = set(splits["excluded_ids"])
    folds = {str(f["fold"]): f for f in load_folds(args.splits)}
    train = load_cache(args.cache, "train", in_memory=False)
    official = {s: load_cache(args.cache, s, in_memory=False) for s in ("validation", "test")}
    usable = [int(i) for i in train.ids if int(i) not in excluded]
    start = time.time()

    for seed in (int(s) for s in args.seeds.split(",")):
        todo = [f for f in args.folds.split(",")
                if not all((args.out / m["key"] / f"seed{seed}" / (f if f == "full" else f"fold{f}") / "run.json").exists()
                           for m in MODELS)]
        if not todo:
            print(f"seed {seed}: toate rulările există deja")
            continue
        t = time.time()
        sample = sample_train_pixels(train, usable, args.pos_per_patch, args.neg_per_patch, np.random.default_rng(seed))
        print(f"\n=== seed {seed}: {len(sample['y'])} pixeli eșantionați ({int(sample['y'].sum())} de alunecare), "
              f"{time.time() - t:.0f} s")
        args.seed = seed  # fit() din run_baselines folosește args.seed ca random_state pentru RF
        for fold_name in todo:
            t = time.time()
            if fold_name == "full":
                train_ids, evals_on = usable, {s: (c, [int(i) for i in c.ids]) for s, c in official.items()}
            else:
                train_ids = folds[fold_name]["train_ids"]
                evals_on = {f"cv_fold_{fold_name}": (train, folds[fold_name]["test_ids"])}
            X, y, w = training_rows(sample, train_ids)
            models = [fit(spec, X, y, w, args) for spec in MODELS]
            evals = {split: evaluate(models, cache, ids, split) for split, (cache, ids) in evals_on.items()}
            seconds = time.time() - t
            for m in models:
                key = m["spec"]["key"]
                save(m, evals, args.out / key / f"seed{seed}" / (fold_name if fold_name == "full" else f"fold{fold_name}"),
                     seed, fold_name, len(train_ids), args, seconds)
                line = " | ".join(f"{split} F1 {100 * e[key]['result']['f1']:5.2f}% AUC {e[key]['result']['auc']:.3f}"
                                  for split, e in evals.items())
                print(f"  {fold_name:<4} {key:<11} {line}")
            print(f"  {fold_name}: {seconds:.0f} s (total {(time.time() - start) / 60:.0f} min)")
    print(f"\nGata în {(time.time() - start) / 60:.0f} min. Rezultate în {args.out}")


if __name__ == "__main__":
    main()
