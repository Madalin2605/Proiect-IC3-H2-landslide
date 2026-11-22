"""CP3, pasul 3: antrenează U-Net-ul pe protocolul din CP2, pe mai multe seed-uri.

O rulare = (reprezentare, fold, seed):
  fold 1-5  antrenare pe train-ul fold-ului, evaluare pe fold-ul de test (CV spațial)
  full      antrenare pe tot train-ul, evaluare pe validarea și testul oficial

În fiecare rulare:
  - normalizarea se calculează doar pe patch-urile de antrenare (src/l4s/dataset.py);
  - din antrenare se scoate o validare internă: 7 rânduri din G1, cu un rând de buffer
    (src/l4s/splits.py, inner_split). Pe ea se aleg epoca (checkpoint-ul) și pragul de
    decizie, maximizând F1. Fold-ul de test, validarea și testul oficial nu se folosesc
    la nicio alegere;
  - configurația e fixă: U-Net lățime 32, 15 epoci (~48 s fiecare pe GTX 1650), AdamW (lr 1e-3, weight decay 1e-4),
    lr cosinus, batch 16, BCE + Dice, augmentare cu rotații de 90° și flip, fp32
    (mixed precision fp16 a dat loss NaN și a fost de 3-5 ori mai lentă pe GTX 1650).

Rezultate, în reports/CP3/runs/unet_{rep}/seed{s}/{fold}/:
  patches.csv  pe patch: tp, fp, fn, tn la pragul ales, pixeli valizi și de alunecare,
               probabilitatea medie și maximă, fracția prezisă (pentru bootstrap și pentru
               metricile pe patch de la pasul 5)
  hist.npz     histograme ale probabilităților pe pixelii de alunecare / fundal (AUC, AP)
  run.json     configurația, istoricul pe epoci, epoca și pragul alese, metricile, durata
  model.pt     greutățile, doar pentru rulările `full` (folosite la CP4-CP5)

O rulare terminată (cu run.json) nu se refacă la o nouă pornire, deci scriptul poate fi oprit
și repornit. Rulare, din rădăcina repo-ului:

  python scripts/train_unet.py --rep s2_topo                  toate cele 18 rulări pentru A (~4 ore)
  python scripts/train_unet.py --rep topo                     toate cele 18 rulări pentru B (~4 ore)
  python scripts/train_unet.py --rep s2_topo --folds 1 --seeds 42 --epochs 1 --out reports/CP3/runs_smoke
                                                              verificare rapidă (~1 minut)
"""
import argparse
import copy
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from l4s.dataset import CHANNELS, GpuBatches, channel_stats, load_cache  # noqa: E402
from l4s.metrics import scores  # noqa: E402
from l4s.splits import inner_split, load_folds  # noqa: E402
from l4s.unet import UNet, count_parameters, masked_bce_dice  # noqa: E402

BINS = 1000  # histograme de probabilitate: pragurile posibile sunt multipli de 0,001
PATCH_FIELDS = ["split", "patch_id", "tp", "fp", "fn", "tn", "n_valid", "n_pos", "prob_mean", "prob_max", "pred_frac"]


# ---------- metrici din histograme ----------

def threshold_curve(pos_hist: np.ndarray, neg_hist: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pentru fiecare prag t = i / BINS: (tp, fp, fn) când alunecare = probabilitate >= t."""
    tp = np.cumsum(pos_hist[::-1])[::-1].astype(np.int64)
    fp = np.cumsum(neg_hist[::-1])[::-1].astype(np.int64)
    fn = pos_hist.sum() - tp
    return tp, fp, fn


def best_threshold(pos_hist: np.ndarray, neg_hist: np.ndarray) -> tuple[float, float]:
    tp, fp, fn = threshold_curve(pos_hist, neg_hist)
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    i = int(np.argmax(f1[1:]) + 1)  # pragul 0 (totul alunecare) nu se alege
    return i / BINS, float(f1[i])


def auc_ap(pos_hist: np.ndarray, neg_hist: np.ndarray) -> tuple[float, float]:
    """ROC-AUC (cu jumătate de credit la egalitate în bin) și AP, din histograme."""
    p, n = pos_hist.astype(np.float64), neg_hist.astype(np.float64)
    if p.sum() == 0 or n.sum() == 0:
        return float("nan"), float("nan")
    neg_below = np.cumsum(n) - n
    auc = float((p * (neg_below + 0.5 * n)).sum() / (p.sum() * n.sum()))
    tp, fp, _ = threshold_curve(pos_hist, neg_hist)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / p.sum()
    recall_next = np.append(recall[1:], 0.0)
    ap = float(((recall - recall_next) * precision).sum())
    return auc, ap


# ---------- antrenare și predicție ----------

@torch.no_grad()
def predict(model, batches, threshold: float | None = None, split: str = "") -> dict:
    """Histogramele globale și, dacă e dat pragul, rândurile pe patch."""
    model.eval()
    pos_hist = torch.zeros(BINS, dtype=torch.int64, device=batches.device)
    neg_hist = torch.zeros_like(pos_hist)
    rows = []
    for x, y, v, ids in batches:
        prob = torch.sigmoid(model(x))
        b = (prob * BINS).long().clamp_(0, BINS - 1)
        valid, target = v.bool(), y.bool()
        pos_hist += torch.bincount(b[valid & target], minlength=BINS)
        neg_hist += torch.bincount(b[valid & ~target], minlength=BINS)
        if threshold is None:
            continue
        pred = (prob >= threshold) & valid
        dims = (1, 2, 3)
        tp = (pred & target).sum(dims)
        fp = (pred & ~target).sum(dims)
        fn = (~pred & target & valid).sum(dims)
        n_valid = valid.sum(dims)
        n_pos = (target & valid).sum(dims)
        p_valid = prob * v
        prob_mean = p_valid.sum(dims) / n_valid.clamp_min(1)
        prob_max = p_valid.amax(dims)
        for j, pid in enumerate(ids):
            rows.append({
                "split": split, "patch_id": int(pid), "tp": int(tp[j]), "fp": int(fp[j]), "fn": int(fn[j]),
                "tn": int(n_valid[j] - tp[j] - fp[j] - fn[j]), "n_valid": int(n_valid[j]), "n_pos": int(n_pos[j]),
                "prob_mean": round(float(prob_mean[j]), 6), "prob_max": round(float(prob_max[j]), 6),
                "pred_frac": round(float((tp[j] + fp[j]) / max(int(n_valid[j]), 1)), 6),
            })
    return {"pos_hist": pos_hist.cpu().numpy(), "neg_hist": neg_hist.cpu().numpy(), "rows": rows}


def summarize(pred: dict) -> dict:
    counts = np.sum([[r["tp"], r["fp"], r["fn"], r["tn"]] for r in pred["rows"]], axis=0)
    auc, ap = auc_ap(pred["pos_hist"], pred["neg_hist"])
    s = scores(counts)
    return {k: s[k] for k in ("f1", "precision", "recall", "iou")} | {"auc": auc, "ap": ap, "patches": len(pred["rows"])}


def train_run(args, train_cache, eval_caches: dict, rep: str, fold_name: str, train_ids: list[int],
              test_ids: list[int], seed: int, out_dir: Path) -> None:
    start = time.time()
    torch.manual_seed(seed)
    np.random.seed(seed)
    fit_ids, val_ids, buf_ids = inner_split(train_ids, test_ids)
    mean, std = channel_stats(train_cache, fit_ids, rep)
    common = dict(representation=rep, mean=mean, std=std, batch_size=args.batch_size, device="cuda")
    fit = GpuBatches(train_cache, fit_ids, shuffle=True, augment=True, seed=seed, **common)
    inner = GpuBatches(train_cache, val_ids, **common)

    model = UNet(len(CHANNELS[rep]), base=args.base).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * len(fit))
    print(f"\n=== unet {rep} | {fold_name} | seed {seed} | antrenare {len(fit_ids)}, validare internă "
          f"{len(val_ids)}, buffer {len(buf_ids)} | {count_parameters(model) / 1e6:.2f} M parametri")

    history, best = [], {"f1": -1.0}
    for epoch in range(1, args.epochs + 1):
        t = time.time()
        model.train()
        total, n = 0.0, 0
        for x, y, v, _ in fit:
            opt.zero_grad(set_to_none=True)
            loss = masked_bce_dice(model(x), y, v)
            loss.backward()
            opt.step()
            sched.step()
            total, n = total + float(loss), n + 1
        p = predict(model, inner)
        thr, f1 = best_threshold(p["pos_hist"], p["neg_hist"])
        history.append({"epoch": epoch, "train_loss": round(total / n, 5), "inner_f1": round(f1, 5),
                        "inner_threshold": thr, "seconds": round(time.time() - t, 1)})
        mark = ""
        if f1 > best["f1"]:
            best = {"f1": f1, "epoch": epoch, "threshold": thr,
                    "state": copy.deepcopy({k: w.detach().cpu() for k, w in model.state_dict().items()})}
            mark = " *"
        print(f"  epoca {epoch:>2}/{args.epochs}: loss {total / n:.4f} | F1 validare internă {100 * f1:5.2f}% "
              f"(prag {thr:.3f}) | {time.time() - t:4.0f} s{mark}")

    model.load_state_dict(best["state"])
    print(f"  ales: epoca {best['epoch']}, prag {best['threshold']:.3f}, F1 validare internă {100 * best['f1']:.2f}%")

    out_dir.mkdir(parents=True, exist_ok=True)
    results, rows, hists = {}, [], {}
    for split, (cache, ids) in eval_caches.items():
        batches = GpuBatches(cache, ids, **common)
        p = predict(model, batches, threshold=best["threshold"], split=split)
        results[split] = summarize(p)
        rows += p["rows"]
        hists[f"{split}_pos"], hists[f"{split}_neg"] = p["pos_hist"], p["neg_hist"]
        r = results[split]
        print(f"  {split:<10} F1 {100 * r['f1']:5.2f}% | precizie {100 * r['precision']:5.2f}% | "
              f"recall {100 * r['recall']:5.2f}% | AUC {r['auc']:.3f} | AP {r['ap']:.3f}")

    with open(out_dir / "patches.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PATCH_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(out_dir / "hist.npz", **hists)
    if fold_name == "full":
        torch.save({"state_dict": best["state"], "mean": mean, "std": std, "representation": rep,
                    "base": args.base, "threshold": best["threshold"]}, out_dir / "model.pt")
    (out_dir / "run.json").write_text(json.dumps({
        "model": "unet", "representation": rep, "fold": fold_name, "seed": seed,
        "config": {k: v for k, v in vars(args).items() if k in ("epochs", "base", "batch_size", "lr", "weight_decay")},
        "n_fit": len(fit_ids), "n_inner_val": len(val_ids), "n_inner_buffer": len(buf_ids),
        "inner_val_ids": [min(val_ids), max(val_ids)],
        "best_epoch": best["epoch"], "threshold": best["threshold"], "inner_f1": best["f1"],
        "results": results, "history": history, "runtime_seconds": round(time.time() - start),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  salvat în {out_dir} ({(time.time() - start) / 60:.1f} min)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rep", choices=list(CHANNELS), required=True, help="s2_topo = A, topo = B")
    parser.add_argument("--folds", default="1,2,3,4,5,full", help="fold-uri de rulat, separate prin virgulă")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--base", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--cache", type=Path, default=ROOT / "data" / "cache")
    parser.add_argument("--splits", type=Path, default=ROOT / "reports" / "CP2" / "splits.json")
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "CP3" / "runs")
    args = parser.parse_args()
    torch.backends.cudnn.benchmark = True

    splits = json.loads(args.splits.read_text(encoding="utf-8"))
    excluded = set(splits["excluded_ids"])
    folds = {str(f["fold"]): f for f in load_folds(args.splits)}
    train_cache = load_cache(args.cache, "train")
    usable = [int(i) for i in train_cache.ids if int(i) not in excluded]
    official = {}

    plan = [(s, f) for s in (int(x) for x in args.seeds.split(",")) for f in args.folds.split(",")]
    start = time.time()
    for done, (seed, fold_name) in enumerate(plan, 1):
        out_dir = args.out / f"unet_{args.rep}" / f"seed{seed}" / (fold_name if fold_name == "full" else f"fold{fold_name}")
        if (out_dir / "run.json").exists():
            print(f"[{done}/{len(plan)}] există deja: {out_dir}")
            continue
        if fold_name == "full":
            if not official:
                for split in ("validation", "test"):
                    cache = load_cache(args.cache, split)
                    official[split] = (cache, [int(i) for i in cache.ids])
            train_ids, test_ids, evals = usable, [], official
        else:
            fold = folds[fold_name]
            train_ids, test_ids = fold["train_ids"], fold["test_ids"]
            evals = {f"cv_fold_{fold_name}": (train_cache, test_ids)}
        print(f"[{done}/{len(plan)}] după {(time.time() - start) / 60:.0f} min")
        train_run(args, train_cache, evals, args.rep, fold_name, train_ids, test_ids, seed, out_dir)
    print(f"\nGata: {len(plan)} rulări în {(time.time() - start) / 60:.0f} min.")


if __name__ == "__main__":
    main()
