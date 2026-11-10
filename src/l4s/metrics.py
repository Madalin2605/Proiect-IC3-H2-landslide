"""Metricile protocolului CP2, pe clasa alunecare (1).

F1, precizia, recall-ul și IoU se calculează din matricea de confuzie cumulată pe toți
pixelii split-ului (ca în competiția Landslide4Sense), nu ca medie pe patch.
AUC și AP (aria sub curba precizie-recall) măsoară scorurile fără prag.
Acuratețea se raportează doar ca să arate de ce nu e potrivită (dezechilibru ~2%).
"""
import numpy as np


def confusion(pred: np.ndarray, target: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    """[tp, fp, fn, tn] pentru predicții binare; `valid` scoate pixelii goi."""
    pred, target = pred.astype(bool).ravel(), target.astype(bool).ravel()
    if valid is not None:
        keep = valid.astype(bool).ravel()
        pred, target = pred[keep], target[keep]
    tp = int((pred & target).sum())
    fp = int((pred & ~target).sum())
    fn = int((~pred & target).sum())
    tn = int((~pred & ~target).sum())
    return np.array([tp, fp, fn, tn], dtype=np.int64)


def scores(counts: np.ndarray) -> dict[str, float]:
    tp, fp, fn, tn = (int(v) for v in counts)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "accuracy": (tp + tn) / max(tp + fp + fn + tn, 1),
    }


def auc(values: np.ndarray, labels: np.ndarray) -> float:
    """ROC-AUC prin statistica Mann-Whitney, cu ranguri medii la egalități."""
    values, labels = np.asarray(values, dtype=np.float64), np.asarray(labels)
    order = np.argsort(values, kind="mergesort")
    _, inverse, counts = np.unique(values[order], return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    ranks = np.empty(len(values))
    ranks[order] = (ends - (counts - 1) / 2)[inverse]
    pos = labels == 1
    n_pos, n_neg = pos.sum(), (~pos).sum()
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def average_precision(values: np.ndarray, labels: np.ndarray) -> float:
    """AP: media preciziei la fiecare pixel pozitiv, cu scorurile sortate descrescător."""
    order = np.argsort(-np.asarray(values, dtype=np.float64), kind="mergesort")
    hits = (np.asarray(labels)[order] == 1)
    if not hits.any():
        return 0.0
    precision_at_k = np.cumsum(hits) / np.arange(1, len(hits) + 1)
    return float(precision_at_k[hits].mean())
