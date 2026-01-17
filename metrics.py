from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
import warnings
from sklearn.exceptions import UndefinedMetricWarning


def pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    num = float((x * y).sum())
    den = float(np.sqrt((x * x).sum()) * np.sqrt((y * y).sum()) + 1e-8)
    return num / den


def spearmanr(x: np.ndarray, y: np.ndarray) -> float:
    # 简易近似：使用秩变换后 Pearson
    x_rank = x.argsort().argsort().astype(np.float32)
    y_rank = y.argsort().argsort().astype(np.float32)
    return pearsonr(x_rank, y_rank)


def mse(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0:
        return float("nan")
    diff = x - y
    return float((diff * diff).mean())


def per_species_metrics(y_true: np.ndarray, y_pred: np.ndarray, species_id: np.ndarray) -> Dict[int, Dict[str, float]]:
    out: Dict[int, Dict[str, float]] = {}
    sp_ids = np.unique(species_id)
    for sp in sp_ids:
        m = species_id == sp
        yt = y_true[m]
        yp = y_pred[m]
        out[int(sp)] = {
            "spearman": spearmanr(yt, yp),
            "pearson": pearsonr(yt, yp),
            "mse": mse(yt, yp),
        }
    return out


def macro_average(metrics_per_species: Dict[int, Dict[str, float]]) -> Dict[str, float]:
    keys = ["spearman", "pearson", "mse"]
    agg = {k: [] for k in keys}
    for sp, d in metrics_per_species.items():
        for k in keys:
            v = d.get(k, float("nan"))
            if not np.isnan(v):
                agg[k].append(v)
    return {k: float(np.mean(v)) if len(v) > 0 else float("nan") for k, v in agg.items()}


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> Dict[str, float | Dict[int, Dict[str, float]]]:
    """Compute accuracy, macro precision/recall/F1, and per-class metrics.

    y_true/y_pred: shape [N] int labels in [0, num_classes)
    """
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    N = y_true.size
    if N == 0:
        return {"accuracy": float("nan"), "macro_precision": float("nan"), "macro_recall": float("nan"), "macro_f1": float("nan"), "per_class": {}}

    C = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            C[t, p] += 1

    correct = np.trace(C)
    accuracy = float(correct / max(1, C.sum()))
    tp = np.diag(C).astype(np.float64)
    pred_sum = C.sum(axis=0).astype(np.float64)
    true_sum = C.sum(axis=1).astype(np.float64)

    precision_c = np.divide(tp, np.maximum(1.0, pred_sum))
    recall_c = np.divide(tp, np.maximum(1.0, true_sum))
    # safe F1 without invalid division warnings
    denom = precision_c + recall_c
    f1_c = np.zeros_like(denom)
    mask = denom > 0
    f1_c[mask] = 2.0 * precision_c[mask] * recall_c[mask] / denom[mask]

    # macro over present classes (with true support > 0) to avoid degenerate bias
    present = true_sum > 0
    if present.any():
        macro_precision = float(np.mean(precision_c[present]))
        macro_recall = float(np.mean(recall_c[present]))
        macro_f1 = float(np.mean(f1_c[present]))
    else:
        macro_precision = float(np.mean(precision_c))
        macro_recall = float(np.mean(recall_c))
        macro_f1 = float(np.mean(f1_c))

    per_class = {
        int(i): {
            "precision": float(precision_c[i]),
            "recall": float(recall_c[i]),
            "f1": float(f1_c[i]),
            "support": int(true_sum[i]),
        }
        for i in range(num_classes)
    }

    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "per_class": per_class,
    }


def classification_auroc(y_true: np.ndarray, y_score: np.ndarray, num_classes: int) -> Dict[str, float | Dict[int, float]]:
    """Compute multiclass AUROC macro (ovr), micro, and per-class AUC.

    y_true: shape [N] int labels
    y_score: shape [N, C] probabilities or scores
    """
    out: Dict[str, float | Dict[int, float]] = {"macro_ovr": float("nan"), "micro": float("nan"), "per_class": {}}
    if y_true.size == 0 or y_score.size == 0:
        return out
    # sanitize and guard
    y_true = y_true.astype(int)
    # drop rows with any NaN in scores
    valid_rows = ~np.isnan(y_score).any(axis=1)
    if not valid_rows.any():
        return out
    y_true_f = y_true[valid_rows]
    y_score_f = y_score[valid_rows]

    present_labels = sorted(set(int(c) for c in np.unique(y_true_f)))
    if len(present_labels) <= 1:
        return out

    # per-class (ovr), compute only for classes with positive support
    per_class: Dict[int, float] = {}
    auc_list: List[float] = []
    for c in range(num_classes):
        try:
            # compute only if class c has positives and negatives
            pos = (y_true_f == c)
            neg = (y_true_f != c)
            if pos.any() and neg.any():
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
                    auc_c = float(roc_auc_score(pos.astype(np.int32), y_score_f[:, c]))
            else:
                auc_c = float("nan")
        except Exception:
            auc_c = float("nan")
        per_class[c] = auc_c
        if not np.isnan(auc_c) and (c in present_labels):
            auc_list.append(auc_c)
    out["per_class"] = per_class
    out["macro_ovr"] = float(np.mean(auc_list)) if len(auc_list) > 0 else float("nan")

    # micro: flatten one-vs-rest across present labels only
    try:
        cols = [c for c in present_labels if 0 <= c < y_score_f.shape[1]]
        if len(cols) >= 2:
            y_true_bin = np.stack([(y_true_f == c).astype(np.int32) for c in cols], axis=1)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
                out["micro"] = float(roc_auc_score(y_true_bin.ravel(), y_score_f[:, cols].ravel()))
    except Exception:
        pass

    return out