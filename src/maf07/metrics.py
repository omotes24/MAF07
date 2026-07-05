from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import logsumexp, softmax
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


EPS = 1e-12


def ensure_probabilities(logits_or_probs: np.ndarray) -> np.ndarray:
    arr = as_class_logits(logits_or_probs)
    if arr.ndim != 2:
        raise ValueError("Expected a 2D array")
    row_sums = arr.sum(axis=1)
    if np.all(arr >= 0) and np.allclose(row_sums, 1.0, atol=1e-5):
        return np.clip(arr, EPS, 1.0)
    return np.clip(softmax(arr, axis=1), EPS, 1.0)


def as_class_logits(logits_or_scores: np.ndarray) -> np.ndarray:
    arr = np.asarray(logits_or_scores, dtype=float)
    if arr.ndim == 1:
        return np.stack([-arr, arr], axis=1)
    return arr


def topk_accuracy(logits_or_probs: np.ndarray, y_true: np.ndarray, k: int) -> float:
    arr = np.asarray(logits_or_probs)
    y = np.asarray(y_true)
    k = min(k, arr.shape[1])
    topk = np.argpartition(-arr, kth=k - 1, axis=1)[:, :k]
    return float(np.mean([label in row for label, row in zip(y, topk, strict=True)]))


def expected_calibration_error(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 15) -> float:
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = pred == y_true
    ece = 0.0
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    for lo, hi in zip(bins[:-1], bins[1:], strict=True):
        mask = (conf > lo) & (conf <= hi)
        if not np.any(mask):
            continue
        ece += np.mean(mask) * abs(float(correct[mask].mean()) - float(conf[mask].mean()))
    return float(ece)


def brier_score_multiclass(probs: np.ndarray, y_true: np.ndarray) -> float:
    y = np.zeros_like(probs)
    y[np.arange(len(y_true)), y_true] = 1.0
    return float(np.mean(np.sum((probs - y) ** 2, axis=1)))


def closed_classification_metrics(
    logits_or_probs: np.ndarray,
    y_true: np.ndarray,
    class_names: list[str],
) -> dict[str, object]:
    probs = ensure_probabilities(logits_or_probs)
    y = np.asarray(y_true, dtype=int)
    pred = probs.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        y, pred, labels=np.arange(len(class_names)), zero_division=0
    )
    nll = -np.log(probs[np.arange(len(y)), y] + EPS).mean()
    return {
        "top1_accuracy": float(accuracy_score(y, pred)),
        "top3_accuracy": topk_accuracy(probs, y, 3),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y, pred, average="micro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "per_class_precision": dict(zip(class_names, precision.astype(float), strict=True)),
        "per_class_recall": dict(zip(class_names, recall.astype(float), strict=True)),
        "per_class_f1": dict(zip(class_names, f1.astype(float), strict=True)),
        "per_class_support": dict(zip(class_names, support.astype(int), strict=True)),
        "confusion_matrix": confusion_matrix(y, pred, labels=np.arange(len(class_names))).tolist(),
        "NLL": float(nll),
        "ECE": expected_calibration_error(probs, y),
        "Brier": brier_score_multiclass(probs, y),
    }


def fpr95(id_scores: np.ndarray, ood_scores: np.ndarray) -> float:
    id_scores = np.asarray(id_scores, dtype=float)
    ood_scores = np.asarray(ood_scores, dtype=float)
    threshold = float(np.quantile(id_scores, 0.05, method="lower"))
    return float(np.mean(ood_scores >= threshold))


def detection_error(id_scores: np.ndarray, ood_scores: np.ndarray) -> float:
    scores = np.concatenate([id_scores, ood_scores])
    thresholds = np.unique(scores)
    best = 1.0
    for threshold in thresholds:
        fnr = np.mean(id_scores < threshold)
        fpr = np.mean(ood_scores >= threshold)
        best = min(best, 0.5 * (fnr + fpr))
    return float(best)


def oscr(id_scores: np.ndarray, ood_scores: np.ndarray, correct: np.ndarray | None = None) -> float:
    id_scores = np.asarray(id_scores, dtype=float)
    ood_scores = np.asarray(ood_scores, dtype=float)
    correct = np.ones_like(id_scores, dtype=bool) if correct is None else np.asarray(correct, dtype=bool)
    thresholds = np.sort(np.unique(np.concatenate([id_scores, ood_scores])))
    xs: list[float] = []
    ys: list[float] = []
    for threshold in thresholds:
        fpr = float(np.mean(ood_scores >= threshold))
        ccr = float(np.mean((id_scores >= threshold) & correct))
        xs.append(fpr)
        ys.append(ccr)
    order = np.argsort(xs)
    return float(np.trapezoid(np.asarray(ys)[order], np.asarray(xs)[order]))


def ood_metrics(
    y_is_id: np.ndarray,
    scores: np.ndarray,
    *,
    id_correct: np.ndarray | None = None,
) -> dict[str, float]:
    y = np.asarray(y_is_id, dtype=int)
    s = np.asarray(scores, dtype=float)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("OOD metrics require both ID and OOD samples")
    id_scores = s[y == 1]
    ood_scores = s[y == 0]
    return {
        "AUROC": float(roc_auc_score(y, s)),
        "FPR95": fpr95(id_scores, ood_scores),
        "AUPR_IN": float(average_precision_score(y, s)),
        "AUPR_OUT": float(average_precision_score(1 - y, -s)),
        "DetectionError": detection_error(id_scores, ood_scores),
        "OSCR": oscr(id_scores, ood_scores, id_correct),
        "AUOSC": oscr(id_scores, ood_scores, id_correct),
        "score_mean_id": float(np.mean(id_scores)),
        "score_std_id": float(np.std(id_scores)),
        "score_mean_ood": float(np.mean(ood_scores)),
        "score_std_ood": float(np.std(ood_scores)),
    }


def logits_to_scores(logits: np.ndarray, method: str, temperature: float = 1.0) -> np.ndarray:
    z = as_class_logits(logits) / float(temperature)
    probs = softmax(z, axis=1)
    method = method.lower()
    if method == "msp":
        return probs.max(axis=1)
    if method == "entropy":
        ent = -np.sum(probs * np.log(probs + EPS), axis=1)
        return -ent
    if method == "energy":
        return temperature * logsumexp(z, axis=1)
    if method == "maxlogit":
        return z.max(axis=1)
    if method == "gen":
        return -np.sum(np.sqrt(probs + EPS), axis=1)
    if method == "odin":
        return softmax(as_class_logits(logits) / 1000.0, axis=1).max(axis=1)
    raise ValueError(f"Unknown logit score method: {method}")


def score_summary_by_class(frame: pd.DataFrame, score_col: str = "score") -> pd.DataFrame:
    return (
        frame.groupby(["class_name", "ood_label"], dropna=False)[score_col]
        .agg(["mean", "std", "count"])
        .reset_index()
    )


@dataclass(frozen=True)
class MetricResult:
    method: str
    metrics: dict[str, float]
