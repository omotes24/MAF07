from __future__ import annotations

import numpy as np
from scipy.special import logsumexp, softmax

EPS = 1e-12


def react_features(train_features: np.ndarray, features: np.ndarray, percentile: float = 90.0) -> np.ndarray:
    threshold = np.percentile(train_features, percentile, axis=0, keepdims=True)
    return np.minimum(features, threshold)


def ash_features(features: np.ndarray, keep_fraction: float = 0.1, variant: str = "p") -> np.ndarray:
    x = np.asarray(features, dtype=float).copy()
    k = max(1, int(x.shape[1] * keep_fraction))
    idx = np.argpartition(np.abs(x), kth=x.shape[1] - k, axis=1)[:, : x.shape[1] - k]
    rows = np.arange(x.shape[0])[:, None]
    removed = x[rows, idx].sum(axis=1, keepdims=True)
    x[rows, idx] = 0.0
    if variant.lower() == "s":
        kept_sum = np.sum(np.abs(x), axis=1, keepdims=True) + EPS
        x = x * ((kept_sum + np.abs(removed)) / kept_sum)
    elif variant.lower() == "b":
        kept = x != 0
        mean_val = np.sum(np.abs(x), axis=1, keepdims=True) / (np.sum(kept, axis=1, keepdims=True) + EPS)
        x = np.where(kept, mean_val * np.sign(x), 0.0)
    return x


def dice_features(train_features: np.ndarray, features: np.ndarray, keep_fraction: float = 0.9) -> np.ndarray:
    importance = np.var(train_features, axis=0)
    keep = max(1, int(features.shape[1] * keep_fraction))
    threshold = np.partition(importance, -keep)[-keep]
    mask = importance >= threshold
    return features * mask[None, :]


def feature_energy(features: np.ndarray, prototypes: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    logits = np.asarray(features) @ np.asarray(prototypes).T
    return temperature * logsumexp(logits / temperature, axis=1)


def scale_score(features: np.ndarray, prototypes: np.ndarray) -> np.ndarray:
    logits = np.asarray(features) @ np.asarray(prototypes).T
    probs = softmax(logits, axis=1)
    norm = np.linalg.norm(features, axis=1)
    return probs.max(axis=1) * np.log1p(norm)


def nci_score(features: np.ndarray, prototypes: np.ndarray) -> np.ndarray:
    logits = np.asarray(features) @ np.asarray(prototypes).T
    probs = softmax(logits, axis=1)
    margin = np.partition(logits, -2, axis=1)[:, -1] - np.partition(logits, -2, axis=1)[:, -2]
    concentration = np.sum(probs**2, axis=1)
    return margin * concentration

