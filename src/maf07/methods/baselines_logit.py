from __future__ import annotations

import numpy as np
from scipy.special import logsumexp, softmax

from ..metrics import as_class_logits

EPS = 1e-12


def msp(logits: np.ndarray) -> np.ndarray:
    return softmax(as_class_logits(logits), axis=1).max(axis=1)


def entropy(logits: np.ndarray) -> np.ndarray:
    p = softmax(as_class_logits(logits), axis=1)
    return np.sum(p * np.log(p + EPS), axis=1)


def energy(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = as_class_logits(logits) / float(temperature)
    return temperature * logsumexp(z, axis=1)


def maxlogit(logits: np.ndarray) -> np.ndarray:
    return as_class_logits(logits).max(axis=1)


def odin(logits: np.ndarray, temperature: float = 1000.0) -> np.ndarray:
    return softmax(as_class_logits(logits) / float(temperature), axis=1).max(axis=1)


def gradnorm(
    logits: np.ndarray,
    features: np.ndarray | None = None,
    temperature: float = 1.0,
) -> np.ndarray:
    if features is None:
        raise ValueError("GradNorm requires penultimate features as well as logits")
    z = as_class_logits(logits) / float(temperature)
    p = softmax(z, axis=1)
    class_term = np.sum(np.abs(p * p.shape[1] - 1.0), axis=1)
    feature_term = np.sum(np.abs(np.asarray(features, dtype=float)), axis=1)
    return class_term * feature_term / float(temperature)


def kl_matching(logits: np.ndarray, train_logits_by_class: dict[int, np.ndarray]) -> np.ndarray:
    p = softmax(as_class_logits(logits), axis=1)
    templates = []
    for cls in sorted(train_logits_by_class):
        templates.append(softmax(as_class_logits(train_logits_by_class[cls]), axis=1).mean(axis=0))
    q = np.clip(np.vstack(templates), EPS, 1.0)
    kl = np.sum(p[:, None, :] * (np.log(p[:, None, :] + EPS) - np.log(q[None, :, :])), axis=2)
    return -kl.min(axis=1)


def gen(logits: np.ndarray, gamma: float = 0.1) -> np.ndarray:
    p = softmax(as_class_logits(logits), axis=1)
    return -np.sum(np.power(p + EPS, gamma) * np.power(1.0 - p + EPS, gamma), axis=1)
