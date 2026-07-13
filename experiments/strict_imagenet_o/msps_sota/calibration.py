"""Empirical CDF calibration using ID calibration samples only."""

from __future__ import annotations

import numpy as np


def empirical_cdf(values: np.ndarray, sorted_reference: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    reference = np.asarray(sorted_reference)
    ranks = np.searchsorted(reference, values, side="right")
    return (1.0 + ranks) / (1.0 + len(reference))


def build_sorted_calibration(
    own_support: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return class-specific [L,C,M] and global [L,C*M] sorted supports."""
    if own_support.ndim != 3:
        raise ValueError("own_support must be [layers, classes, samples]")
    class_sorted = np.sort(own_support.astype(np.float32), axis=2)
    global_sorted = np.sort(own_support.reshape(own_support.shape[0], -1), axis=1)
    return class_sorted, global_sorted


def calibrate_candidates(
    candidate_support: np.ndarray,
    candidate_classes: np.ndarray,
    class_sorted: np.ndarray,
    global_sorted: np.ndarray,
    class_shrinkage: float,
) -> np.ndarray:
    """Calibrate support shaped [N,L,K] into CDF values with the same shape."""
    if not 0.0 <= class_shrinkage <= 1.0:
        raise ValueError("class_shrinkage must be in [0, 1]")
    n_rows, n_layers, k = candidate_support.shape
    if candidate_classes.shape != (n_rows, k):
        raise ValueError("candidate class shape does not match support")
    output = np.empty_like(candidate_support, dtype=np.float32)
    for layer in range(n_layers):
        global_q = empirical_cdf(candidate_support[:, layer, :], global_sorted[layer])
        class_q = np.empty((n_rows, k), dtype=np.float32)
        for candidate_index in range(k):
            classes = candidate_classes[:, candidate_index]
            values = candidate_support[:, layer, candidate_index]
            references = class_sorted[layer, classes]
            ranks = np.sum(references <= values[:, None], axis=1)
            class_q[:, candidate_index] = (1.0 + ranks) / (1.0 + references.shape[1])
        output[:, layer, :] = (
            class_shrinkage * class_q + (1.0 - class_shrinkage) * global_q
        )
    return output
