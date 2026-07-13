"""Coefficient-free fusion of global, angular, and radial ID support."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


GLOBAL_COMPONENTS = ("rc_msps", "deep_support", "msps", "proto_relative")
EPSILON = 1e-12


def empirical_id_quantile(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Map a higher-is-ID score to its smoothed empirical ID CDF."""
    reference = np.sort(np.asarray(reference, dtype=np.float64).reshape(-1))
    values = np.asarray(values, dtype=np.float64)
    if reference.size == 0:
        raise ValueError("empirical reference must not be empty")
    return (1.0 + np.searchsorted(reference, values, side="right")) / (
        1.0 + reference.size
    )


def activation_dispersion(final_features: np.ndarray) -> np.ndarray:
    """Higher-is-ID radial evidence from within-vector activation concentration."""
    features = np.asarray(final_features, dtype=np.float32)
    if features.ndim != 2:
        raise ValueError("final_features must have shape [N, D]")
    return -features.std(axis=1, dtype=np.float64)


def robust_global_support(quantiles: Mapping[str, np.ndarray]) -> np.ndarray:
    """Combine four global supports while requiring two independent strong signals."""
    matrix = np.column_stack([quantiles[name] for name in GLOBAL_COMPONENTS]).astype(
        np.float64
    )
    matrix = np.clip(matrix, EPSILON, 1.0)
    ordered = np.sort(matrix, axis=1)
    lower_pair = np.sqrt(ordered[:, 0] * ordered[:, 1])
    all_four = np.prod(matrix, axis=1) ** 0.25
    return np.sqrt(lower_pair * all_four)


def weighted_support_confidence(
    global_quantiles: Mapping[str, np.ndarray],
    knn_quantile: np.ndarray,
    radial_quantile: np.ndarray,
    *,
    lower_tail_weight: float,
    angular_weight: float,
    global_weight: float,
) -> np.ndarray:
    """Fuse three supports with exponents selected exclusively by ID-only proxies."""
    weights = (lower_tail_weight, angular_weight, global_weight)
    if any(weight < 0.0 or weight > 1.0 for weight in weights):
        raise ValueError("support weights must lie in [0, 1]")
    matrix = np.column_stack(
        [global_quantiles[name] for name in GLOBAL_COMPONENTS]
    ).astype(np.float64)
    matrix = np.clip(matrix, EPSILON, 1.0)
    ordered = np.sort(matrix, axis=1)
    lower_pair = np.sqrt(ordered[:, 0] * ordered[:, 1])
    all_four = np.prod(matrix, axis=1) ** 0.25
    global_support = lower_pair**lower_tail_weight * all_four ** (
        1.0 - lower_tail_weight
    )
    angular = np.clip(np.asarray(knn_quantile, dtype=np.float64), EPSILON, 1.0)
    radial = np.clip(np.asarray(radial_quantile, dtype=np.float64), EPSILON, 1.0)
    local_support = angular**angular_weight * radial ** (1.0 - angular_weight)
    if global_support.shape != local_support.shape:
        raise ValueError("global and local support shapes differ")
    return global_support**global_weight * local_support ** (1.0 - global_weight)


def symmetric_support_confidence(
    global_quantiles: Mapping[str, np.ndarray],
    knn_quantile: np.ndarray,
    radial_quantile: np.ndarray,
) -> np.ndarray:
    """Return the final higher-is-ID confidence using three symmetric supports."""
    global_support = robust_global_support(global_quantiles)
    angular_radial = np.sqrt(
        np.clip(np.asarray(knn_quantile, dtype=np.float64), EPSILON, 1.0)
        * np.clip(np.asarray(radial_quantile, dtype=np.float64), EPSILON, 1.0)
    )
    if global_support.shape != angular_radial.shape:
        raise ValueError("global and local support shapes differ")
    return np.sqrt(global_support * angular_radial)


def score_from_raw(
    calibration: Mapping[str, np.ndarray],
    values: Mapping[str, np.ndarray],
) -> np.ndarray:
    """Calibrate ID-only atomic scores and compute symmetric support confidence."""
    required = (*GLOBAL_COMPONENTS, "knn", "radial")
    missing = [name for name in required if name not in calibration or name not in values]
    if missing:
        raise KeyError(f"missing symmetric-support components: {missing}")
    q = {
        name: empirical_id_quantile(calibration[name], values[name]) for name in required
    }
    return symmetric_support_confidence(q, q["knn"], q["radial"])


def weighted_score_from_raw(
    calibration: Mapping[str, np.ndarray],
    values: Mapping[str, np.ndarray],
    *,
    lower_tail_weight: float,
    angular_weight: float,
    global_weight: float,
) -> np.ndarray:
    """Calibrate raw scores and apply an ID-only-selected weighted fusion."""
    required = (*GLOBAL_COMPONENTS, "knn", "radial")
    missing = [name for name in required if name not in calibration or name not in values]
    if missing:
        raise KeyError(f"missing weighted-support components: {missing}")
    q = {
        name: empirical_id_quantile(calibration[name], values[name]) for name in required
    }
    return weighted_support_confidence(
        q,
        q["knn"],
        q["radial"],
        lower_tail_weight=lower_tail_weight,
        angular_weight=angular_weight,
        global_weight=global_weight,
    )
