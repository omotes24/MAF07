"""Compressed ID statistics for multi-stage prototype support."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from feature_cache import STAGE_DIMS, split_stages


def normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norm = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norm <= 0):
        raise ValueError(f"found {int(np.sum(norm <= 0))} zero-norm feature rows")
    return values / norm


def robust_center(values: np.ndarray, estimator: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if estimator == "mean":
        return values.mean(axis=0)
    if estimator == "trimmed_mean":
        trim = max(1, int(np.floor(0.05 * len(values))))
        ordered = np.sort(values, axis=0)
        return ordered[trim:-trim].mean(axis=0)
    if estimator == "huber":
        center = np.median(values, axis=0)
        scale = 1.4826 * np.median(np.abs(values - center), axis=0)
        scale = np.maximum(scale, 1e-4)
        bound = 1.345 * scale
        for _ in range(4):
            center = center + np.clip(values - center, -bound, bound).mean(axis=0)
        return center
    raise ValueError(f"unknown prototype estimator: {estimator}")


@dataclass
class ReferenceStats:
    prototypes: list[np.ndarray]
    diagonal_variances: list[np.ndarray]
    global_variances: list[np.ndarray]
    counts: np.ndarray
    estimator: str

    def save(self, path: Path, **extra: np.ndarray) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, np.ndarray] = {
            "stage_dims": np.asarray(STAGE_DIMS, dtype=np.int64),
            "counts": self.counts.astype(np.int64),
            "estimator": np.asarray(self.estimator),
        }
        for layer, (prototype, variance, global_variance) in enumerate(
            zip(self.prototypes, self.diagonal_variances, self.global_variances)
        ):
            payload[f"prototype_{layer}"] = prototype.astype(np.float32)
            payload[f"diagonal_variance_{layer}"] = variance.astype(np.float32)
            payload[f"global_variance_{layer}"] = global_variance.astype(np.float32)
        payload.update(extra)
        np.savez(path, **payload)

    @classmethod
    def load(cls, path: Path) -> tuple["ReferenceStats", dict[str, np.ndarray]]:
        with np.load(path, allow_pickle=False) as data:
            dims = tuple(int(value) for value in data["stage_dims"])
            if dims != STAGE_DIMS:
                raise ValueError(f"unexpected stage dimensions in state: {dims}")
            stats = cls(
                prototypes=[data[f"prototype_{i}"].astype(np.float32) for i in range(4)],
                diagonal_variances=[
                    data[f"diagonal_variance_{i}"].astype(np.float32) for i in range(4)
                ],
                global_variances=[
                    data[f"global_variance_{i}"].astype(np.float32) for i in range(4)
                ],
                counts=data["counts"].astype(np.int64),
                estimator=str(data["estimator"]),
            )
            reserved = {"stage_dims", "counts", "estimator"}
            reserved.update({f"prototype_{i}" for i in range(4)})
            reserved.update({f"diagonal_variance_{i}" for i in range(4)})
            reserved.update({f"global_variance_{i}" for i in range(4)})
            extra = {key: data[key] for key in data.files if key not in reserved}
        return stats, extra


def fit_reference_stats(
    pooled_features: np.ndarray,
    labels: np.ndarray,
    *,
    estimator: str = "mean",
    num_classes: int = 1000,
) -> ReferenceStats:
    labels = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(labels, minlength=num_classes)
    if len(counts) != num_classes or np.any(counts == 0):
        raise ValueError("every class must have prototype samples")
    prototypes, class_variances, global_variances = [], [], []
    for stage in split_stages(pooled_features):
        normalized = normalize_rows(stage)
        prototype = np.empty((num_classes, normalized.shape[1]), dtype=np.float32)
        variance = np.empty_like(prototype)
        for class_id in range(num_classes):
            values = normalized[labels == class_id]
            center = robust_center(values, estimator)
            center_norm = np.linalg.norm(center)
            if center_norm <= 0:
                raise ValueError(f"zero prototype for class {class_id}")
            prototype[class_id] = center / center_norm
            variance[class_id] = values.var(axis=0, ddof=1)
        prototypes.append(prototype)
        class_variances.append(variance)
        global_variances.append(normalized.var(axis=0, ddof=1))
    return ReferenceStats(prototypes, class_variances, global_variances, counts, estimator)
