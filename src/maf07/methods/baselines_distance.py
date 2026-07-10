from __future__ import annotations

import numpy as np

from .verified_baselines import (
    knn_score as _verified_knn_score,
    mahalanobis_score as _verified_mahalanobis_score,
    rmd_score as _verified_rmd_score,
)

EPS = 1e-12


def class_means(features: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    classes = np.asarray(sorted(np.unique(labels)))
    means = np.vstack([features[labels == cls].mean(axis=0) for cls in classes])
    return classes, means


def mahalanobis_score(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    *,
    ledoit_wolf: bool = True,
) -> np.ndarray:
    del ledoit_wolf
    return _verified_mahalanobis_score(train_features, train_labels, test_features)


def rmd_score(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
) -> np.ndarray:
    return _verified_rmd_score(train_features, train_labels, test_features)


def knn_score(train_features: np.ndarray, test_features: np.ndarray, k: int = 50) -> np.ndarray:
    return _verified_knn_score(train_features, test_features, k=k)


def mahalanobispp_score(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    test_features: np.ndarray,
) -> np.ndarray:
    del val_features
    return _verified_mahalanobis_score(
        train_features,
        train_labels,
        test_features,
        l2_normalize=True,
    )
