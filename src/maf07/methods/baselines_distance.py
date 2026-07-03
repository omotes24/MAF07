from __future__ import annotations

import numpy as np
from sklearn.covariance import EmpiricalCovariance, LedoitWolf
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

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
    _, means = class_means(train_features, train_labels)
    cov = LedoitWolf().fit(train_features) if ledoit_wolf else EmpiricalCovariance().fit(train_features)
    precision = cov.precision_
    diffs = test_features[:, None, :] - means[None, :, :]
    d2 = np.einsum("ncd,dd,ncd->nc", diffs, precision, diffs)
    return -np.sqrt(np.maximum(d2, 0.0) + EPS).min(axis=1)


def rmd_score(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
) -> np.ndarray:
    _, means = class_means(train_features, train_labels)
    global_mean = train_features.mean(axis=0, keepdims=True)
    cov = LedoitWolf().fit(train_features)
    precision = cov.precision_
    class_diffs = test_features[:, None, :] - means[None, :, :]
    global_diffs = test_features - global_mean
    class_d2 = np.einsum("ncd,dd,ncd->nc", class_diffs, precision, class_diffs)
    global_d2 = np.einsum("nd,dd,nd->n", global_diffs, precision, global_diffs)
    return -(np.min(class_d2, axis=1) - global_d2)


def knn_score(train_features: np.ndarray, test_features: np.ndarray, k: int = 50) -> np.ndarray:
    k = min(int(k), len(train_features))
    nn = NearestNeighbors(n_neighbors=k, metric="cosine").fit(normalize(train_features))
    distances, _ = nn.kneighbors(normalize(test_features))
    return -distances[:, -1]


def mahalanobispp_score(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    test_features: np.ndarray,
) -> np.ndarray:
    base_test = mahalanobis_score(train_features, train_labels, test_features)
    base_val = mahalanobis_score(train_features, train_labels, val_features)
    mu = float(base_val.mean())
    sigma = float(base_val.std() + EPS)
    return (base_test - mu) / sigma

