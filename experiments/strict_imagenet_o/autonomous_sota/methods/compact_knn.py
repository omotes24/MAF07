"""Compact train-only cosine KNN using a FAISS IVF-PQ index."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("features must have shape [N, D]")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def build_ivfpq(
    train_features: np.ndarray,
    *,
    nlist: int,
    pq_m: int,
    pq_bits: int,
    train_samples: int,
    seed: int,
):
    """Fit and populate an inner-product IVF-PQ index from ID train only."""
    import faiss

    train = normalize_rows(train_features)
    n, dimension = train.shape
    if dimension % pq_m:
        raise ValueError("feature dimension must be divisible by pq_m")
    if nlist <= 0 or nlist >= n:
        raise ValueError("nlist must be positive and smaller than the train bank")
    rng = np.random.default_rng(seed)
    count = min(int(train_samples), n)
    fit_indices = np.sort(rng.choice(n, size=count, replace=False))
    quantizer = faiss.IndexFlatIP(dimension)
    index = faiss.IndexIVFPQ(
        quantizer,
        dimension,
        int(nlist),
        int(pq_m),
        int(pq_bits),
        faiss.METRIC_INNER_PRODUCT,
    )
    index.cp.seed = int(seed)
    index.pq.cp.seed = int(seed)
    index.train(np.ascontiguousarray(train[fit_indices]))
    index.add(np.ascontiguousarray(train))
    if not index.is_trained or index.ntotal != n:
        raise RuntimeError("IVF-PQ index was not populated correctly")
    return index


def mean_neighbor_confidence(
    index,
    query_features: np.ndarray,
    *,
    k: int,
    nprobe: int,
    batch_size: int = 1024,
) -> np.ndarray:
    """Return the mean approximate cosine similarity of the top-k neighbors."""
    query = normalize_rows(query_features)
    if k <= 0 or k > index.ntotal:
        raise ValueError("invalid neighbor count")
    index.nprobe = int(nprobe)
    parts = []
    for begin in range(0, len(query), batch_size):
        distances, indices = index.search(
            np.ascontiguousarray(query[begin : begin + batch_size]), int(k)
        )
        if np.any(indices < 0):
            raise RuntimeError("IVF-PQ search returned fewer than k neighbors")
        parts.append(distances.mean(axis=1))
    return np.concatenate(parts).astype(np.float32, copy=False)


def serialized_size(index, path: Path) -> int:
    import faiss

    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))
    return path.stat().st_size
