"""Candidate selection, stage fusion, disagreement, and residual scores."""

from __future__ import annotations

import numpy as np
import torch

from feature_cache import split_stages
from prototype_stats import ReferenceStats


def stage_similarities(
    pooled_features: np.ndarray,
    prototypes: list[np.ndarray],
    *,
    device: str = "cuda",
    batch_size: int = 256,
) -> np.ndarray:
    """Return cosine support [N,L,C] using compressed class prototypes only."""
    stages = split_stages(pooled_features)
    output = np.empty((len(pooled_features), len(stages), len(prototypes[0])), dtype=np.float16)
    prototype_tensors = [torch.as_tensor(p, device=device) for p in prototypes]
    with torch.inference_mode():
        for begin in range(0, len(pooled_features), batch_size):
            stop = min(begin + batch_size, len(pooled_features))
            for layer, (stage, prototype) in enumerate(zip(stages, prototype_tensors)):
                query = torch.as_tensor(stage[begin:stop], dtype=torch.float32, device=device)
                query = torch.nn.functional.normalize(query, dim=1)
                output[begin:stop, layer] = (query @ prototype.T).cpu().half().numpy()
    return output


def select_candidates(similarity: np.ndarray, known_classes: np.ndarray, k: int) -> np.ndarray:
    if k not in (1, 3, 5):
        raise ValueError("K must be one of 1, 3, 5")
    deep = similarity[:, -1, :].astype(np.float32)
    masked = np.where(known_classes[None, :], deep, -np.inf)
    indices = np.argpartition(masked, -k, axis=1)[:, -k:]
    values = np.take_along_axis(masked, indices, axis=1)
    order = np.argsort(values, axis=1)[:, ::-1]
    return np.take_along_axis(indices, order, axis=1).astype(np.int64)


def gather_candidate_support(similarity: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    return np.stack(
        [
            np.take_along_axis(similarity[:, layer, :], candidates, axis=1)
            for layer in range(similarity.shape[1])
        ],
        axis=1,
    ).astype(np.float32)


def fuse_support(
    q: np.ndarray,
    weights: np.ndarray,
    method: str,
    *,
    softmin_tau: float = 0.1,
) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float32)
    weights = weights / weights.sum()
    shaped = weights[None, :, None]
    clipped = np.clip(q, 1e-8, 1.0)
    if method == "arithmetic_mean":
        return (clipped * shaped).sum(axis=1)
    if method == "geometric_mean":
        return np.exp((np.log(clipped) * shaped).sum(axis=1))
    if method == "harmonic_mean":
        return 1.0 / (shaped / clipped).sum(axis=1)
    if method == "soft_min":
        # Normalized smooth minimum; equal inputs map to that same value.
        return -softmin_tau * np.log(
            (np.exp(-clipped / softmin_tau) * shaped).sum(axis=1)
        )
    raise ValueError(f"unknown fusion method: {method}")


def candidate_confidence(
    q: np.ndarray,
    candidates: np.ndarray,
    stage_winners: np.ndarray,
    *,
    weights: np.ndarray,
    fusion: str,
    variance_penalty: float,
    consensus_reward: float,
) -> np.ndarray:
    fused = fuse_support(q, weights, fusion)
    log_q = np.log(np.clip(q, 1e-8, 1.0))
    variance = log_q.var(axis=1)
    consensus = np.stack(
        [(stage_winners == candidates[:, index, None]).mean(axis=1) for index in range(candidates.shape[1])],
        axis=1,
    )
    score = (
        np.log(np.clip(fused, 1e-8, None))
        - variance_penalty * variance
        + consensus_reward * consensus
    )
    return score.max(axis=1).astype(np.float32)


def original_msps_confidence(similarity: np.ndarray, known_classes: np.ndarray) -> np.ndarray:
    class_support = similarity.astype(np.float32).mean(axis=1)
    class_support[:, ~known_classes] = -np.inf
    return class_support.max(axis=1)


def diagonal_residuals(
    pooled_features: np.ndarray,
    candidates: np.ndarray,
    stats: ReferenceStats,
    *,
    rho: float,
    device: str = "cuda",
    batch_size: int = 128,
) -> np.ndarray:
    """Return class-conditional diagonal residual [N,L,K]."""
    if not 0.0 <= rho <= 1.0:
        raise ValueError("rho must be in [0, 1]")
    stages = split_stages(pooled_features)
    n_rows, k = candidates.shape
    output = np.empty((n_rows, len(stages), k), dtype=np.float32)
    for layer, stage in enumerate(stages):
        prototype = torch.as_tensor(stats.prototypes[layer], device=device)
        class_var = torch.as_tensor(stats.diagonal_variances[layer], device=device)
        global_var = torch.as_tensor(stats.global_variances[layer], device=device)
        positive = stats.global_variances[layer][stats.global_variances[layer] > 0]
        floor = max(float(np.quantile(positive, 0.001)) * 0.01, 1e-8)
        with torch.inference_mode():
            for begin in range(0, n_rows, batch_size):
                stop = min(begin + batch_size, n_rows)
                query = torch.as_tensor(stage[begin:stop], dtype=torch.float32, device=device)
                query = torch.nn.functional.normalize(query, dim=1)
                cls = torch.as_tensor(candidates[begin:stop], dtype=torch.long, device=device)
                center = prototype[cls]
                variance = (1.0 - rho) * class_var[cls] + rho * global_var[None, None, :]
                variance = variance.clamp_min(floor)
                residual = ((query[:, None, :] - center).square() / variance).mean(dim=2)
                output[begin:stop, layer] = residual.cpu().numpy()
    if not np.isfinite(output).all():
        raise ValueError("diagonal residual produced NaN or Inf")
    return output
