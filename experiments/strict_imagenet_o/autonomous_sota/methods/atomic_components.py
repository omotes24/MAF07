"""Reusable, higher-is-ID atomic scores for compact OOD formula search."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


def class_empirical_cdf(values: np.ndarray, classes: np.ndarray, references: np.ndarray) -> np.ndarray:
    """Apply per-class empirical CDF references to arrays shaped [N, K]."""
    values = np.asarray(values, dtype=np.float32)
    classes = np.asarray(classes, dtype=np.int64)
    refs = references[classes]
    return ((refs <= values[..., None]).sum(axis=-1) + 1.0) / (refs.shape[-1] + 1.0)


def logits_numpy(
    features: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    *,
    device: str,
    batch_size: int = 512,
) -> np.ndarray:
    output: list[np.ndarray] = []
    weight_t = torch.as_tensor(weight, dtype=torch.float32, device=device)
    bias_t = torch.as_tensor(bias, dtype=torch.float32, device=device)
    with torch.inference_mode():
        for begin in range(0, len(features), batch_size):
            x = torch.as_tensor(features[begin : begin + batch_size], dtype=torch.float32, device=device)
            output.append((x @ weight_t.T + bias_t).cpu().numpy())
    return np.concatenate(output).astype(np.float32, copy=False)


@dataclass
class AtomicFit:
    own_proto_sorted: np.ndarray
    own_logit_sorted: np.ndarray
    own_margin_sorted: np.ndarray
    norm_median: np.ndarray
    norm_mad: np.ndarray
    global_centers: list[np.ndarray]

    def payload(self) -> dict[str, np.ndarray]:
        output = {
            "own_proto_sorted": self.own_proto_sorted,
            "own_logit_sorted": self.own_logit_sorted,
            "own_margin_sorted": self.own_margin_sorted,
            "norm_median": self.norm_median,
            "norm_mad": self.norm_mad,
        }
        output.update({f"global_center_{i}": value for i, value in enumerate(self.global_centers)})
        return output

    @classmethod
    def from_mapping(cls, values: dict[str, np.ndarray]) -> "AtomicFit":
        centers = []
        index = 0
        while f"global_center_{index}" in values:
            centers.append(np.asarray(values[f"global_center_{index}"], dtype=np.float32))
            index += 1
        if not centers:
            raise KeyError("atomic state has no global centers")
        return cls(
            own_proto_sorted=np.asarray(values["own_proto_sorted"], dtype=np.float32),
            own_logit_sorted=np.asarray(values["own_logit_sorted"], dtype=np.float32),
            own_margin_sorted=np.asarray(values["own_margin_sorted"], dtype=np.float32),
            norm_median=np.asarray(values["norm_median"], dtype=np.float32),
            norm_mad=np.asarray(values["norm_mad"], dtype=np.float32),
            global_centers=centers,
        )


def fit_atomic_references(
    features: np.ndarray,
    labels: np.ndarray,
    similarity: np.ndarray,
    logits: np.ndarray,
    stage_weights: np.ndarray,
    stage_dims: tuple[int, ...],
    *,
    num_classes: int = 1000,
) -> AtomicFit:
    labels = np.asarray(labels, dtype=np.int64)
    weighted = np.einsum(
        "nlc,l->nc", similarity.astype(np.float32), stage_weights.astype(np.float32)
    )
    rows = np.arange(len(features))
    own_proto = weighted[rows, labels]
    own_logit = logits[rows, labels]
    masked = weighted.copy()
    masked[rows, labels] = -np.inf
    own_margin = own_proto - masked.max(axis=1)
    final_norm = np.linalg.norm(features[:, -stage_dims[-1] :].astype(np.float32), axis=1)

    proto_refs, logit_refs, margin_refs = [], [], []
    norm_median, norm_mad = [], []
    for class_id in range(num_classes):
        selected = labels == class_id
        if not np.any(selected):
            raise ValueError(f"missing atomic calibration class {class_id}")
        proto_refs.append(np.sort(own_proto[selected]))
        logit_refs.append(np.sort(own_logit[selected]))
        margin_refs.append(np.sort(own_margin[selected]))
        class_norm = final_norm[selected]
        median = np.median(class_norm)
        norm_median.append(median)
        norm_mad.append(max(1.4826 * np.median(np.abs(class_norm - median)), 1e-6))

    global_centers = []
    boundaries = np.cumsum((0, *stage_dims))
    for begin, stop in zip(boundaries[:-1], boundaries[1:]):
        stage = features[:, begin:stop].astype(np.float32)
        stage /= np.maximum(np.linalg.norm(stage, axis=1, keepdims=True), 1e-12)
        center = stage.mean(axis=0)
        center /= max(float(np.linalg.norm(center)), 1e-12)
        global_centers.append(center.astype(np.float32))
    return AtomicFit(
        own_proto_sorted=np.stack(proto_refs).astype(np.float32),
        own_logit_sorted=np.stack(logit_refs).astype(np.float32),
        own_margin_sorted=np.stack(margin_refs).astype(np.float32),
        norm_median=np.asarray(norm_median, dtype=np.float32),
        norm_mad=np.asarray(norm_mad, dtype=np.float32),
        global_centers=global_centers,
    )


def geometric_barycentric_confidence(
    final_features: np.ndarray,
    top_classes: np.ndarray,
    top_support: np.ndarray,
    prototypes: np.ndarray,
    *,
    device: str,
    batch_size: int = 256,
    temperature: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return barycentric, Fisher, and classifier-independent boundary primitives."""
    barycentric, normalized = [], []
    proto_t = torch.as_tensor(prototypes, dtype=torch.float32, device=device)
    with torch.inference_mode():
        for begin in range(0, len(final_features), batch_size):
            stop = min(begin + batch_size, len(final_features))
            x = torch.as_tensor(final_features[begin:stop], dtype=torch.float32, device=device)
            unit = F.normalize(x, dim=1)
            classes = torch.as_tensor(top_classes[begin:stop, :3], dtype=torch.long, device=device)
            centers = proto_t[classes]
            support = torch.as_tensor(top_support[begin:stop, :3], dtype=torch.float32, device=device)
            mixing = torch.softmax(support / temperature, dim=1)
            reconstruction = F.normalize((mixing[..., None] * centers).sum(1), dim=1)
            barycentric.append(-(unit - reconstruction).square().mean(1).cpu().numpy())
            normalized.append(unit.cpu().numpy())
    unit_features = np.concatenate(normalized).astype(np.float32, copy=False)
    return np.concatenate(barycentric).astype(np.float32), unit_features, top_classes[:, 0]
