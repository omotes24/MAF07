"""Deterministic pseudo-OOD generators using ID validation features only."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def class_partner_indices(labels: np.ndarray, shift: int = 1) -> np.ndarray:
    """Pair each sample with the same within-class offset in another class."""
    labels = np.asarray(labels, dtype=np.int64)
    classes = np.unique(labels)
    if len(classes) < 2:
        raise ValueError("at least two classes are required")
    by_class = {class_id: np.flatnonzero(labels == class_id) for class_id in classes}
    output = np.empty(len(labels), dtype=np.int64)
    for class_position, class_id in enumerate(classes):
        source = by_class[class_id]
        target_class = classes[(class_position + shift) % len(classes)]
        target = by_class[target_class]
        output[source] = target[np.arange(len(source)) % len(target)]
    if np.any(labels[output] == labels):
        raise RuntimeError("partner construction did not change class")
    return output


def generate_id_only_proxies(
    source: np.ndarray,
    labels: np.ndarray,
    calibration: np.ndarray,
    class_means: np.ndarray,
    *,
    stage_dims: Sequence[int],
) -> dict[str, np.ndarray]:
    """Create multiple fixed pseudo-OOD families without target OOD access."""
    source = np.asarray(source, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    calibration = np.asarray(calibration, dtype=np.float32)
    class_means = np.asarray(class_means, dtype=np.float32)
    if source.ndim != 2 or source.shape[1] != sum(stage_dims):
        raise ValueError("source feature shape does not match stage dimensions")
    if calibration.ndim != 2 or calibration.shape[1] != source.shape[1]:
        raise ValueError("calibration feature shape does not match source")
    if class_means.ndim != 2 or class_means.shape[1] != source.shape[1]:
        raise ValueError("class mean feature shape does not match source")

    partner = class_partner_indices(labels, shift=1)
    own_mean = class_means[labels]
    partner_mean = class_means[labels[partner]]
    output: dict[str, np.ndarray] = {
        "interclass_midpoint": 0.5 * source + 0.5 * source[partner],
        "interclass_near": 0.75 * source + 0.25 * source[partner],
        "radial_extrapolation": own_mean + 1.8 * (source - own_mean),
        "class_direction_extrapolation": source + 0.75 * (own_mean - partner_mean),
    }

    boundaries = np.cumsum((0, *stage_dims))
    incoherent_parts = []
    shifts = (1, 7, 17, 37)
    for layer, (begin, stop) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        layer_partner = class_partner_indices(labels, shift=shifts[layer % len(shifts)])
        incoherent_parts.append(source[layer_partner, begin:stop])
    output["stage_incoherence"] = np.concatenate(incoherent_parts, axis=1)

    low_variance = source.copy()
    variance = calibration.var(axis=0, ddof=1)
    for begin, stop in zip(boundaries[:-1], boundaries[1:]):
        layer_variance = variance[begin:stop]
        count = max(1, int(np.ceil(0.1 * len(layer_variance))))
        selected = np.argsort(layer_variance)[:count] + begin
        scale = np.sqrt(np.maximum(variance[selected], 1e-12))
        signs = np.where(labels % 2 == 0, 1.0, -1.0)[:, None]
        low_variance[:, selected] += 2.0 * signs * scale[None, :]
    output["low_variance_shift"] = low_variance

    channel_mask = source.copy()
    for begin, stop in zip(boundaries[:-1], boundaries[1:]):
        selected = np.arange(begin, stop, 10)
        channel_mask[:, selected] = 0.0
        channel_mask[:, begin:stop] /= 0.9
    output["channel_mask"] = channel_mask

    for name, values in output.items():
        if values.shape != source.shape or not np.isfinite(values).all():
            raise ValueError(f"invalid generated proxy {name}")
    return output
