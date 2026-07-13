"""Per-image activation-shift scores for strict inductive OOD detection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


PERTURB_FRACTION = 0.05
PERTURB_MAGNITUDE = 0.5
ADASCALE_K1 = 0.01
ADASCALE_K2 = 0.05
ADASCALE_LAMBDA = 10.0
ADASCALE_PERCENTILE = (75.0, 85.0)


def low_gradient_perturb(
    images: torch.Tensor,
    gradient: torch.Tensor,
    *,
    fraction: float = PERTURB_FRACTION,
    magnitude: float = PERTURB_MAGNITUDE,
) -> torch.Tensor:
    """Perturb the fixed fraction of input coordinates with smallest |gradient|."""
    if images.shape != gradient.shape or images.ndim != 4:
        raise ValueError("images and gradient must have the same [N,C,H,W] shape")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    flat = gradient.abs().flatten(1)
    count = max(1, int(flat.shape[1] * fraction))
    indices = torch.topk(flat, count, dim=1, largest=False).indices
    mask = torch.zeros_like(flat)
    mask.scatter_(1, indices, 1.0)
    mask = mask.reshape_as(images)
    return (images.detach() + magnitude * gradient.sign() * mask).detach()


def _gather_top_fraction(
    source: torch.Tensor, values: torch.Tensor, fraction: float
) -> torch.Tensor:
    count = max(1, int(source.shape[1] * fraction))
    indices = torch.topk(source, count, dim=1).indices
    return torch.gather(values, 1, indices)


@torch.no_grad()
def activation_shift_statistics(
    feature: torch.Tensor,
    perturbed_feature: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return official AdaSCALE atoms and a scale-free shift ratio.

    The official repository applies k1 and k2 in opposite positions while
    constructing the ID ECDF and while querying it. Both quantities are kept
    explicitly so the reproduction remains faithful without propagating that
    behavior into ShiftRC.
    """
    if feature.shape != perturbed_feature.shape or feature.ndim != 2:
        raise ValueError("features must have the same [N,D] shape")
    shift = (feature - perturbed_feature).abs()
    positive_perturbed = perturbed_feature.relu()

    # Official setup()/set_hyperparam() behavior.
    fit_correction = _gather_top_fraction(feature, positive_perturbed, ADASCALE_K2).sum(1)
    fit_shift = _gather_top_fraction(feature, shift, ADASCALE_K1).sum(1)
    qprime_fit = fit_correction + ADASCALE_LAMBDA * fit_shift

    # Official get_percentile() behavior.
    eval_correction = _gather_top_fraction(feature, positive_perturbed, ADASCALE_K1).sum(1)
    eval_shift = _gather_top_fraction(feature, shift, ADASCALE_K2).sum(1)
    qprime_eval = eval_correction + ADASCALE_LAMBDA * eval_shift

    top_shift = _gather_top_fraction(feature, shift, ADASCALE_K2).sum(1)
    top_signal = _gather_top_fraction(feature, feature.relu(), ADASCALE_K2).sum(1)
    shift_ratio = top_shift / top_signal.clamp_min(1e-12)
    return {
        "qprime_fit": qprime_fit,
        "qprime_eval": qprime_eval,
        "shift_ratio": shift_ratio,
    }


def robust_location_scale(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("robust calibration values must be finite and one-dimensional")
    center = float(np.median(values))
    scale = float(1.4826 * np.median(np.abs(values - center)))
    return center, max(scale, 1e-6)


@dataclass(frozen=True)
class ShiftRCState:
    rc_center: float
    rc_scale: float
    shift_center: float
    shift_scale: float

    def confidence(self, rc_confidence: np.ndarray, shift_ratio: np.ndarray) -> np.ndarray:
        """Coefficient-free sum of two robustly standardized OOD atoms."""
        rc_uncertainty = -np.asarray(rc_confidence, dtype=np.float64)
        shift_uncertainty = np.log1p(np.asarray(shift_ratio, dtype=np.float64))
        rc_z = (rc_uncertainty - self.rc_center) / self.rc_scale
        shift_z = (shift_uncertainty - self.shift_center) / self.shift_scale
        result = -(rc_z + shift_z)
        if result.ndim != 1 or not np.isfinite(result).all():
            raise ValueError("ShiftRC produced invalid confidence")
        return result.astype(np.float32)


def fit_shift_rc(rc_confidence: np.ndarray, shift_ratio: np.ndarray) -> ShiftRCState:
    rc_center, rc_scale = robust_location_scale(-np.asarray(rc_confidence, dtype=np.float64))
    shift_center, shift_scale = robust_location_scale(
        np.log1p(np.asarray(shift_ratio, dtype=np.float64))
    )
    return ShiftRCState(rc_center, rc_scale, shift_center, shift_scale)


def empirical_cdf(sorted_reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    reference = np.asarray(sorted_reference, dtype=np.float64)
    if reference.ndim != 1 or len(reference) == 0:
        raise ValueError("ECDF reference must be a nonempty vector")
    return np.searchsorted(reference, values, side="right") / float(len(reference))


def adascale_confidence(
    features: np.ndarray,
    qprime_eval: np.ndarray,
    qprime_fit_sorted: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    *,
    logit_scaling: bool,
    batch_size: int = 512,
) -> np.ndarray:
    """Reproduce AdaSCALE-A/L from cached penultimate features and Q-prime."""
    feature = np.asarray(features, dtype=np.float32)
    qprime = np.asarray(qprime_eval, dtype=np.float64)
    if feature.ndim != 2 or qprime.shape != (len(feature),):
        raise ValueError("invalid AdaSCALE input shapes")
    quantile = empirical_cdf(qprime_fit_sorted, qprime)
    lo, hi = ADASCALE_PERCENTILE
    percentiles = lo + (1.0 - quantile) * (hi - lo)
    output: list[np.ndarray] = []
    weight64 = np.asarray(weight, dtype=np.float64)
    bias64 = np.asarray(bias, dtype=np.float64)
    dimension = feature.shape[1]
    retained = dimension - np.rint(dimension * percentiles / 100.0).astype(np.int64)
    retained = np.clip(retained, 1, dimension)
    for begin in range(0, len(feature), batch_size):
        stop = min(begin + batch_size, len(feature))
        current = feature[begin:stop].astype(np.float64)
        positive = np.maximum(current, 0.0)
        ordered = np.sort(positive, axis=1)[:, ::-1]
        cumulative = np.cumsum(ordered, axis=1)
        count = retained[begin:stop]
        top_sum = cumulative[np.arange(len(current)), count - 1]
        scale = positive.sum(axis=1) / np.maximum(top_sum, 1e-12)
        logits = current @ weight64.T + bias64
        if logit_scaling:
            logits *= scale[:, None] ** 2.0
        else:
            scaled = current * np.exp(np.minimum(scale, 80.0))[:, None]
            logits = scaled @ weight64.T + bias64
        maximum = logits.max(axis=1)
        energy = maximum + np.log(np.exp(logits - maximum[:, None]).sum(axis=1))
        output.append(energy.astype(np.float32))
    result = np.concatenate(output)
    if not np.isfinite(result).all():
        raise ValueError("AdaSCALE produced non-finite confidence")
    return result
