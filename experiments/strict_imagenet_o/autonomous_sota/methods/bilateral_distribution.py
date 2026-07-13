"""Class-center distribution evidence for compact inductive OOD scoring."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


COMPONENT_ORIENTATION = {
    "prototype_spread": 1.0,
    "top_background_contrast": 1.0,
    "top10_background_contrast": 1.0,
    "standardized_peak": 1.0,
    "top_gap": 1.0,
    "uniform_cosine": -1.0,
}


@torch.no_grad()
def bilateral_components(
    features: np.ndarray,
    prototypes: np.ndarray,
    *,
    device: str = "cuda:0",
    batch_size: int = 512,
) -> dict[str, np.ndarray]:
    """Summarize each query's similarities to all compact class centers.

    The ideal-OOD premise is that an OOD query has nearly equal support from
    every class center. The returned statistics quantify departure from that
    uniform-support vector without retaining an image-level feature bank.
    """
    feature = np.asarray(features)
    prototype = np.asarray(prototypes)
    if feature.ndim != 2 or prototype.ndim != 2 or feature.shape[1] != prototype.shape[1]:
        raise ValueError("features and prototypes must be compatible matrices")
    if len(prototype) < 2:
        raise ValueError("bilateral evidence requires at least two class centers")
    center = F.normalize(
        torch.as_tensor(prototype, dtype=torch.float32, device=device), dim=1
    )
    output = {name: [] for name in COMPONENT_ORIENTATION}
    for start in range(0, len(feature), batch_size):
        query = F.normalize(
            torch.as_tensor(feature[start : start + batch_size], dtype=torch.float32, device=device),
            dim=1,
        )
        similarity = query @ center.T
        mean = similarity.mean(dim=1)
        centered = similarity - mean[:, None]
        spread = centered.square().mean(dim=1).sqrt()
        top = torch.topk(similarity, min(10, len(prototype)), dim=1).values
        peak = top[:, 0]
        rms = similarity.square().mean(dim=1).sqrt().clamp_min(1e-12)
        values = {
            "prototype_spread": spread,
            "top_background_contrast": peak - mean,
            "top10_background_contrast": top.mean(dim=1) - mean,
            "standardized_peak": (peak - mean) / spread.clamp_min(1e-12),
            "top_gap": top[:, 0] - top[:, 1],
            "uniform_cosine": mean / rms,
        }
        for name, value in values.items():
            output[name].append(value.cpu().numpy())
    result = {name: np.concatenate(parts).astype(np.float32) for name, parts in output.items()}
    if any(values.shape != (len(feature),) or not np.isfinite(values).all() for values in result.values()):
        raise ValueError("bilateral component extraction produced invalid values")
    return result
