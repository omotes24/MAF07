#!/usr/bin/env python3
"""ViM fitted from frozen ResNet50d classifier and ID-train features only."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import torch


FEATURE_DIM = 2048


def frozen_classifier() -> tuple[np.ndarray, np.ndarray]:
    import timm

    model = timm.create_model("resnet50d", pretrained=True).eval()
    weight = model.fc.weight.detach().cpu().numpy().astype(np.float32)
    bias = model.fc.bias.detach().cpu().numpy().astype(np.float32)
    return weight, bias


@dataclass
class VimState:
    origin: np.ndarray
    principal: np.ndarray
    alpha: float
    weight: np.ndarray
    bias: np.ndarray
    class_ids: np.ndarray
    dimension: int
    n_fit: int

    def score_ood(
        self,
        features: np.ndarray,
        *,
        device: str = "cuda",
        batch_size: int = 512,
    ) -> np.ndarray:
        """Return higher-is-OOD ViM score: alpha*residual - energy."""
        weight = torch.as_tensor(self.weight, dtype=torch.float32, device=device)
        bias = torch.as_tensor(self.bias, dtype=torch.float32, device=device)
        origin = torch.as_tensor(self.origin, dtype=torch.float32, device=device)
        principal = torch.as_tensor(self.principal, dtype=torch.float32, device=device)
        output = []
        with torch.inference_mode():
            for begin in range(0, len(features), batch_size):
                x = torch.as_tensor(
                    features[begin : begin + batch_size], dtype=torch.float32, device=device
                )
                logits = x @ weight.T + bias
                energy = torch.logsumexp(logits, dim=1)
                centered = x - origin
                residual_sq = centered.square().sum(dim=1) - (centered @ principal).square().sum(dim=1)
                residual = residual_sq.clamp_min(0).sqrt()
                output.append((self.alpha * residual - energy).cpu().numpy())
        scores = np.concatenate(output).astype(np.float32)
        if not np.isfinite(scores).all():
            raise ValueError("ViM produced NaN or Inf")
        return scores

    def payload(self, prefix: str = "vim_") -> dict[str, np.ndarray]:
        return {
            f"{prefix}origin": self.origin.astype(np.float32),
            f"{prefix}principal": self.principal.astype(np.float32),
            f"{prefix}alpha": np.asarray(self.alpha, dtype=np.float64),
            f"{prefix}weight": self.weight.astype(np.float32),
            f"{prefix}bias": self.bias.astype(np.float32),
            f"{prefix}class_ids": self.class_ids.astype(np.int64),
            f"{prefix}dimension": np.asarray(self.dimension, dtype=np.int64),
            f"{prefix}n_fit": np.asarray(self.n_fit, dtype=np.int64),
        }

    @classmethod
    def from_mapping(cls, values: Mapping[str, np.ndarray], prefix: str = "vim_") -> "VimState":
        return cls(
            origin=np.asarray(values[f"{prefix}origin"], dtype=np.float32),
            principal=np.asarray(values[f"{prefix}principal"], dtype=np.float32),
            alpha=float(values[f"{prefix}alpha"]),
            weight=np.asarray(values[f"{prefix}weight"], dtype=np.float32),
            bias=np.asarray(values[f"{prefix}bias"], dtype=np.float32),
            class_ids=np.asarray(values[f"{prefix}class_ids"], dtype=np.int64),
            dimension=int(values[f"{prefix}dimension"]),
            n_fit=int(values[f"{prefix}n_fit"]),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **self.payload())


def fit_vim(
    train_features: np.ndarray,
    classifier_weight: np.ndarray,
    classifier_bias: np.ndarray,
    class_ids: np.ndarray,
    *,
    dimension: int = 1000,
    device: str = "cuda",
    covariance_batch_size: int = 4096,
) -> VimState:
    """Fit official-form ViM origin, principal subspace, and alpha on ID train only."""
    class_ids = np.asarray(class_ids, dtype=np.int64)
    weight_np = np.asarray(classifier_weight[class_ids], dtype=np.float32)
    bias_np = np.asarray(classifier_bias[class_ids], dtype=np.float32)
    if train_features.ndim != 2 or train_features.shape[1] != FEATURE_DIM:
        raise ValueError(f"ViM expects [N,{FEATURE_DIM}] final-stage features")
    dimension = min(int(dimension), FEATURE_DIM - 1)
    weight = torch.as_tensor(weight_np, dtype=torch.float32, device=device)
    bias = torch.as_tensor(bias_np, dtype=torch.float32, device=device)
    with torch.inference_mode():
        origin = -(torch.linalg.pinv(weight) @ bias)
        covariance = torch.zeros((FEATURE_DIM, FEATURE_DIM), dtype=torch.float32, device=device)
        for begin in range(0, len(train_features), covariance_batch_size):
            x = torch.as_tensor(
                train_features[begin : begin + covariance_batch_size],
                dtype=torch.float32,
                device=device,
            )
            centered = x - origin
            covariance.addmm_(centered.T, centered)
        covariance /= max(len(train_features) - 1, 1)
        _, eigenvectors = torch.linalg.eigh(covariance)
        principal = eigenvectors[:, -dimension:].contiguous()
        del covariance, eigenvectors

        maxlogit_sum = 0.0
        residual_sum = 0.0
        for begin in range(0, len(train_features), 512):
            x = torch.as_tensor(
                train_features[begin : begin + 512], dtype=torch.float32, device=device
            )
            logits = x @ weight.T + bias
            centered = x - origin
            residual_sq = centered.square().sum(dim=1) - (centered @ principal).square().sum(dim=1)
            residual = residual_sq.clamp_min(0).sqrt()
            maxlogit_sum += float(logits.amax(dim=1).sum())
            residual_sum += float(residual.sum())
        mean_residual = residual_sum / len(train_features)
        if mean_residual <= 0:
            raise ValueError("ViM mean residual is not positive")
        alpha = (maxlogit_sum / len(train_features)) / mean_residual
    return VimState(
        origin=origin.cpu().numpy().astype(np.float32),
        principal=principal.cpu().numpy().astype(np.float32),
        alpha=float(alpha),
        weight=weight_np,
        bias=bias_np,
        class_ids=class_ids,
        dimension=dimension,
        n_fit=len(train_features),
    )
