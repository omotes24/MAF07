"""NNGuide using a fixed classifier and an ID-train feature bank.

This follows the official NNGuide scoring rule: each normalized train feature is
scaled by its energy, the query guidance is the mean top-k inner product against
that bank, and ID confidence is query energy times guidance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class NNGuideState:
    """Compact fitted NNGuide state; all arrays are estimated from ID train."""

    scaled_bank: np.ndarray
    weight: np.ndarray
    bias: np.ndarray
    k: int = 10

    def __post_init__(self) -> None:
        if self.scaled_bank.ndim != 2:
            raise ValueError("scaled_bank must have shape [N, D]")
        if self.weight.ndim != 2 or self.weight.shape[1] != self.scaled_bank.shape[1]:
            raise ValueError("classifier and feature dimensions do not match")
        if self.bias.shape != (self.weight.shape[0],):
            raise ValueError("classifier bias has the wrong shape")
        if not 1 <= self.k <= len(self.scaled_bank):
            raise ValueError("k must be between 1 and bank size")

    @classmethod
    def fit(
        cls,
        train_features: np.ndarray,
        weight: np.ndarray,
        bias: np.ndarray,
        *,
        k: int = 10,
        device: str = "cuda",
        batch_size: int = 512,
    ) -> "NNGuideState":
        """Fit the official energy-weighted bank without any OOD samples."""
        train_features = np.asarray(train_features, dtype=np.float32)
        weight = np.asarray(weight, dtype=np.float32)
        bias = np.asarray(bias, dtype=np.float32)
        bank_parts: list[np.ndarray] = []
        weight_t = torch.as_tensor(weight, device=device)
        bias_t = torch.as_tensor(bias, device=device)
        with torch.inference_mode():
            for begin in range(0, len(train_features), batch_size):
                x = torch.as_tensor(train_features[begin : begin + batch_size], device=device)
                unit = F.normalize(x, dim=1)
                energy = torch.logsumexp(x @ weight_t.T + bias_t, dim=1)
                bank_parts.append((unit * energy[:, None]).cpu().numpy())
        scaled_bank = np.concatenate(bank_parts).astype(np.float32, copy=False)
        if not np.isfinite(scaled_bank).all():
            raise ValueError("NNGuide bank contains NaN or Inf")
        return cls(scaled_bank=scaled_bank, weight=weight, bias=bias, k=int(k))

    def confidence_batches(
        self,
        features: np.ndarray,
        *,
        device: str = "cuda",
        batch_size: int = 128,
    ) -> Iterator[np.ndarray]:
        """Yield higher-is-ID NNGuide confidence batches."""
        features = np.asarray(features)
        bank = torch.as_tensor(self.scaled_bank, dtype=torch.float32, device=device)
        weight = torch.as_tensor(self.weight, dtype=torch.float32, device=device)
        bias = torch.as_tensor(self.bias, dtype=torch.float32, device=device)
        with torch.inference_mode():
            for begin in range(0, len(features), batch_size):
                x = torch.as_tensor(
                    features[begin : begin + batch_size], dtype=torch.float32, device=device
                )
                unit = F.normalize(x, dim=1)
                energy = torch.logsumexp(x @ weight.T + bias, dim=1)
                guidance = (unit @ bank.T).topk(self.k, dim=1).values.mean(dim=1)
                yield (energy * guidance).cpu().numpy().astype(np.float32, copy=False)

    def confidence(
        self,
        features: np.ndarray,
        *,
        device: str = "cuda",
        batch_size: int = 128,
    ) -> np.ndarray:
        """Return higher-is-ID confidence; negate it to obtain an OOD score."""
        parts = list(self.confidence_batches(features, device=device, batch_size=batch_size))
        result = np.concatenate(parts) if parts else np.empty(0, dtype=np.float32)
        if not np.isfinite(result).all():
            raise ValueError("NNGuide confidence contains NaN or Inf")
        return result
