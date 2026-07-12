from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class CWKNNMeanDetector:
    """Class-wise kNN mean-distance detector on raw features.

    For each class, the detector averages the Euclidean distances to the
    ``k`` nearest training features from that class. The minimum class-wise
    mean is the OOD distance. No feature normalization, scale estimation,
    classifier head, or calibration is used.
    """

    k: int = 150
    device: str | None = None
    score_batch: int = 256

    def __post_init__(self) -> None:
        if self.k < 1:
            raise ValueError("k must be positive")
        if self.score_batch < 1:
            raise ValueError("score_batch must be positive")
        self.bank_: dict[int, object] = {}
        self.bank_norm_: dict[int, object] = {}
        self.feature_dim_: int | None = None

    def _torch(self):
        import torch

        return torch

    def _device(self):
        torch = self._torch()
        if self.device is not None:
            return torch.device(self.device)
        configured = os.environ.get("MAF07_CWKNN_DEVICE")
        if configured:
            return torch.device(configured)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(
        self,
        train_features: np.ndarray,
        train_labels: np.ndarray,
    ) -> "CWKNNMeanDetector":
        torch = self._torch()
        features = np.asarray(train_features, dtype=np.float32)
        labels = np.asarray(train_labels, dtype=int)
        if features.ndim != 2 or features.shape[0] == 0:
            raise ValueError("train_features must be a non-empty 2D matrix")
        if labels.ndim != 1 or len(labels) != len(features):
            raise ValueError("train_labels must match train_features")
        if not np.isfinite(features).all():
            raise ValueError("train_features contain non-finite values")

        device = self._device()
        tensor = torch.as_tensor(features, dtype=torch.float32, device=device)
        self.bank_ = {}
        self.bank_norm_ = {}
        self.feature_dim_ = int(features.shape[1])
        with torch.no_grad():
            for class_id in sorted(np.unique(labels)):
                indices = torch.as_tensor(
                    np.flatnonzero(labels == int(class_id)),
                    dtype=torch.long,
                    device=device,
                )
                bank = tensor[indices].contiguous()
                self.bank_[int(class_id)] = bank
                self.bank_norm_[int(class_id)] = torch.sum(
                    bank * bank, dim=1
                ).contiguous()
        return self

    def _candidate_classes(
        self, candidate_classes: Sequence[int] | None
    ) -> list[int]:
        if not self.bank_:
            raise RuntimeError("CWKNNMeanDetector is not fitted")
        classes = (
            list(self.bank_)
            if candidate_classes is None
            else [int(value) for value in candidate_classes]
        )
        if not classes:
            raise ValueError("candidate_classes must not be empty")
        if len(classes) != len(set(classes)):
            raise ValueError("candidate_classes must be unique")
        missing = sorted(set(classes) - set(self.bank_))
        if missing:
            raise ValueError(f"Unknown candidate classes: {missing}")
        return classes

    def class_distances(
        self,
        features: np.ndarray,
        *,
        candidate_classes: Sequence[int] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return class IDs and their mean kNN distances for every query."""

        torch = self._torch()
        values = np.asarray(features, dtype=np.float32)
        if values.ndim != 2:
            raise ValueError("features must be a 2D matrix")
        if self.feature_dim_ is None:
            raise RuntimeError("CWKNNMeanDetector is not fitted")
        if values.shape[1] != self.feature_dim_:
            raise ValueError(
                f"Expected feature dimension {self.feature_dim_}, got {values.shape[1]}"
            )
        if not np.isfinite(values).all():
            raise ValueError("features contain non-finite values")

        classes = self._candidate_classes(candidate_classes)
        device = self._device()
        query = torch.as_tensor(values, dtype=torch.float32, device=device)
        distances = torch.empty(
            (len(query), len(classes)), dtype=torch.float32, device=device
        )
        with torch.no_grad():
            for column, class_id in enumerate(classes):
                bank = self.bank_[class_id]
                bank_norm = self.bank_norm_[class_id]
                neighbors = min(int(self.k), len(bank))
                for start in range(0, len(query), int(self.score_batch)):
                    end = min(start + int(self.score_batch), len(query))
                    batch = query[start:end]
                    distance2 = (
                        torch.sum(batch * batch, dim=1, keepdim=True)
                        + bank_norm[None, :]
                        - 2.0 * batch @ bank.T
                    ).clamp_min(0.0)
                    nearest = torch.topk(
                        distance2, k=neighbors, dim=1, largest=False
                    ).values
                    distances[start:end, column] = torch.sqrt(nearest).mean(dim=1)

        return (
            np.asarray(classes, dtype=int),
            distances.detach().cpu().numpy(),
        )

    def ood_scores(
        self,
        features: np.ndarray,
        *,
        candidate_classes: Sequence[int] | None = None,
    ) -> np.ndarray:
        """Return OOD scores, where larger values mean more OOD-like."""

        _, distances = self.class_distances(
            features, candidate_classes=candidate_classes
        )
        return distances.min(axis=1)

    def id_scores(
        self,
        features: np.ndarray,
        *,
        candidate_classes: Sequence[int] | None = None,
    ) -> np.ndarray:
        """Return ID scores, where larger values mean more ID-like."""

        return -self.ood_scores(features, candidate_classes=candidate_classes)

    def predict(
        self,
        features: np.ndarray,
        *,
        candidate_classes: Sequence[int] | None = None,
    ) -> np.ndarray:
        """Return the class whose class-wise mean kNN distance is smallest."""

        classes, distances = self.class_distances(
            features, candidate_classes=candidate_classes
        )
        return classes[np.argmin(distances, axis=1)]


def cwknn_mean_from_env() -> CWKNNMeanDetector:
    return CWKNNMeanDetector(
        k=int(os.environ.get("MAF07_CWKNN_K", "150")),
        device=os.environ.get("MAF07_CWKNN_DEVICE") or None,
        score_batch=int(os.environ.get("MAF07_CWKNN_SCORE_BATCH", "256")),
    )
