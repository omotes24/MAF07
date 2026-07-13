"""PULSE: four-component strict-inductive OOD evidence."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from methods.activation_shift import robust_location_scale


@dataclass(frozen=True)
class PulseState:
    shift_center: float
    shift_scale: float
    uniform_center: float
    uniform_scale: float
    localized_center: float
    localized_scale: float
    knn_center: float
    knn_scale: float

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            shift_center=self.shift_center,
            shift_scale=self.shift_scale,
            uniform_center=self.uniform_center,
            uniform_scale=self.uniform_scale,
            localized_center=self.localized_center,
            localized_scale=self.localized_scale,
            knn_center=self.knn_center,
            knn_scale=self.knn_scale,
        )

    @classmethod
    def load(cls, path: Path) -> "PulseState":
        with np.load(path, allow_pickle=False) as data:
            return cls(**{name: float(data[name]) for name in cls.__dataclass_fields__})

    def components(
        self,
        shift_ratio: np.ndarray,
        uniform_cosine: np.ndarray,
        localized_prototype_confidence: np.ndarray,
        compact_knn_confidence: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Return aligned higher-is-OOD components with fixed unit weights."""
        arrays = [
            np.asarray(value, dtype=np.float64)
            for value in (
                shift_ratio,
                uniform_cosine,
                localized_prototype_confidence,
                compact_knn_confidence,
            )
        ]
        if len({value.shape for value in arrays}) != 1 or arrays[0].ndim != 1:
            raise ValueError("PULSE inputs must be aligned vectors")
        shift, uniform, localized, knn = arrays
        result = {
            "shift": np.maximum(
                (np.log1p(shift) - self.shift_center) / self.shift_scale, 0.0
            ),
            "uniform": np.maximum(
                (uniform - self.uniform_center) / self.uniform_scale, 0.0
            ),
            "localized": (-localized - self.localized_center) / self.localized_scale,
            "knn": (-knn - self.knn_center) / self.knn_scale,
        }
        if any(not np.isfinite(value).all() for value in result.values()):
            raise ValueError("PULSE produced non-finite components")
        return result

    def score(
        self,
        shift_ratio: np.ndarray,
        uniform_cosine: np.ndarray,
        localized_prototype_confidence: np.ndarray,
        compact_knn_confidence: np.ndarray,
    ) -> np.ndarray:
        """Return the unweighted sum; larger values indicate OOD."""
        components = self.components(
            shift_ratio,
            uniform_cosine,
            localized_prototype_confidence,
            compact_knn_confidence,
        )
        return sum(components.values())


def fit_pulse_state(
    *,
    shift_center: float,
    shift_scale: float,
    uniform_cosine: np.ndarray,
    localized_prototype_confidence: np.ndarray,
    compact_knn_confidence: np.ndarray,
) -> PulseState:
    """Fit scalar calibration using only independent ID validation scores."""
    uniform_center, uniform_scale = robust_location_scale(uniform_cosine)
    localized_center, localized_scale = robust_location_scale(
        -np.asarray(localized_prototype_confidence, dtype=np.float64)
    )
    knn_center, knn_scale = robust_location_scale(
        -np.asarray(compact_knn_confidence, dtype=np.float64)
    )
    return PulseState(
        shift_center=float(shift_center),
        shift_scale=float(shift_scale),
        uniform_center=uniform_center,
        uniform_scale=uniform_scale,
        localized_center=localized_center,
        localized_scale=localized_scale,
        knn_center=knn_center,
        knn_scale=knn_scale,
    )
