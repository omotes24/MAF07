from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np


ScaleMode = Literal["none", "global_std", "class_std", "class_mad", "class_iqr"]
Aggregation = Literal["euclidean", "squared", "huber"]
BankMode = Literal["pooled", "classwise"]
NeighborReduce = Literal["mean", "kth"]


@dataclass(frozen=True)
class FactorSpec:
    name: str
    normalize: bool = False
    scale: ScaleMode = "class_std"
    aggregation: Aggregation = "huber"
    bank_mode: BankMode = "classwise"
    neighbor_reduce: NeighborReduce = "mean"
    k: int = 150
    delta: float = 1.345
    min_scale: float = 1e-3
    equal_bank_n: int | None = None

    def __post_init__(self) -> None:
        if self.k < 1:
            raise ValueError("k must be positive")
        if self.delta <= 0:
            raise ValueError("delta must be positive")
        if self.min_scale <= 0:
            raise ValueError("min_scale must be positive")
        if self.equal_bank_n is not None and self.equal_bank_n < 1:
            raise ValueError("equal_bank_n must be positive")
        if self.bank_mode == "pooled" and self.scale.startswith("class_"):
            raise ValueError("class-specific scales require classwise banks")
        if self.aggregation == "huber" and self.neighbor_reduce != "mean":
            raise ValueError("Huber variants use mean neighbor aggregation")


def reviewer_factor_specs() -> list[FactorSpec]:
    """Return the pre-registered factor and one-factor sensitivity grid."""

    specs = [
        # Published KNN and raw KNN controls.
        FactorSpec(
            "knn_l2_pooled_k50_kth",
            normalize=True,
            scale="none",
            aggregation="euclidean",
            bank_mode="pooled",
            neighbor_reduce="kth",
            k=50,
        ),
        FactorSpec(
            "knn_raw_pooled_k150_kth",
            scale="none",
            aggregation="euclidean",
            bank_mode="pooled",
            neighbor_reduce="kth",
        ),
        FactorSpec(
            "knn_raw_pooled_k150_mean",
            scale="none",
            aggregation="euclidean",
            bank_mode="pooled",
        ),
        FactorSpec(
            "raw_pooled_huber_k150",
            scale="none",
            aggregation="huber",
            bank_mode="pooled",
        ),
        # Candidate-structure and residual-aggregation controls.
        FactorSpec("knn_raw_classwise_k150_mean", scale="none", aggregation="euclidean"),
        FactorSpec("raw_classwise_squared_k150", scale="none", aggregation="squared"),
        FactorSpec("raw_classwise_huber_k150", scale="none", aggregation="huber"),
        # Global versus class-conditional scaling.
        FactorSpec(
            "global_std_euclidean_k150",
            scale="global_std",
            aggregation="euclidean",
        ),
        FactorSpec("global_std_squared_k150", scale="global_std", aggregation="squared"),
        FactorSpec("global_std_huber_k150", scale="global_std", aggregation="huber"),
        FactorSpec("class_std_euclidean_k150", scale="class_std", aggregation="euclidean"),
        FactorSpec("class_std_squared_k150", scale="class_std", aggregation="squared"),
        FactorSpec("rsn_class_std_huber_k150", scale="class_std", aggregation="huber"),
        # Robust alternatives to the standard deviation.
        FactorSpec("class_mad_huber_k150", scale="class_mad", aggregation="huber"),
        FactorSpec("class_iqr_huber_k150", scale="class_iqr", aggregation="huber"),
        # Feature-normalization controls with all other factors fixed.
        FactorSpec(
            "l2_class_std_euclidean_k150",
            normalize=True,
            scale="class_std",
            aggregation="euclidean",
        ),
        FactorSpec(
            "l2_class_std_squared_k150",
            normalize=True,
            scale="class_std",
            aggregation="squared",
        ),
        FactorSpec(
            "l2_class_std_huber_k150",
            normalize=True,
            scale="class_std",
            aggregation="huber",
        ),
        # Equal class-bank size tests the order-statistic confound directly.
        FactorSpec(
            "rsn_equalbank3000",
            scale="class_std",
            aggregation="huber",
            equal_bank_n=3000,
        ),
    ]

    for k in (25, 50, 100, 300):
        specs.append(
            FactorSpec(
                f"rsn_k{k}",
                scale="class_std",
                aggregation="huber",
                k=k,
            )
        )
    for delta_text, delta in (("075", 0.75), ("100", 1.0), ("200", 2.0), ("300", 3.0)):
        specs.append(
            FactorSpec(
                f"rsn_delta{delta_text}",
                scale="class_std",
                aggregation="huber",
                delta=delta,
            )
        )
    for tau_text, tau in (("1e4", 1e-4), ("1e2", 1e-2), ("1e1", 1e-1)):
        specs.append(
            FactorSpec(
                f"rsn_tau{tau_text}",
                scale="class_std",
                aggregation="huber",
                min_scale=tau,
            )
        )

    names = [spec.name for spec in specs]
    if len(names) != len(set(names)):
        raise AssertionError("Reviewer factor specification names must be unique")
    return specs


class FactorialNeighborSuite:
    """Shared-search evaluator for the RSN factor decomposition.

    All variants in one preprocessing group reuse the same nearest-neighbor
    search. Returned scores are ID scores, so larger values mean more ID-like.
    """

    def __init__(
        self,
        train_features: np.ndarray,
        train_labels: np.ndarray,
        *,
        device: str | None = None,
        score_batch: int = 64,
        eps: float = 1e-12,
    ) -> None:
        import torch

        self.torch = torch
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.train = torch.as_tensor(
            np.asarray(train_features, dtype=np.float32), device=self.device
        )
        self.labels = np.asarray(train_labels, dtype=int)
        if len(self.train) != len(self.labels):
            raise ValueError("Feature and label lengths differ")
        if self.train.ndim != 2:
            raise ValueError("Expected a 2D feature matrix")
        self.score_batch = max(1, int(score_batch))
        self.eps = float(eps)

    def _normalize(self, values):
        return values / values.norm(dim=1, keepdim=True).clamp_min(self.eps)

    def _selected_train(self, normalize: bool, equal_bank_n: int | None):
        train = self._normalize(self.train) if normalize else self.train
        if equal_bank_n is None:
            return train, self.labels
        selected = []
        selected_labels = []
        for cls in sorted(np.unique(self.labels)):
            indices = np.flatnonzero(self.labels == int(cls))[: int(equal_bank_n)]
            index_tensor = self.torch.as_tensor(
                indices, dtype=self.torch.long, device=self.device
            )
            selected.append(train[index_tensor])
            selected_labels.extend([int(cls)] * len(indices))
        return self.torch.cat(selected, dim=0), np.asarray(selected_labels, dtype=int)

    def _scale(self, values, mode: ScaleMode, min_scale: float):
        torch = self.torch
        if mode == "none":
            scale = torch.ones(values.shape[1], dtype=values.dtype, device=values.device)
        elif mode.endswith("std"):
            scale = torch.std(values, dim=0, unbiased=True)
        elif mode == "class_mad":
            median = torch.median(values, dim=0).values
            scale = 1.4826 * torch.median(torch.abs(values - median), dim=0).values
        elif mode == "class_iqr":
            q1 = torch.quantile(values, 0.25, dim=0)
            q3 = torch.quantile(values, 0.75, dim=0)
            scale = (q3 - q1) / 1.349
        else:
            raise ValueError(f"Unknown scale mode: {mode}")
        return scale.clamp_min(float(min_scale)).contiguous()

    def _banks(
        self,
        *,
        normalize: bool,
        scale_mode: ScaleMode,
        bank_mode: BankMode,
        min_scale: float,
        equal_bank_n: int | None,
    ):
        train, labels = self._selected_train(normalize, equal_bank_n)
        if scale_mode == "global_std":
            global_scale = self._scale(train, scale_mode, min_scale)
        else:
            global_scale = None
        if bank_mode == "pooled":
            scale = global_scale
            if scale is None:
                scale = self._scale(train, scale_mode, min_scale)
            return [("pooled", (train / scale).contiguous(), scale)]

        banks = []
        for cls in sorted(np.unique(labels)):
            indices = self.torch.as_tensor(
                np.flatnonzero(labels == int(cls)),
                dtype=self.torch.long,
                device=self.device,
            )
            class_values = train[indices]
            scale = global_scale
            if scale is None:
                scale = self._scale(class_values, scale_mode, min_scale)
            banks.append((int(cls), (class_values / scale).contiguous(), scale))
        return banks

    def score_many(
        self,
        query_features: np.ndarray,
        specs: Sequence[FactorSpec],
    ) -> dict[str, np.ndarray]:
        torch = self.torch
        requested = list(specs)
        if not requested:
            return {}
        query_raw = torch.as_tensor(
            np.asarray(query_features, dtype=np.float32), device=self.device
        )
        if query_raw.ndim != 2:
            raise ValueError("Expected a 2D query feature matrix")

        grouped: dict[tuple[object, ...], list[FactorSpec]] = {}
        for spec in requested:
            key = (
                spec.normalize,
                spec.scale,
                spec.bank_mode,
                spec.min_scale,
                spec.equal_bank_n,
            )
            grouped.setdefault(key, []).append(spec)

        outputs: dict[str, np.ndarray] = {}
        with torch.no_grad():
            for key, group_specs in grouped.items():
                normalize, scale_mode, bank_mode, min_scale, equal_bank_n = key
                query = self._normalize(query_raw) if normalize else query_raw
                banks = self._banks(
                    normalize=bool(normalize),
                    scale_mode=scale_mode,
                    bank_mode=bank_mode,
                    min_scale=float(min_scale),
                    equal_bank_n=equal_bank_n,
                )
                best = {
                    spec.name: torch.full(
                        (len(query),), torch.inf, dtype=torch.float32, device=self.device
                    )
                    for spec in group_specs
                }

                for _, bank, scale in banks:
                    bank_norm = torch.sum(bank * bank, dim=1)
                    max_k = min(max(spec.k for spec in group_specs), len(bank))
                    for start in range(0, len(query), self.score_batch):
                        end = min(start + self.score_batch, len(query))
                        scaled_query = (query[start:end] / scale).contiguous()
                        query_norm = torch.sum(
                            scaled_query * scaled_query, dim=1, keepdim=True
                        )
                        distance2 = (
                            query_norm
                            + bank_norm[None, :]
                            - 2.0 * (scaled_query @ bank.T)
                        ).clamp_min(0.0)
                        top = torch.topk(distance2, k=max_k, dim=1, largest=False)

                        needs_huber = any(
                            spec.aggregation == "huber" for spec in group_specs
                        )
                        differences = None
                        if needs_huber:
                            differences = scaled_query[:, None, :] - bank[top.indices]

                        huber_cache = {}
                        for spec in group_specs:
                            k = min(spec.k, max_k)
                            if spec.aggregation == "euclidean":
                                neighbor_values = torch.sqrt(top.values[:, :k])
                            elif spec.aggregation == "squared":
                                neighbor_values = top.values[:, :k]
                            else:
                                cache_key = (float(spec.delta), k)
                                if cache_key not in huber_cache:
                                    diff = differences[:, :k]
                                    abs_diff = torch.abs(diff)
                                    delta = float(spec.delta)
                                    penalty = torch.where(
                                        abs_diff <= delta,
                                        diff * diff,
                                        2.0 * delta * abs_diff - delta * delta,
                                    )
                                    huber_cache[cache_key] = penalty.sum(dim=2)
                                neighbor_values = huber_cache[cache_key]

                            if spec.neighbor_reduce == "kth":
                                distance = neighbor_values[:, k - 1]
                            else:
                                distance = neighbor_values.mean(dim=1)
                            best[spec.name][start:end] = torch.minimum(
                                best[spec.name][start:end], distance
                            )

                for spec in group_specs:
                    score = -best[spec.name]
                    if not torch.isfinite(score).all():
                        raise RuntimeError(f"Non-finite scores for {spec.name}")
                    outputs[spec.name] = score.detach().cpu().numpy()
        return outputs


def nnguide_score(
    train_features: np.ndarray,
    train_logits: np.ndarray,
    query_features: np.ndarray,
    query_logits: np.ndarray,
    *,
    k: int = 10,
    batch_size: int = 128,
    device: str | None = None,
) -> np.ndarray:
    """Official NNGuide score using the experiment's shared ridge head.

    This follows the authors' released implementation: training features are
    multiplied by their energy, inner-product kNN guidance is averaged, and the
    result is multiplied by the query energy.
    """

    import torch

    torch_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    train_x = torch.as_tensor(
        np.asarray(train_features, dtype=np.float32), device=torch_device
    )
    train_z = torch.as_tensor(
        np.asarray(train_logits, dtype=np.float32), device=torch_device
    )
    query_x = torch.as_tensor(
        np.asarray(query_features, dtype=np.float32), device=torch_device
    )
    query_z = torch.as_tensor(
        np.asarray(query_logits, dtype=np.float32), device=torch_device
    )
    train_energy = torch.logsumexp(train_z, dim=1)
    weighted_bank = (train_x * train_energy[:, None]).contiguous()
    query_energy = torch.logsumexp(query_z, dim=1)
    neighbors = min(max(1, int(k)), len(weighted_bank))
    chunks = []
    with torch.no_grad():
        for start in range(0, len(query_x), max(1, int(batch_size))):
            end = min(start + max(1, int(batch_size)), len(query_x))
            similarity = query_x[start:end] @ weighted_bank.T
            guidance = torch.topk(
                similarity, k=neighbors, dim=1, largest=True
            ).values.mean(dim=1)
            chunks.append(guidance * query_energy[start:end])
    scores = torch.cat(chunks).detach().cpu().numpy()
    if not np.isfinite(scores).all():
        raise RuntimeError("NNGuide produced non-finite scores")
    return scores
