from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

EPS = 1e-12


@dataclass
class CATSDetector:
    anchors_per_class: int = 256
    local_k: int = 64
    tangent_dim: int = 16
    candidate_anchors: int = 16
    boundary_quantile: float = 0.95
    normal_quantile: float = 0.95
    softmin_temperature: float = 0.05
    normal_weight: float = 1.0
    normalize: bool = True
    device: str | None = None
    score_batch: int = 256
    eps: float = EPS

    def __post_init__(self) -> None:
        self.models_: dict[int, dict[str, object]] = {}
        self.cal_scores_: dict[int, object] = {}

    def _torch(self):
        import torch

        return torch

    def _device(self):
        torch = self._torch()
        if self.device is not None:
            return torch.device(self.device)
        env_device = os.environ.get("MAF07_CATS_DEVICE") or os.environ.get("MAF07_TORCH_DEVICE")
        if env_device:
            return torch.device(env_device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _l2_normalize(self, x):
        return x / (x.norm(dim=-1, keepdim=True) + float(self.eps))

    def fit(
        self,
        z_train: np.ndarray,
        y_train: np.ndarray,
        *,
        z_cal: np.ndarray | None = None,
        y_cal: np.ndarray | None = None,
    ) -> "CATSDetector":
        torch = self._torch()
        device = self._device()
        with torch.no_grad():
            train_x = torch.as_tensor(np.asarray(z_train, dtype=np.float32), device=device)
            train_y = np.asarray(y_train, dtype=int)
            if self.normalize:
                train_x = self._l2_normalize(train_x)
            self.models_ = {}
            for class_id in sorted(np.unique(train_y)):
                class_x = train_x[train_y == int(class_id)]
                self.models_[int(class_id)] = self._fit_class(class_x)

            self.cal_scores_ = {}
            if z_cal is not None and y_cal is not None:
                cal_x = torch.as_tensor(np.asarray(z_cal, dtype=np.float32), device=device)
                cal_y = np.asarray(y_cal, dtype=int)
                if self.normalize:
                    cal_x = self._l2_normalize(cal_x)
                for class_id in self.models_:
                    rows = np.where(cal_y == int(class_id))[0]
                    if len(rows) == 0:
                        self.cal_scores_[int(class_id)] = torch.zeros(1, dtype=torch.float32, device=device)
                        continue
                    values = self._class_energy(cal_x[rows], int(class_id))
                    self.cal_scores_[int(class_id)] = torch.sort(values).values
            else:
                for class_id in self.models_:
                    values = self._class_energy(train_x[train_y == int(class_id)], int(class_id))
                    self.cal_scores_[int(class_id)] = torch.sort(values).values
        return self

    def _fit_class(self, class_x):
        torch = self._torch()
        n_samples, n_features = class_x.shape
        n_anchors = min(int(self.anchors_per_class), n_samples)
        anchor_idx = self._farthest_point_indices(class_x, n_anchors)
        anchors = class_x[anchor_idx].contiguous()
        local_k = min(int(self.local_k), n_samples)
        tangent_dim = min(int(self.tangent_dim), local_k - 1, n_features)

        dists = torch.cdist(anchors, class_x)
        nn_idx = torch.topk(dists, local_k, dim=1, largest=False).indices
        neighbors = class_x[nn_idx]
        centered = neighbors - anchors[:, None, :]
        u, s, vh = torch.linalg.svd(centered, full_matrices=False)
        bases = vh[:, :tangent_dim, :].contiguous()
        coords = torch.einsum("mkd,mrd->mkr", centered, bases).contiguous()
        coord_norms = torch.linalg.norm(coords, dim=2).clamp_min(float(self.eps)).contiguous()
        residual_sq = torch.clamp(
            torch.sum(centered * centered, dim=2) - torch.sum(coords * coords, dim=2),
            min=0.0,
        )
        residual = torch.sqrt(residual_sq)
        q_perp = torch.quantile(residual, float(self.normal_quantile), dim=1).clamp_min(float(self.eps))
        return {
            "anchors": anchors,
            "bases": bases,
            "coords": coords,
            "coord_norms": coord_norms,
            "q_perp": q_perp.contiguous(),
        }

    def _farthest_point_indices(self, x, n_anchors: int):
        torch = self._torch()
        n_samples = x.shape[0]
        indices = torch.empty(n_anchors, dtype=torch.long, device=x.device)
        mean = x.mean(dim=0, keepdim=True)
        dist = torch.sum((x - mean) ** 2, dim=1)
        indices[0] = torch.argmax(dist)
        min_dist = torch.sum((x - x[indices[0]]) ** 2, dim=1)
        for i in range(1, n_anchors):
            indices[i] = torch.argmax(min_dist)
            new_dist = torch.sum((x - x[indices[i]]) ** 2, dim=1)
            min_dist = torch.minimum(min_dist, new_dist)
        return indices

    def _weighted_quantile(self, values, weights, q: float):
        torch = self._torch()
        sorted_values, order = torch.sort(values, dim=-1)
        sorted_weights = torch.gather(weights, dim=-1, index=order)
        cdf = torch.cumsum(sorted_weights, dim=-1)
        threshold = q * cdf[..., -1:].clamp_min(float(self.eps))
        idx = torch.searchsorted(cdf.contiguous(), threshold.contiguous()).squeeze(-1)
        idx = torch.clamp(idx, max=sorted_values.shape[-1] - 1)
        return torch.gather(sorted_values, dim=-1, index=idx.unsqueeze(-1)).squeeze(-1)

    def _class_energy(self, q, class_id: int):
        torch = self._torch()
        model = self.models_[int(class_id)]
        anchors = model["anchors"]
        bases = model["bases"]
        coords = model["coords"]
        coord_norms = model["coord_norms"]
        q_perp = model["q_perp"]
        top_m = min(int(self.candidate_anchors), anchors.shape[0])
        energies = torch.empty(q.shape[0], dtype=torch.float32, device=q.device)
        batch_size = max(1, int(self.score_batch))
        for start in range(0, q.shape[0], batch_size):
            end = min(start + batch_size, q.shape[0])
            qb = q[start:end]
            anchor_dist = torch.cdist(qb, anchors)
            idx = torch.topk(anchor_dist, top_m, dim=1, largest=False).indices
            a = anchors[idx]
            u_basis = bases[idx]
            neigh_coords = coords[idx]
            neigh_norms = coord_norms[idx]
            q_perp_sel = q_perp[idx]
            delta = qb[:, None, :] - a
            y = torch.einsum("bmd,bmrd->bmr", delta, u_basis)
            y_norm = torch.linalg.norm(y, dim=2).clamp_min(float(self.eps))
            direction = y / y_norm[..., None]
            proj_raw = torch.einsum("bmkr,bmr->bmk", neigh_coords, direction)
            proj = torch.clamp(proj_raw, min=0.0)
            cosine = proj_raw / neigh_norms.clamp_min(float(self.eps))
            weights = torch.exp(torch.clamp(cosine / float(os.environ.get("MAF07_CATS_TAU", "0.1")), -50.0, 50.0))
            radius = self._weighted_quantile(proj, weights, float(self.boundary_quantile)).clamp_min(float(self.eps))
            delta_sq = torch.sum(delta * delta, dim=2)
            normal = torch.sqrt(torch.clamp(delta_sq - y_norm * y_norm, min=0.0))
            local = (y_norm / radius) ** 2 + float(self.normal_weight) * (normal / q_perp_sel.clamp_min(float(self.eps))) ** 2
            temp = max(float(self.softmin_temperature), float(self.eps))
            energies[start:end] = -temp * torch.logsumexp(-local / temp, dim=1)
        return energies

    def score_samples(self, z: np.ndarray) -> np.ndarray:
        torch = self._torch()
        if not self.models_:
            raise RuntimeError("CATSDetector is not fitted")
        device = next(iter(self.models_.values()))["anchors"].device
        with torch.no_grad():
            q = torch.as_tensor(np.asarray(z, dtype=np.float32), device=device)
            if q.ndim == 1:
                q = q.reshape(1, -1)
            if self.normalize:
                q = self._l2_normalize(q)
            best_p = torch.zeros(q.shape[0], dtype=torch.float32, device=device)
            for class_id in self.models_:
                energy = self._class_energy(q, int(class_id))
                cal = self.cal_scores_[int(class_id)]
                rank = len(cal) - torch.searchsorted(cal, energy, right=False)
                pvals = (1.0 + rank.to(torch.float32)) / (1.0 + len(cal))
                best_p = torch.maximum(best_p, pvals)
            return (1.0 - best_p).detach().cpu().numpy()

    def id_scores(self, z: np.ndarray) -> np.ndarray:
        return -self.score_samples(z)


def cats_from_env() -> CATSDetector:
    return CATSDetector(
        anchors_per_class=int(os.environ.get("MAF07_CATS_ANCHORS", "256")),
        local_k=int(os.environ.get("MAF07_CATS_LOCAL_K", "64")),
        tangent_dim=int(os.environ.get("MAF07_CATS_TANGENT_DIM", "16")),
        candidate_anchors=int(os.environ.get("MAF07_CATS_CANDIDATE_ANCHORS", "16")),
        boundary_quantile=float(os.environ.get("MAF07_CATS_BOUNDARY_Q", "0.95")),
        normal_quantile=float(os.environ.get("MAF07_CATS_NORMAL_Q", "0.95")),
        softmin_temperature=float(os.environ.get("MAF07_CATS_SOFTMIN_T", "0.05")),
        normal_weight=float(os.environ.get("MAF07_CATS_NORMAL_WEIGHT", "1.0")),
        normalize=os.environ.get("MAF07_CATS_NORMALIZE", "1") != "0",
        device=os.environ.get("MAF07_CATS_DEVICE") or None,
        score_batch=int(os.environ.get("MAF07_CATS_SCORE_BATCH", "256")),
    )
