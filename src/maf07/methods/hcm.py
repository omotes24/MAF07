from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

EPS = 1e-12


def _class_logits(logits: np.ndarray) -> np.ndarray:
    arr = np.asarray(logits, dtype=np.float64)
    if arr.ndim == 1:
        arr = np.stack([-arr, arr], axis=1)
    return arr


@dataclass
class HCMDetector:
    delta: float = 1.345
    topq: int = 3
    normalize: bool = False
    device: str | None = None
    score_batch: int = 512
    eps: float = EPS

    def __post_init__(self) -> None:
        self.class_stats_: dict[int, dict[str, object]] = {}
        self.cal_x_ = None
        self.cal_y_: np.ndarray | None = None
        self.cal_scores_: dict[tuple[int, str], object] = {}

    def _torch(self):
        import torch

        return torch

    def _device(self):
        torch = self._torch()
        if self.device is not None:
            return torch.device(self.device)
        env_device = os.environ.get("MAF07_HCM_DEVICE") or os.environ.get("MAF07_TORCH_DEVICE")
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
    ) -> "HCMDetector":
        torch = self._torch()
        device = self._device()
        with torch.no_grad():
            train_x = torch.as_tensor(np.asarray(z_train, dtype=np.float64), device=device)
            train_y = np.asarray(y_train, dtype=int)
            if self.normalize:
                train_x = self._l2_normalize(train_x)

            self.class_stats_ = {}
            for cls in sorted(np.unique(train_y)):
                class_id = int(cls)
                class_x = train_x[train_y == class_id]
                self.class_stats_[class_id] = self._fit_class(class_x)

            if z_cal is not None and y_cal is not None:
                self.cal_x_ = torch.as_tensor(np.asarray(z_cal, dtype=np.float64), device=device)
                self.cal_y_ = np.asarray(y_cal, dtype=int)
                if self.normalize:
                    self.cal_x_ = self._l2_normalize(self.cal_x_)
            else:
                self.cal_x_ = train_x
                self.cal_y_ = train_y
            self.cal_scores_ = {}
        return self

    def _fit_class(self, class_x):
        torch = self._torch()
        n_samples, n_features = class_x.shape
        mu = class_x.mean(dim=0)
        centered = class_x - mu
        if n_samples > 1:
            cov = (centered.T @ centered) / (n_samples - 1)
        else:
            cov = torch.eye(n_features, dtype=torch.float64, device=class_x.device)
        alpha = float(n_features) / float(n_features + max(1, n_samples))
        target = torch.eye(n_features, dtype=torch.float64, device=class_x.device) * (cov.trace() / n_features)
        shrunk = (1.0 - alpha) * cov + alpha * target
        eigval, eigvec = torch.linalg.eigh(shrunk)
        return {
            "mu": mu.contiguous(),
            "U": eigvec.contiguous(),
            "lam_sqrt": eigval.clamp_min(1e-8).sqrt().contiguous(),
        }

    def _distance_class(self, q, class_id: int, *, delta: float | None = None):
        torch = self._torch()
        stats = self.class_stats_[int(class_id)]
        d = float(self.delta if delta is None else delta)
        t = ((q - stats["mu"]) @ stats["U"]) / stats["lam_sqrt"]
        if d >= 1e8:
            return torch.sum(t * t, dim=1)
        a = t.abs()
        psi = torch.where(a <= d, t * t, 2.0 * d * a - d * d)
        return torch.sum(psi, dim=1)

    def _calibration_scores(self, class_id: int, *, delta: float):
        torch = self._torch()
        key = (int(class_id), f"{float(delta):.12g}")
        if key in self.cal_scores_:
            return self.cal_scores_[key]
        if self.cal_x_ is None or self.cal_y_ is None:
            raise RuntimeError("HCMDetector is not fitted with calibration data")
        rows_np = np.where(self.cal_y_ == int(class_id))[0]
        if len(rows_np) == 0:
            scores = torch.zeros(1, dtype=torch.float64, device=self.cal_x_.device)
        else:
            rows = torch.as_tensor(rows_np, dtype=torch.long, device=self.cal_x_.device)
            values = []
            batch_size = max(1, int(self.score_batch))
            for start in range(0, len(rows_np), batch_size):
                end = min(start + batch_size, len(rows_np))
                values.append(self._distance_class(self.cal_x_[rows[start:end]], class_id, delta=delta))
            scores = torch.sort(torch.cat(values) if values else torch.zeros(1, dtype=torch.float64, device=self.cal_x_.device)).values
        self.cal_scores_[key] = scores
        return scores

    def _candidate_classes(self, logits: np.ndarray | None, n: int) -> np.ndarray:
        classes = np.asarray(list(self.class_stats_.keys()), dtype=int)
        if logits is None:
            return np.tile(classes, (n, 1))
        class_logits = _class_logits(logits)
        topq = min(max(1, int(self.topq)), class_logits.shape[1])
        return np.argsort(class_logits, axis=1)[:, ::-1][:, :topq]

    def id_scores(
        self,
        z: np.ndarray,
        logits: np.ndarray | None = None,
        *,
        delta: float | None = None,
        calibrated: bool = True,
    ) -> np.ndarray:
        torch = self._torch()
        if not self.class_stats_:
            raise RuntimeError("HCMDetector is not fitted")
        device = next(iter(self.class_stats_.values()))["mu"].device
        d = float(self.delta if delta is None else delta)
        with torch.no_grad():
            q = torch.as_tensor(np.asarray(z, dtype=np.float64), device=device)
            if q.ndim == 1:
                q = q.reshape(1, -1)
            if self.normalize:
                q = self._l2_normalize(q)
            top = self._candidate_classes(logits, q.shape[0])
            scores = torch.empty(q.shape[0], dtype=torch.float64, device=device)
            batch_size = max(1, int(self.score_batch))
            for start in range(0, q.shape[0], batch_size):
                end = min(start + batch_size, q.shape[0])
                qb = q[start:end]
                topb = top[start:end]
                if calibrated:
                    best_p = torch.zeros(end - start, dtype=torch.float64, device=device)
                    for class_id in self.class_stats_:
                        rows_np = np.where(np.any(topb == int(class_id), axis=1))[0]
                        if len(rows_np) == 0:
                            continue
                        rows = torch.as_tensor(rows_np, dtype=torch.long, device=device)
                        dist = self._distance_class(qb[rows], class_id, delta=d)
                        cal = self._calibration_scores(class_id, delta=d)
                        rank = len(cal) - torch.searchsorted(cal, dist, right=False)
                        pvals = (1.0 + rank.to(torch.float64)) / (1.0 + len(cal))
                        best_p[rows] = torch.maximum(best_p[rows], pvals)
                    scores[start:end] = torch.log(torch.clamp(best_p, min=float(self.eps)))
                else:
                    best_dist = torch.full((end - start,), torch.inf, dtype=torch.float64, device=device)
                    for class_id in self.class_stats_:
                        rows_np = np.where(np.any(topb == int(class_id), axis=1))[0]
                        if len(rows_np) == 0:
                            continue
                        rows = torch.as_tensor(rows_np, dtype=torch.long, device=device)
                        dist = self._distance_class(qb[rows], class_id, delta=d)
                        best_dist[rows] = torch.minimum(best_dist[rows], dist)
                    scores[start:end] = -best_dist
            return scores.detach().cpu().numpy()


def hcm_from_env() -> HCMDetector:
    return HCMDetector(
        delta=float(os.environ.get("MAF07_HCM_DELTA", "1.345")),
        topq=int(os.environ.get("MAF07_HCM_TOPQ", os.environ.get("MAF07_CARD_TOPQ", "3"))),
        normalize=os.environ.get("MAF07_HCM_NORMALIZE", "0") == "1",
        device=os.environ.get("MAF07_HCM_DEVICE") or None,
        score_batch=int(os.environ.get("MAF07_HCM_SCORE_BATCH", "512")),
    )
