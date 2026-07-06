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
class CSRDetector:
    r: int = 128
    gamma_tail: float = 1e-2
    topq: int = 3
    normalize: bool = False
    device: str | None = None
    score_batch: int = 256
    eps: float = EPS

    def __post_init__(self) -> None:
        self.mean_ = None
        self.Ur_ = None
        self.Ut_ = None
        self.tail_w_ = None
        self.class_stats_: dict[int, dict[str, object]] = {}
        self.cal_scores_: dict[int, object] = {}

    def _torch(self):
        import torch

        return torch

    def _device(self):
        torch = self._torch()
        if self.device is not None:
            return torch.device(self.device)
        env_device = os.environ.get("MAF07_CSR_DEVICE") or os.environ.get("MAF07_TORCH_DEVICE")
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
    ) -> "CSRDetector":
        torch = self._torch()
        device = self._device()
        with torch.no_grad():
            train_x = torch.as_tensor(np.asarray(z_train, dtype=np.float64), device=device)
            train_y = np.asarray(y_train, dtype=int)
            if self.normalize:
                train_x = self._l2_normalize(train_x)
            n_samples, n_features = train_x.shape
            rank = max(1, min(int(self.r), n_features - 1))

            self.mean_ = train_x.mean(dim=0)
            centered = train_x - self.mean_
            cov = (centered.T @ centered) / max(1, n_samples - 1)
            eigval, eigvec = torch.linalg.eigh(cov)
            eigval = torch.flip(eigval, dims=(0,))
            eigvec = torch.flip(eigvec, dims=(1,))
            self.Ur_ = eigvec[:, :rank].contiguous()
            self.Ut_ = eigvec[:, rank:].contiguous()
            self.tail_w_ = (1.0 / (eigval[rank:] + float(self.gamma_tail))).contiguous()

            self.class_stats_ = {}
            for cls in sorted(np.unique(train_y)):
                class_id = int(cls)
                class_x = train_x[train_y == class_id]
                zr = (class_x - self.mean_) @ self.Ur_
                mu = zr.mean(dim=0)
                zc = zr - mu
                if zr.shape[0] > 1:
                    cov_c = (zc.T @ zc) / (zr.shape[0] - 1)
                else:
                    cov_c = torch.eye(rank, dtype=torch.float64, device=device)
                alpha = float(rank) / float(rank + max(1, zr.shape[0]))
                target = torch.eye(rank, dtype=torch.float64, device=device) * (cov_c.trace() / rank)
                shrunk = (1.0 - alpha) * cov_c + alpha * target
                precision = torch.linalg.inv(shrunk)
                self.class_stats_[class_id] = {"mu": mu.contiguous(), "precision": precision.contiguous()}

            self.cal_scores_ = {}
            if z_cal is not None and y_cal is not None:
                cal_x = torch.as_tensor(np.asarray(z_cal, dtype=np.float64), device=device)
                cal_y = np.asarray(y_cal, dtype=int)
                if self.normalize:
                    cal_x = self._l2_normalize(cal_x)
                for class_id in self.class_stats_:
                    rows = np.where(cal_y == class_id)[0]
                    if len(rows) == 0:
                        self.cal_scores_[class_id] = torch.zeros(1, dtype=torch.float64, device=device)
                        continue
                    values = self._distance_class(cal_x[rows], class_id)
                    self.cal_scores_[class_id] = torch.sort(values).values
            else:
                for class_id in self.class_stats_:
                    values = self._distance_class(train_x[train_y == class_id], class_id)
                    self.cal_scores_[class_id] = torch.sort(values).values
        return self

    def _distance_class(self, q, class_id: int):
        torch = self._torch()
        if self.mean_ is None or self.Ur_ is None or self.Ut_ is None or self.tail_w_ is None:
            raise RuntimeError("CSRDetector is not fitted")
        centered = q - self.mean_
        zr = centered @ self.Ur_
        stats = self.class_stats_[int(class_id)]
        zc = zr - stats["mu"]
        precision = stats["precision"]
        maha = torch.einsum("nr,rs,ns->n", zc, precision, zc)
        if self.Ut_.shape[1] > 0:
            tail = ((centered @ self.Ut_) ** 2 * self.tail_w_).sum(dim=1)
        else:
            tail = torch.zeros(q.shape[0], dtype=torch.float64, device=q.device)
        return maha + tail

    def score_samples(self, z: np.ndarray, logits: np.ndarray | None = None) -> np.ndarray:
        torch = self._torch()
        if not self.class_stats_:
            raise RuntimeError("CSRDetector is not fitted")
        device = self.mean_.device
        with torch.no_grad():
            q = torch.as_tensor(np.asarray(z, dtype=np.float64), device=device)
            if q.ndim == 1:
                q = q.reshape(1, -1)
            if self.normalize:
                q = self._l2_normalize(q)
            classes = list(self.class_stats_.keys())
            if logits is None:
                top = np.tile(np.asarray(classes, dtype=int), (q.shape[0], 1))
            else:
                class_logits = _class_logits(logits)
                topq = min(max(1, int(self.topq)), class_logits.shape[1])
                top = np.argsort(class_logits, axis=1)[:, ::-1][:, :topq]
            scores = torch.empty(q.shape[0], dtype=torch.float64, device=device)
            batch_size = max(1, int(self.score_batch))
            for start in range(0, q.shape[0], batch_size):
                end = min(start + batch_size, q.shape[0])
                qb = q[start:end]
                topb = top[start:end]
                best_p = torch.zeros(end - start, dtype=torch.float64, device=device)
                for class_id in classes:
                    rows_np = np.where(np.any(topb == int(class_id), axis=1))[0]
                    if len(rows_np) == 0:
                        continue
                    rows = torch.as_tensor(rows_np, dtype=torch.long, device=device)
                    dist = self._distance_class(qb[rows], class_id)
                    cal = self.cal_scores_[int(class_id)]
                    rank = len(cal) - torch.searchsorted(cal, dist, right=False)
                    pvals = (1.0 + rank.to(torch.float64)) / (1.0 + len(cal))
                    best_p[rows] = torch.maximum(best_p[rows], pvals)
                scores[start:end] = -torch.log(torch.clamp(best_p, min=float(self.eps)))
            return scores.detach().cpu().numpy()

    def id_scores(self, z: np.ndarray, logits: np.ndarray | None = None) -> np.ndarray:
        return -self.score_samples(z, logits)


def csr_from_env() -> CSRDetector:
    return CSRDetector(
        r=int(os.environ.get("MAF07_CSR_R", "128")),
        gamma_tail=float(os.environ.get("MAF07_CSR_GAMMA_TAIL", "1e-2")),
        topq=int(os.environ.get("MAF07_CSR_TOPQ", os.environ.get("MAF07_CARD_TOPQ", "3"))),
        normalize=os.environ.get("MAF07_CSR_NORMALIZE", "0") == "1",
        device=os.environ.get("MAF07_CSR_DEVICE") or None,
        score_batch=int(os.environ.get("MAF07_CSR_SCORE_BATCH", "256")),
    )
