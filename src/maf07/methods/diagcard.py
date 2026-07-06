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
class DiagCARDDetector:
    k: int = 150
    delta: float = 1.345
    topq: int = 3
    normalize: bool = False
    min_std: float = 1e-3
    device: str | None = None
    score_batch: int = 256
    eps: float = EPS

    def __post_init__(self) -> None:
        self.bank_: dict[int, object] = {}
        self.bank_norm_: dict[int, object] = {}
        self.inv_std_: dict[int, object] = {}
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
        env_device = os.environ.get("MAF07_DIAGCARD_DEVICE") or os.environ.get("MAF07_TORCH_DEVICE")
        if env_device:
            return torch.device(env_device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _l2_normalize(self, x):
        return x / x.norm(dim=-1, keepdim=True).clamp_min(float(self.eps))

    def fit(
        self,
        z_train: np.ndarray,
        y_train: np.ndarray,
        *,
        z_cal: np.ndarray | None = None,
        y_cal: np.ndarray | None = None,
    ) -> "DiagCARDDetector":
        torch = self._torch()
        device = self._device()
        with torch.no_grad():
            train_x = torch.as_tensor(np.asarray(z_train, dtype=np.float32), device=device)
            train_y = np.asarray(y_train, dtype=int)
            if self.normalize:
                train_x = self._l2_normalize(train_x)
            self.bank_ = {}
            self.bank_norm_ = {}
            self.inv_std_ = {}
            self.cal_scores_ = {}
            for cls in sorted(np.unique(train_y)):
                class_id = int(cls)
                class_x = train_x[train_y == class_id]
                std = torch.std(class_x, dim=0, unbiased=True).clamp_min(float(self.min_std))
                inv_std = (1.0 / std).contiguous()
                bank = (class_x * inv_std).contiguous()
                self.inv_std_[class_id] = inv_std
                self.bank_[class_id] = bank
                self.bank_norm_[class_id] = torch.sum(bank * bank, dim=1).contiguous()
            if z_cal is not None and y_cal is not None:
                cal_x = torch.as_tensor(np.asarray(z_cal, dtype=np.float32), device=device)
                if self.normalize:
                    cal_x = self._l2_normalize(cal_x)
                self.cal_x_ = cal_x
                self.cal_y_ = np.asarray(y_cal, dtype=int)
            else:
                self.cal_x_ = train_x
                self.cal_y_ = train_y
        return self

    def _distance_class(self, q, class_id: int, *, delta: float | None = None):
        torch = self._torch()
        bank = self.bank_[int(class_id)]
        bank_norm = self.bank_norm_[int(class_id)]
        inv_std = self.inv_std_[int(class_id)]
        k = min(max(1, int(self.k)), bank.shape[0])
        out = torch.empty(q.shape[0], dtype=torch.float32, device=q.device)
        batch_size = max(1, int(self.score_batch))
        use_huber = delta is not None
        for start in range(0, q.shape[0], batch_size):
            end = min(start + batch_size, q.shape[0])
            qs = (q[start:end] * inv_std).contiguous()
            qs_norm = torch.sum(qs * qs, dim=1, keepdim=True)
            d2 = (qs_norm + bank_norm[None, :] - 2.0 * (qs @ bank.T)).clamp_min(0.0)
            top = torch.topk(d2, k=k, dim=1, largest=False)
            if not use_huber:
                out[start:end] = torch.sqrt(top.values).mean(dim=1)
                continue
            nb = bank[top.indices]
            diff = qs[:, None, :] - nb
            d = float(delta)
            a = diff.abs()
            psi = torch.where(a <= d, diff * diff, 2.0 * d * a - d * d)
            out[start:end] = psi.sum(dim=2).mean(dim=1)
        return out

    def _calibration_scores(self, class_id: int, *, delta: float | None):
        torch = self._torch()
        key = (int(class_id), "none" if delta is None else f"{float(delta):.12g}")
        if key in self.cal_scores_:
            return self.cal_scores_[key]
        if self.cal_x_ is None or self.cal_y_ is None:
            raise RuntimeError("DiagCARDDetector is not fitted with calibration data")
        rows_np = np.where(self.cal_y_ == int(class_id))[0]
        if len(rows_np) == 0:
            scores = torch.zeros(1, dtype=torch.float32, device=self.cal_x_.device)
        else:
            rows = torch.as_tensor(rows_np, dtype=torch.long, device=self.cal_x_.device)
            chunks = []
            for start in range(0, len(rows_np), max(1, int(self.score_batch))):
                end = min(start + max(1, int(self.score_batch)), len(rows_np))
                chunks.append(self._distance_class(self.cal_x_[rows[start:end]], class_id, delta=delta))
            scores = torch.sort(torch.cat(chunks)).values
        self.cal_scores_[key] = scores
        return scores

    def _candidate_classes(self, logits: np.ndarray | None, n: int) -> np.ndarray:
        classes = np.asarray(list(self.bank_.keys()), dtype=int)
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
        if not self.bank_:
            raise RuntimeError("DiagCARDDetector is not fitted")
        device = next(iter(self.bank_.values())).device
        with torch.no_grad():
            q = torch.as_tensor(np.asarray(z, dtype=np.float32), device=device)
            if q.ndim == 1:
                q = q.reshape(1, -1)
            if self.normalize:
                q = self._l2_normalize(q)
            top = self._candidate_classes(logits, q.shape[0])
            scores = torch.empty(q.shape[0], dtype=torch.float32, device=device)
            batch_size = max(1, int(self.score_batch))
            for start in range(0, q.shape[0], batch_size):
                end = min(start + batch_size, q.shape[0])
                qb = q[start:end]
                topb = top[start:end]
                if calibrated:
                    best_p = torch.zeros(end - start, dtype=torch.float32, device=device)
                    for class_id in self.bank_:
                        rows_np = np.where(np.any(topb == int(class_id), axis=1))[0]
                        if len(rows_np) == 0:
                            continue
                        rows = torch.as_tensor(rows_np, dtype=torch.long, device=device)
                        dist = self._distance_class(qb[rows], class_id, delta=delta)
                        cal = self._calibration_scores(class_id, delta=delta)
                        rank = len(cal) - torch.searchsorted(cal, dist, right=False)
                        pvals = (1.0 + rank.to(torch.float32)) / (1.0 + len(cal))
                        best_p[rows] = torch.maximum(best_p[rows], pvals)
                    scores[start:end] = torch.log(best_p.clamp_min(float(self.eps)))
                else:
                    best_dist = torch.full((end - start,), torch.inf, dtype=torch.float32, device=device)
                    for class_id in self.bank_:
                        rows_np = np.where(np.any(topb == int(class_id), axis=1))[0]
                        if len(rows_np) == 0:
                            continue
                        rows = torch.as_tensor(rows_np, dtype=torch.long, device=device)
                        dist = self._distance_class(qb[rows], class_id, delta=delta)
                        best_dist[rows] = torch.minimum(best_dist[rows], dist)
                    scores[start:end] = -best_dist
            return scores.detach().cpu().numpy()


def diagcard_from_env() -> DiagCARDDetector:
    return DiagCARDDetector(
        k=int(os.environ.get("MAF07_DIAGCARD_K", "150")),
        delta=float(os.environ.get("MAF07_DIAGCARD_DELTA", "1.345")),
        topq=int(os.environ.get("MAF07_DIAGCARD_TOPQ", os.environ.get("MAF07_CARD_TOPQ", "3"))),
        normalize=os.environ.get("MAF07_DIAGCARD_NORMALIZE", "0") == "1",
        min_std=float(os.environ.get("MAF07_DIAGCARD_MIN_STD", "1e-3")),
        device=os.environ.get("MAF07_DIAGCARD_DEVICE") or None,
        score_batch=int(os.environ.get("MAF07_DIAGCARD_SCORE_BATCH", "256")),
    )
