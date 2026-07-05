from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np


@dataclass
class PSMScorer:
    r: int = 48
    k: int = 200
    gamma_local: float = 1e-2
    gamma_tail: float = 1e-2
    normalize: bool = False
    device: str | None = None
    score_batch: int = 256
    eps: float = 1e-8

    def __post_init__(self) -> None:
        self.mean = None
        self.Ur = None
        self.Ut = None
        self.tail_w = None
        self.Xr = None
        self.Xr_sq = None

    def _torch(self):
        import torch

        return torch

    def _device(self):
        torch = self._torch()
        if self.device is not None:
            return torch.device(self.device)
        env_device = os.environ.get("MAF07_PSM_DEVICE") or os.environ.get("MAF07_TORCH_DEVICE")
        if env_device:
            return torch.device(env_device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _l2_normalize(self, x):
        return x / (x.norm(dim=-1, keepdim=True) + float(self.eps))

    def fit(self, features: np.ndarray) -> "PSMScorer":
        torch = self._torch()
        device = self._device()
        with torch.no_grad():
            x = torch.as_tensor(np.asarray(features, dtype=np.float64), device=device)
            if self.normalize:
                x = self._l2_normalize(x)
            n_samples, n_features = x.shape
            rank = max(1, min(int(self.r), n_features - 1))
            self.mean = x.mean(dim=0)
            centered = x - self.mean
            cov = (centered.T @ centered) / max(1, n_samples - 1)
            eigval, eigvec = torch.linalg.eigh(cov)
            eigval = torch.flip(eigval, dims=(0,))
            eigvec = torch.flip(eigvec, dims=(1,))
            self.Ur = eigvec[:, :rank].contiguous()
            self.Ut = eigvec[:, rank:].contiguous()
            self.tail_w = (1.0 / (eigval[rank:] + float(self.gamma_tail))).contiguous()
            xr = (centered @ self.Ur).float()
            self.Xr = xr
            self.Xr_sq = (xr * xr).sum(dim=1)
        return self

    def score_samples(self, features: np.ndarray) -> np.ndarray:
        torch = self._torch()
        if self.mean is None or self.Ur is None or self.Xr is None or self.Xr_sq is None:
            raise RuntimeError("PSMScorer is not fitted")
        device = self.mean.device
        with torch.no_grad():
            q = torch.as_tensor(np.asarray(features, dtype=np.float64), device=device)
            if q.shape[0] == 0:
                return np.empty((0,), dtype=np.float64)
            if self.normalize:
                q = self._l2_normalize(q)
            centered = q - self.mean
            k = min(int(self.k), int(self.Xr.shape[0]))
            batch_size = max(1, int(self.score_batch))
            scores = torch.empty(q.shape[0], dtype=torch.float64, device=device)
            eye = torch.eye(self.Ur.shape[1], dtype=torch.float64, device=device) * float(self.gamma_local)
            for start in range(0, q.shape[0], batch_size):
                end = min(start + batch_size, q.shape[0])
                centered_batch = centered[start:end]
                if self.Ut is not None and self.Ut.shape[1] > 0:
                    tail = ((centered_batch @ self.Ut) ** 2 * self.tail_w).sum(dim=1)
                else:
                    tail = torch.zeros(end - start, dtype=torch.float64, device=device)

                zr = (centered_batch @ self.Ur).float()
                dist = -2.0 * zr @ self.Xr.T + self.Xr_sq[None, :]
                idx = torch.topk(dist, k, dim=1, largest=False).indices
                nb = self.Xr[idx].double()
                mu = nb.mean(dim=1, keepdim=True)
                nb_centered = nb - mu
                cov = torch.einsum("mkr,mks->mrs", nb_centered, nb_centered) / max(1, k - 1)
                cov = cov + eye
                zc = zr.double() - mu.squeeze(1)
                solved = torch.linalg.solve(cov, zc.unsqueeze(-1)).squeeze(-1)
                local = torch.sum(zc * solved, dim=1)
                scores[start:end] = local + tail
            return scores.detach().cpu().numpy()

    def id_scores(self, features: np.ndarray) -> np.ndarray:
        return -self.score_samples(features)


def psm_from_env() -> PSMScorer:
    return PSMScorer(
        r=int(os.environ.get("MAF07_PSM_R", "48")),
        k=int(os.environ.get("MAF07_PSM_K", "200")),
        gamma_local=float(os.environ.get("MAF07_PSM_GAMMA_LOCAL", "1e-2")),
        gamma_tail=float(os.environ.get("MAF07_PSM_GAMMA_TAIL", "1e-2")),
        normalize=os.environ.get("MAF07_PSM_NORMALIZE", "0") == "1",
        score_batch=int(os.environ.get("MAF07_PSM_SCORE_BATCH", "256")),
    )
