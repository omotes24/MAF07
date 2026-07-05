from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np


@dataclass
class LARScorer:
    k: int = 200
    gamma: float = 1e-2
    beta: float = 1.0
    normalize: bool = True
    device: str | None = None
    search_chunk: int = 2048
    score_batch: int = 128
    eps: float = 1e-8

    def __post_init__(self) -> None:
        self.bank = None
        self.bank_sq = None

    def _torch(self):
        import torch

        return torch

    def _device(self):
        torch = self._torch()
        if self.device is not None:
            return torch.device(self.device)
        env_device = os.environ.get("MAF07_LAR_DEVICE") or os.environ.get("MAF07_TORCH_DEVICE")
        if env_device:
            return torch.device(env_device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _l2_normalize(self, x):
        return x / (x.norm(dim=-1, keepdim=True) + float(self.eps))

    def fit(self, features: np.ndarray) -> "LARScorer":
        torch = self._torch()
        device = self._device()
        with torch.no_grad():
            x = torch.as_tensor(np.asarray(features, dtype=np.float32), device=device)
            if self.normalize:
                x = self._l2_normalize(x)
            self.bank = x
            self.bank_sq = (x * x).sum(dim=1)
        return self

    def _knn_idx(self, q, k: int):
        torch = self._torch()
        if self.bank is None or self.bank_sq is None:
            raise RuntimeError("LARScorer is not fitted")
        chunks = []
        for start in range(0, q.shape[0], int(self.search_chunk)):
            dist = -2.0 * q[start : start + int(self.search_chunk)] @ self.bank.T + self.bank_sq[None, :]
            chunks.append(torch.topk(dist, k, dim=1, largest=False).indices)
        return torch.cat(chunks, dim=0)

    def score_samples(self, features: np.ndarray) -> np.ndarray:
        torch = self._torch()
        if self.bank is None:
            raise RuntimeError("LARScorer is not fitted")
        device = self.bank.device
        with torch.no_grad():
            q = torch.as_tensor(np.asarray(features, dtype=np.float32), device=device)
            if self.normalize:
                q = self._l2_normalize(q)
            k = min(int(self.k), int(self.bank.shape[0]))
            idx = self._knn_idx(q, k)
            denom = max(1, k - 1)
            out = torch.empty(q.shape[0], device=device, dtype=torch.float32)
            for start in range(0, q.shape[0], int(self.score_batch)):
                end = min(start + int(self.score_batch), q.shape[0])
                z = q[start:end].double()
                nb = self.bank[idx[start:end]].double()
                mu = nb.mean(dim=1, keepdim=True)
                centered = nb - mu
                zc = z - mu.squeeze(1)
                gram = centered @ centered.transpose(1, 2)
                gram = 0.5 * (gram + gram.transpose(1, 2))
                eigvals, eigvecs = torch.linalg.eigh(gram)
                eigvals = eigvals.clamp_min(0.0)
                proj = (centered @ zc.unsqueeze(-1)).squeeze(-1)
                t = (eigvecs.transpose(1, 2) @ proj.unsqueeze(-1)).squeeze(-1)
                pos = eigvals > 1e-6
                safe = eigvals.clamp_min(1e-6)
                a2 = torch.where(pos, t * t / safe, torch.zeros_like(t))
                lam = eigvals / float(denom)
                in_term = torch.where(
                    pos,
                    a2 / (lam + float(self.gamma)),
                    torch.zeros_like(a2),
                ).sum(dim=1)
                residual = ((zc * zc).sum(dim=1) - a2.sum(dim=1)).clamp_min(0.0)
                out[start:end] = (in_term + residual / float(self.beta)).float()
            return out.detach().cpu().numpy()

    def id_scores(self, features: np.ndarray) -> np.ndarray:
        return -self.score_samples(features)


def lar_from_env() -> LARScorer:
    return LARScorer(
        k=int(os.environ.get("MAF07_LAR_K", "200")),
        gamma=float(os.environ.get("MAF07_LAR_GAMMA", "1e-2")),
        beta=float(os.environ.get("MAF07_LAR_BETA", "1.0")),
        normalize=os.environ.get("MAF07_LAR_NORMALIZE", "1") != "0",
        search_chunk=int(os.environ.get("MAF07_LAR_SEARCH_CHUNK", "2048")),
        score_batch=int(os.environ.get("MAF07_LAR_SCORE_BATCH", "128")),
    )
