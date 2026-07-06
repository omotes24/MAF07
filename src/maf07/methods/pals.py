from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

EPS = 1e-12


def _robust_stats_torch(x, eps: float):
    torch = __import__("torch")
    vals = x.reshape(-1)
    median = torch.median(vals)
    q = torch.quantile(vals, torch.tensor([0.25, 0.75], dtype=vals.dtype, device=vals.device))
    iqr = (q[1] - q[0]).clamp_min(float(eps))
    return median, iqr


@dataclass
class PALSDetector:
    num_prototypes: int = 384
    patches_per_image_for_codebook: int = 16
    kmeans_iter: int = 30
    kappa: float = 20.0
    alpha: float = 0.1
    epsilon: float = 1e-5
    lowrank_dim: int = 64
    max_codebook_images: int = 4096
    device: str | None = None
    image_batch_size: int = 48
    kmeans_batch_size: int = 8192
    eps: float = EPS
    models_: dict[str, dict[str, object]] = field(default_factory=dict)

    def _torch(self):
        import torch

        return torch

    def _device(self):
        torch = self._torch()
        if self.device is not None:
            return torch.device(self.device)
        env_device = os.environ.get("MAF07_PALS_DEVICE") or os.environ.get("MAF07_TORCH_DEVICE")
        if env_device:
            return torch.device(env_device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load_patch_batch(self, patches: np.ndarray, indices: np.ndarray, *, device=None):
        torch = self._torch()
        dev = device or self._device()
        arr = np.asarray(patches[np.asarray(indices, dtype=int)], dtype=np.float32)
        tensor = torch.as_tensor(arr, dtype=torch.float32, device=dev)
        return tensor / tensor.norm(dim=-1, keepdim=True).clamp_min(float(self.eps))

    def _sample_codebook_patches(self, patches: np.ndarray, indices: np.ndarray, *, seed: int):
        torch = self._torch()
        rng = np.random.default_rng(seed)
        indices = np.asarray(indices, dtype=int)
        if int(self.max_codebook_images) > 0 and len(indices) > int(self.max_codebook_images):
            indices = rng.choice(indices, size=int(self.max_codebook_images), replace=False)
        n_images = len(indices)
        n_fg = int(patches.shape[1])
        per_image = min(int(self.patches_per_image_for_codebook), n_fg)
        patch_ids = rng.integers(0, n_fg, size=(n_images, per_image), endpoint=False)
        sampled = np.empty((n_images, per_image, int(patches.shape[2])), dtype=np.float32)
        batch_size = max(1, int(self.image_batch_size))
        for start in range(0, n_images, batch_size):
            end = min(start + batch_size, n_images)
            arr = np.asarray(patches[indices[start:end]], dtype=np.float32)
            sampled[start:end] = arr[np.arange(end - start)[:, None], patch_ids[start:end]]
        flat = sampled.reshape(-1, sampled.shape[-1])
        x = torch.as_tensor(flat, dtype=torch.float32, device=self._device())
        return x / x.norm(dim=-1, keepdim=True).clamp_min(float(self.eps))

    def _spherical_kmeans(self, samples, *, seed: int):
        torch = self._torch()
        n_samples, dim = samples.shape
        k = min(int(self.num_prototypes), n_samples)
        gen = torch.Generator(device=samples.device).manual_seed(int(seed))
        perm = torch.randperm(n_samples, generator=gen, device=samples.device)
        centers = samples[perm[:k]].clone()
        if k < int(self.num_prototypes):
            extra = samples[torch.randint(0, n_samples, (int(self.num_prototypes) - k,), generator=gen, device=samples.device)]
            centers = torch.cat([centers, extra], dim=0)
        centers = centers / centers.norm(dim=1, keepdim=True).clamp_min(float(self.eps))
        batch_size = max(1, int(self.kmeans_batch_size))
        for _ in range(max(1, int(self.kmeans_iter))):
            sums = torch.zeros_like(centers)
            counts = torch.zeros(centers.shape[0], dtype=torch.float32, device=samples.device)
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                logits = samples[start:end] @ centers.T
                labels = torch.argmax(logits, dim=1)
                sums.index_add_(0, labels, samples[start:end])
                counts.index_add_(0, labels, torch.ones(end - start, dtype=torch.float32, device=samples.device))
            nonempty = counts > 0
            updated = sums[nonempty] / counts[nonempty, None].clamp_min(1.0)
            centers[nonempty] = updated / updated.norm(dim=1, keepdim=True).clamp_min(float(self.eps))
        return centers.contiguous()

    def _counts_for_class(self, patches: np.ndarray, indices: np.ndarray, prototypes):
        torch = self._torch()
        counts = torch.zeros(prototypes.shape[0], dtype=torch.float32, device=prototypes.device)
        batch_size = max(1, int(self.image_batch_size))
        for start in range(0, len(indices), batch_size):
            end = min(start + batch_size, len(indices))
            x = self._load_patch_batch(patches, indices[start:end], device=prototypes.device)
            logits = float(self.kappa) * torch.einsum("bpd,md->bpm", x, prototypes)
            assign = torch.softmax(logits, dim=2)
            counts += assign.sum(dim=(0, 1))
        return counts

    def _clr(self, h):
        torch = self._torch()
        log_h = torch.log(h.clamp_min(float(self.epsilon)))
        return log_h - torch.mean(log_h, dim=1, keepdim=True)

    def _raw_components(self, x, model: dict[str, object]):
        torch = self._torch()
        prototypes = model["prototypes"]
        log_pi = model["log_pi"]
        logits = float(self.kappa) * torch.einsum("bpd,md->bpm", x, prototypes) + log_pi
        patch_nll = -torch.logsumexp(logits, dim=2).mean(dim=1)
        assign = torch.softmax(logits, dim=2)
        hist = assign.mean(dim=1)
        comp_vec = self._clr(hist + float(self.epsilon))
        return patch_nll, comp_vec

    def _fit_lowrank_diag(self, comp_vectors):
        torch = self._torch()
        x = comp_vectors.to(torch.float64)
        mean = x.mean(dim=0)
        centered = x - mean
        n, dim = centered.shape
        if n > 1:
            cov = (centered.T @ centered) / float(n - 1)
        else:
            cov = torch.eye(dim, dtype=torch.float64, device=x.device)
        rank = max(1, min(int(self.lowrank_dim), dim - 1, n - 1 if n > 1 else 1))
        eigval, eigvec = torch.linalg.eigh(cov)
        eigval = torch.flip(eigval, dims=(0,))
        eigvec = torch.flip(eigvec, dims=(1,))
        top_val = eigval[:rank].clamp_min(float(self.eps))
        top_vec = eigvec[:, :rank].contiguous()
        lowrank_diag = torch.sum((top_vec * top_vec) * top_val[None, :], dim=1)
        diag = torch.diag(cov) - lowrank_diag
        diag = diag.clamp_min(float(self.epsilon))
        dinv_u = top_vec / diag[:, None]
        middle = torch.diag(1.0 / top_val) + top_vec.T @ dinv_u
        middle_inv = torch.linalg.inv(middle)
        return {
            "mean": mean.to(torch.float32).contiguous(),
            "u": top_vec.to(torch.float32).contiguous(),
            "diag": diag.to(torch.float32).contiguous(),
            "middle_inv": middle_inv.to(torch.float32).contiguous(),
        }

    def _comp_energy(self, comp_vec, model: dict[str, object]):
        torch = self._torch()
        stats = model["comp_model"]
        x = comp_vec - stats["mean"]
        diag = stats["diag"].clamp_min(float(self.epsilon))
        x_dinv = x / diag
        base = torch.sum(x * x_dinv, dim=1)
        proj = x_dinv @ stats["u"]
        corr = torch.sum((proj @ stats["middle_inv"]) * proj, dim=1)
        return 0.5 * torch.clamp(base - corr, min=0.0)

    def _components_for_indices(self, patches: np.ndarray, indices: np.ndarray, model: dict[str, object]):
        torch = self._torch()
        patch_vals = []
        comp_vecs = []
        batch_size = max(1, int(self.image_batch_size))
        for start in range(0, len(indices), batch_size):
            end = min(start + batch_size, len(indices))
            x = self._load_patch_batch(patches, indices[start:end], device=model["prototypes"].device)
            patch_nll, comp_vec = self._raw_components(x, model)
            patch_vals.append(patch_nll)
            comp_vecs.append(comp_vec)
        return torch.cat(patch_vals, dim=0), torch.cat(comp_vecs, dim=0)

    def fit_class_model(
        self,
        class_name: str,
        patches: np.ndarray,
        train_indices: np.ndarray,
        cal_indices: np.ndarray,
        *,
        seed: int = 0,
    ) -> dict[str, object]:
        torch = self._torch()
        with torch.no_grad():
            samples = self._sample_codebook_patches(patches, train_indices, seed=seed)
            prototypes = self._spherical_kmeans(samples, seed=seed)
            counts = self._counts_for_class(patches, np.asarray(train_indices, dtype=int), prototypes)
            pi = (counts + float(self.alpha)) / (counts.sum() + float(self.alpha) * prototypes.shape[0])
            model: dict[str, object] = {
                "class_name": class_name,
                "prototypes": prototypes,
                "log_pi": torch.log(pi.clamp_min(float(self.eps))).contiguous(),
            }
            train_patch, train_comp_vec = self._components_for_indices(patches, np.asarray(train_indices, dtype=int), model)
            comp_model = self._fit_lowrank_diag(train_comp_vec)
            model["comp_model"] = comp_model
            train_comp = self._comp_energy(train_comp_vec, model)
            patch_med, patch_iqr = _robust_stats_torch(train_patch, self.eps)
            comp_med, comp_iqr = _robust_stats_torch(train_comp, self.eps)
            model["patch_median"] = patch_med.to(torch.float32)
            model["patch_iqr"] = patch_iqr.to(torch.float32)
            model["comp_median"] = comp_med.to(torch.float32)
            model["comp_iqr"] = comp_iqr.to(torch.float32)
            cal_patch, cal_comp_vec = self._components_for_indices(patches, np.asarray(cal_indices, dtype=int), model)
            cal_comp = self._comp_energy(cal_comp_vec, model)
            cal_components = self._standardized_components(cal_patch, cal_comp, model)
            model["cal_scores"] = {
                "nll": torch.sort(cal_components["nll"]).values,
                "comp": torch.sort(cal_components["comp"]).values,
                "full": torch.sort(cal_components["full"]).values,
            }
            self.models_[class_name] = model
            return model

    def _standardized_components(self, patch_nll, comp_energy, model: dict[str, object]):
        e_patch = (patch_nll - model["patch_median"]) / model["patch_iqr"].clamp_min(float(self.eps))
        e_comp = (comp_energy - model["comp_median"]) / model["comp_iqr"].clamp_min(float(self.eps))
        return {"nll": e_patch, "comp": e_comp, "full": e_patch + e_comp}

    def class_energies(self, patches: np.ndarray, indices: np.ndarray, model: dict[str, object], variant: str):
        torch = self._torch()
        outs = []
        variant = variant.lower()
        if variant not in {"nll", "comp", "full"}:
            raise ValueError(f"Unknown PALS variant: {variant}")
        batch_size = max(1, int(self.image_batch_size))
        with torch.no_grad():
            for start in range(0, len(indices), batch_size):
                end = min(start + batch_size, len(indices))
                x = self._load_patch_batch(patches, indices[start:end], device=model["prototypes"].device)
                patch_nll, comp_vec = self._raw_components(x, model)
                comp = self._comp_energy(comp_vec, model)
                components = self._standardized_components(patch_nll, comp, model)
                outs.append(components[variant])
        return torch.cat(outs, dim=0)

    def id_scores(self, patches: np.ndarray, indices: np.ndarray, models: list[dict[str, object]], *, variant: str) -> np.ndarray:
        torch = self._torch()
        variant = variant.lower()
        with torch.no_grad():
            best_p = torch.zeros(len(indices), dtype=torch.float32, device=models[0]["prototypes"].device)
            for model in models:
                energy = self.class_energies(patches, indices, model, variant)
                cal = model["cal_scores"][variant]
                rank = len(cal) - torch.searchsorted(cal, energy, right=False)
                pvals = (1.0 + rank.to(torch.float32)) / (1.0 + len(cal))
                best_p = torch.maximum(best_p, pvals)
            return torch.log(best_p.clamp_min(float(self.eps))).detach().cpu().numpy()


def pals_from_env() -> PALSDetector:
    return PALSDetector(
        num_prototypes=int(os.environ.get("MAF07_PALS_PROTOTYPES", "384")),
        patches_per_image_for_codebook=int(os.environ.get("MAF07_PALS_CODEBOOK_PATCHES_PER_IMAGE", "16")),
        kmeans_iter=int(os.environ.get("MAF07_PALS_KMEANS_ITER", "30")),
        kappa=float(os.environ.get("MAF07_PALS_KAPPA", "20")),
        alpha=float(os.environ.get("MAF07_PALS_ALPHA", "0.1")),
        epsilon=float(os.environ.get("MAF07_PALS_EPSILON", "1e-5")),
        lowrank_dim=int(os.environ.get("MAF07_PALS_LOWRANK_DIM", "64")),
        max_codebook_images=int(os.environ.get("MAF07_PALS_MAX_CODEBOOK_IMAGES", "4096")),
        device=os.environ.get("MAF07_PALS_DEVICE") or None,
        image_batch_size=int(os.environ.get("MAF07_PALS_IMAGE_BATCH", "48")),
        kmeans_batch_size=int(os.environ.get("MAF07_PALS_KMEANS_BATCH", "8192")),
    )
