from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors

EPS = 1e-12


def _l2_normalize(x: np.ndarray, eps: float = EPS) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    norm = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(norm, eps)


def _class_logits(logits: np.ndarray | None) -> np.ndarray | None:
    if logits is None:
        return None
    arr = np.asarray(logits, dtype=np.float64)
    if arr.ndim == 1:
        arr = np.stack([-arr, arr], axis=1)
    return arr


@dataclass
class LANTERNDetector:
    n_per_patch: int = 200
    max_patches_per_class: int = 16
    min_patch_samples: int = 32
    var_ratio: float = 0.90
    max_rank: int = 64
    topq: int = 3
    candidate_patches: int = 2
    alpha_tan: float = 0.20
    beta_margin: float = 0.50
    shrink: float = 1e-3
    normalize: bool = True
    random_state: int = 0
    eps: float = 1e-8

    classes_: np.ndarray | None = None
    patches_: dict[int, list[dict[str, object]]] = field(default_factory=dict)
    nn_: dict[int, NearestNeighbors] = field(default_factory=dict)

    def fit(
        self,
        z_train: np.ndarray,
        y_train: np.ndarray,
        *,
        logits_train: np.ndarray | None = None,
        z_cal: np.ndarray | None = None,
        y_cal: np.ndarray | None = None,
        logits_cal: np.ndarray | None = None,
    ) -> "LANTERNDetector":
        train_x = np.asarray(z_train, dtype=np.float64)
        train_y = np.asarray(y_train, dtype=int)
        if self.normalize:
            train_x = _l2_normalize(train_x, self.eps)
        train_logits = _class_logits(logits_train)

        if z_cal is None:
            cal_x = train_x
            cal_y = train_y
            cal_logits = train_logits
        else:
            cal_x = np.asarray(z_cal, dtype=np.float64)
            cal_y = np.asarray(y_cal, dtype=int)
            if self.normalize:
                cal_x = _l2_normalize(cal_x, self.eps)
            cal_logits = _class_logits(logits_cal)

        self.classes_ = np.asarray(sorted(np.unique(train_y)))
        self.patches_ = {}
        self.nn_ = {}

        for cls in self.classes_:
            class_id = int(cls)
            idx = np.where(train_y == class_id)[0]
            class_x = train_x[idx]
            class_logits = train_logits[idx] if train_logits is not None else None
            n_class = len(class_x)
            n_patches = max(1, n_class // max(1, int(self.n_per_patch)))
            n_patches = min(n_patches, int(self.max_patches_per_class))
            if n_class < int(self.min_patch_samples) * 2:
                n_patches = 1

            if n_patches == 1:
                labels = np.zeros(n_class, dtype=int)
                centers = class_x.mean(axis=0, keepdims=True)
            else:
                km = MiniBatchKMeans(
                    n_clusters=n_patches,
                    random_state=int(self.random_state),
                    batch_size=min(4096, max(256, n_class)),
                    n_init=3,
                )
                labels = km.fit_predict(class_x)
                centers = np.asarray(km.cluster_centers_, dtype=np.float64)

            patches: list[dict[str, object]] = []
            for patch_id in range(n_patches):
                members = np.where(labels == patch_id)[0]
                if len(members) < int(self.min_patch_samples):
                    center = centers[patch_id]
                    d2 = np.sum((class_x - center) ** 2, axis=1)
                    k = min(int(self.min_patch_samples), n_class)
                    members = np.argsort(d2)[:k]
                patch_logits = class_logits[members] if class_logits is not None else None
                patches.append(self._fit_patch(class_x[members], patch_logits, class_id))

            self.patches_[class_id] = patches
            patch_centers = np.vstack([np.asarray(patch["mu"], dtype=np.float64) for patch in patches])
            self.nn_[class_id] = NearestNeighbors(
                n_neighbors=min(int(self.candidate_patches), len(patches)),
                metric="euclidean",
            ).fit(patch_centers)

        for patches in self.patches_.values():
            for patch in patches:
                patch["cal_scores"] = []

        for i in range(len(cal_x)):
            class_id = int(cal_y[i])
            if class_id not in self.patches_:
                continue
            patch_idx = self._nearest_patch_indices(class_id, cal_x[i], n_neighbors=1)[0]
            patch = self.patches_[class_id][patch_idx]
            logits_i = cal_logits[i] if cal_logits is not None else None
            patch["cal_scores"].append(self._raw_score(cal_x[i], logits_i, class_id, patch))

        for patches in self.patches_.values():
            for patch in patches:
                scores = np.asarray(patch["cal_scores"], dtype=np.float64)
                if len(scores) == 0:
                    scores = np.array([0.0], dtype=np.float64)
                patch["cal_scores"] = np.sort(scores)
        return self

    def _fit_patch(self, x: np.ndarray, logits: np.ndarray | None, class_id: int) -> dict[str, object]:
        mu = x.mean(axis=0)
        centered = x - mu
        n_samples, n_features = centered.shape
        if n_samples >= 2:
            _, singular, vt = np.linalg.svd(centered, full_matrices=False)
            eig = (singular**2) / max(1, n_samples - 1)
            total = float(eig.sum())
            if total <= self.eps:
                rank = 0
            else:
                cumulative = np.cumsum(eig) / (total + self.eps)
                rank = int(np.searchsorted(cumulative, float(self.var_ratio)) + 1)
                rank = min(rank, int(self.max_rank), vt.shape[0])
        else:
            rank = 0
            eig = np.array([], dtype=np.float64)
            vt = np.zeros((0, n_features), dtype=np.float64)

        if rank > 0:
            basis = vt[:rank].T
            eig_rank = eig[:rank]
            coords = centered @ basis
            residual = centered - coords @ basis.T
        else:
            basis = np.zeros((n_features, 0), dtype=np.float64)
            eig_rank = np.array([], dtype=np.float64)
            residual = centered

        normal2 = np.sum(residual * residual, axis=1)
        normal_scale = float(np.median(normal2) + self.eps)
        margin_floor = None
        margin_scale = None
        if logits is not None and logits.shape[1] > class_id:
            margins = self._margin(logits, class_id)
            margin_floor = float(np.quantile(margins, 0.05))
            median = float(np.median(margins))
            mad = float(np.median(np.abs(margins - median)))
            margin_scale = 1.4826 * mad + self.eps

        return {
            "mu": mu,
            "basis": basis,
            "eig": eig_rank,
            "normal_scale": normal_scale,
            "margin_floor": margin_floor,
            "margin_scale": margin_scale,
        }

    def _margin(self, logits: np.ndarray, class_id: int) -> np.ndarray:
        arr = _class_logits(logits)
        if arr is None:
            raise ValueError("logits are required for margin scoring")
        if arr.shape[1] <= class_id:
            return np.zeros(arr.shape[0], dtype=np.float64)
        own = arr[:, class_id]
        other = arr.copy()
        other[:, class_id] = -np.inf
        return own - np.max(other, axis=1)

    def _raw_score(
        self,
        z: np.ndarray,
        logits: np.ndarray | None,
        class_id: int,
        patch: dict[str, object],
    ) -> float:
        mu = np.asarray(patch["mu"], dtype=np.float64)
        basis = np.asarray(patch["basis"], dtype=np.float64)
        eig = np.asarray(patch["eig"], dtype=np.float64)
        delta = z - mu
        if basis.shape[1] > 0:
            coords = delta @ basis
            residual = delta - coords @ basis.T
            eig_mean = float(np.mean(eig)) if len(eig) else 1.0
            denom = eig + float(self.shrink) * eig_mean + self.eps
            d_tan = float(np.sum((coords * coords) / denom) / max(1, len(eig)))
        else:
            residual = delta
            d_tan = 0.0

        d_perp = float(np.dot(residual, residual) / (float(patch["normal_scale"]) + self.eps))
        score = d_perp + float(self.alpha_tan) * d_tan
        if logits is not None and patch.get("margin_floor") is not None:
            margin = float(self._margin(np.asarray(logits).reshape(1, -1), class_id)[0])
            deficit = max(0.0, float(patch["margin_floor"]) - margin)
            d_margin = (deficit / (float(patch["margin_scale"]) + self.eps)) ** 2
            score += float(self.beta_margin) * d_margin
        return float(score)

    def _raw_scores_batch(
        self,
        x: np.ndarray,
        logits: np.ndarray | None,
        class_id: int,
        patch: dict[str, object],
    ) -> np.ndarray:
        mu = np.asarray(patch["mu"], dtype=np.float64)
        basis = np.asarray(patch["basis"], dtype=np.float64)
        eig = np.asarray(patch["eig"], dtype=np.float64)
        delta = x - mu[None, :]
        if basis.shape[1] > 0:
            coords = delta @ basis
            residual = delta - coords @ basis.T
            eig_mean = float(np.mean(eig)) if len(eig) else 1.0
            denom = eig + float(self.shrink) * eig_mean + self.eps
            d_tan = np.sum((coords * coords) / denom[None, :], axis=1) / max(1, len(eig))
        else:
            residual = delta
            d_tan = np.zeros(len(x), dtype=np.float64)

        d_perp = np.sum(residual * residual, axis=1) / (float(patch["normal_scale"]) + self.eps)
        score = d_perp + float(self.alpha_tan) * d_tan
        if logits is not None and patch.get("margin_floor") is not None:
            margins = self._margin(logits, class_id)
            deficit = np.maximum(0.0, float(patch["margin_floor"]) - margins)
            d_margin = (deficit / (float(patch["margin_scale"]) + self.eps)) ** 2
            score = score + float(self.beta_margin) * d_margin
        return np.asarray(score, dtype=np.float64)

    def _nearest_patch_indices(self, class_id: int, z: np.ndarray, n_neighbors: int | None = None) -> np.ndarray:
        patches = self.patches_[class_id]
        k = min(int(n_neighbors or self.candidate_patches), len(patches))
        _, patch_ids = self.nn_[class_id].kneighbors(z.reshape(1, -1), n_neighbors=k)
        return patch_ids[0].astype(int)

    def _p_value(self, raw_score: float, cal_scores: np.ndarray) -> float:
        idx = int(np.searchsorted(cal_scores, raw_score, side="left"))
        tail_count = len(cal_scores) - idx
        return float((tail_count + 1.0) / (len(cal_scores) + 1.0))

    def _p_values(self, raw_scores: np.ndarray, cal_scores: np.ndarray) -> np.ndarray:
        idx = np.searchsorted(cal_scores, raw_scores, side="left")
        tail_count = len(cal_scores) - idx
        return (tail_count + 1.0) / (len(cal_scores) + 1.0)

    def score_samples(self, z: np.ndarray, logits: np.ndarray | None = None) -> np.ndarray:
        if not self.patches_:
            raise RuntimeError("LANTERNDetector is not fitted")
        x = np.asarray(z, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if self.normalize:
            x = _l2_normalize(x, self.eps)
        class_logits = _class_logits(logits)
        best_p = np.zeros(len(x), dtype=np.float64)
        all_classes = list(self.patches_.keys())
        if class_logits is not None:
            top = np.argsort(class_logits, axis=1)[:, ::-1][:, : min(int(self.topq), class_logits.shape[1])]

        for class_id in all_classes:
            if class_logits is None:
                sample_idx = np.arange(len(x))
            else:
                sample_idx = np.where(np.any(top == int(class_id), axis=1))[0]
                if len(sample_idx) == 0:
                    continue
            xs = x[sample_idx]
            patch_count = len(self.patches_[class_id])
            k = min(int(self.candidate_patches), patch_count)
            _, patch_ids = self.nn_[class_id].kneighbors(xs, n_neighbors=k)
            for patch_idx in np.unique(patch_ids):
                row_mask = np.any(patch_ids == int(patch_idx), axis=1)
                rows = sample_idx[row_mask]
                patch = self.patches_[class_id][int(patch_idx)]
                raw = self._raw_scores_batch(
                    x[rows],
                    class_logits[rows] if class_logits is not None else None,
                    int(class_id),
                    patch,
                )
                pvals = self._p_values(raw, np.asarray(patch["cal_scores"], dtype=np.float64))
                best_p[rows] = np.maximum(best_p[rows], pvals)

        missing = best_p <= 0.0
        if np.any(missing):
            best_p[missing] = self.eps
        return -np.log(np.maximum(best_p, self.eps))

    def id_scores(self, z: np.ndarray, logits: np.ndarray | None = None) -> np.ndarray:
        return -self.score_samples(z, logits)
