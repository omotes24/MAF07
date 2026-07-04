from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.covariance import EmpiricalCovariance, LedoitWolf, OAS
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize

EPS = 1e-12


def _class_order(y: np.ndarray) -> np.ndarray:
    return np.asarray(sorted(np.unique(y)))


def _covariance(features: np.ndarray, estimator: str) -> tuple[np.ndarray, np.ndarray]:
    estimator = estimator.lower()
    x = np.asarray(features, dtype=float)
    center = x.mean(axis=0)
    xc = x - center
    if estimator == "tied_ledoit_wolf":
        model = LedoitWolf().fit(x)
        return model.location_, model.precision_
    if estimator == "tied_empirical":
        model = EmpiricalCovariance().fit(x)
        return model.location_, model.precision_
    if estimator == "oas":
        model = OAS().fit(x)
        return model.location_, model.precision_
    if estimator == "tied_diagonal":
        var = np.var(xc, axis=0) + 1e-6
        return center, np.diag(1.0 / var)
    if estimator == "tied_spherical":
        var = float(np.var(xc) + 1e-6)
        return center, np.eye(x.shape[1]) / var
    raise ValueError(f"Unsupported tied covariance estimator: {estimator}")


def feature_transform(
    x_fit: np.ndarray,
    x_apply: np.ndarray,
    *,
    mode: str = "raw",
    pca_dim: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    mode = mode.lower()
    fit = np.asarray(x_fit, dtype=float)
    app = np.asarray(x_apply, dtype=float)
    if mode == "raw":
        fit = fit.copy()
        app = app.copy()
    elif mode == "l2":
        fit = normalize(fit)
        app = normalize(app)
    elif mode == "centered":
        mu = fit.mean(axis=0, keepdims=True)
        fit = fit - mu
        app = app - mu
    elif mode == "whitened":
        mu = fit.mean(axis=0, keepdims=True)
        std = fit.std(axis=0, keepdims=True) + 1e-6
        fit = (fit - mu) / std
        app = (app - mu) / std
    else:
        raise ValueError(f"Unknown feature_norm: {mode}")
    if pca_dim is not None and pca_dim < fit.shape[1]:
        pca = PCA(n_components=int(pca_dim), svd_solver="randomized", random_state=0)
        fit = pca.fit_transform(fit)
        app = pca.transform(app)
    return fit, app


@dataclass
class MAFScorer:
    tau: float = 1.0
    alpha: float = 0.5
    covariance: str = "tied_ledoit_wolf"
    feature_norm: str = "raw"
    distance: str = "mahalanobis"
    prototype: str = "class_mean"
    pca_dim: int | None = None

    classes_: np.ndarray | None = None
    prototypes_: np.ndarray | None = None
    precision_: np.ndarray | None = None
    std_: np.ndarray | None = None
    train_features_: np.ndarray | None = None
    train_labels_: np.ndarray | None = None

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "MAFScorer":
        x = np.asarray(features, dtype=float)
        y = np.asarray(labels)
        x_fit, x_tx = feature_transform(x, x, mode=self.feature_norm, pca_dim=self.pca_dim)
        classes = _class_order(y)
        prototypes = []
        for cls in classes:
            cls_x = x_tx[y == cls]
            if self.prototype == "class_median":
                prototypes.append(np.median(cls_x, axis=0))
            elif self.prototype == "class_mean":
                prototypes.append(cls_x.mean(axis=0))
            else:
                raise ValueError(f"Unsupported prototype: {self.prototype}")
        _, precision = _covariance(x_tx, self.covariance)
        self.classes_ = classes
        self.prototypes_ = np.vstack(prototypes)
        self.precision_ = precision
        self.std_ = x_tx.std(axis=0) + 1e-6
        self.train_features_ = x_fit
        self.train_labels_ = y
        return self

    def _transform_apply(self, features: np.ndarray) -> np.ndarray:
        if self.train_features_ is None:
            raise RuntimeError("MAFScorer is not fitted")
        _, app = feature_transform(
            self.train_features_,
            np.asarray(features, dtype=float),
            mode=self.feature_norm,
            pca_dim=self.pca_dim,
        )
        return app

    def distances(self, features: np.ndarray) -> np.ndarray:
        if self.prototypes_ is None or self.precision_ is None or self.std_ is None:
            raise RuntimeError("MAFScorer is not fitted")
        x = self._transform_apply(features)
        distance = self.distance.lower()
        if distance in {"mahalanobis", "squared_mahalanobis"}:
            diffs = x[:, None, :] - self.prototypes_[None, :, :]
            d2 = np.einsum("ncd,dd,ncd->nc", diffs, self.precision_, diffs)
            d2 = np.maximum(d2, 0.0)
            return d2 if distance == "squared_mahalanobis" else np.sqrt(d2 + EPS)
        if distance == "euclidean":
            return pairwise_distances(x, self.prototypes_, metric="euclidean")
        if distance == "cosine":
            return pairwise_distances(x, self.prototypes_, metric="cosine")
        if distance == "standardized_euclidean":
            diffs = (x[:, None, :] - self.prototypes_[None, :, :]) / self.std_[None, None, :]
            return np.sqrt(np.sum(diffs**2, axis=2) + EPS)
        raise ValueError(f"Unsupported distance: {self.distance}")

    def probabilities(self, features: np.ndarray) -> np.ndarray:
        d = self.distances(features)
        logits = -d / max(float(self.tau), EPS)
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        return exp / np.sum(exp, axis=1, keepdims=True)

    def score_components(self, features: np.ndarray) -> dict[str, np.ndarray]:
        p = self.probabilities(features)
        class_count = p.shape[1]
        s_conf = p.max(axis=1)
        entropy = -np.sum(p * np.log(p + EPS), axis=1)
        h_norm = entropy / max(np.log(class_count), EPS)
        s_cons = 1.0 - h_norm
        return {"s_conf": s_conf, "s_cons": s_cons, "probabilities": p}

    def score(self, features: np.ndarray) -> np.ndarray:
        comp = self.score_components(features)
        return maf_fusion(comp["s_conf"], comp["s_cons"], alpha=self.alpha, mode="alpha")


def maf_fusion(
    s_conf: np.ndarray,
    s_cons: np.ndarray,
    *,
    alpha: float = 0.5,
    mode: str = "alpha",
) -> np.ndarray:
    conf = np.clip(np.asarray(s_conf, dtype=float), EPS, 1.0)
    cons = np.clip(np.asarray(s_cons, dtype=float), EPS, 1.0)
    mode = mode.lower()
    if mode == "alpha":
        return conf**float(alpha) * cons ** (1.0 - float(alpha))
    if mode == "s_conf_only":
        return conf
    if mode == "s_cons_only":
        return cons
    if mode in {"product", "s_conf * s_cons"}:
        return conf * cons
    if mode in {"sqrt_product", "sqrt(s_conf*s_cons)"}:
        return np.sqrt(conf * cons)
    if mode == "arithmetic_mean":
        return 0.5 * (conf + cons)
    if mode == "harmonic_mean":
        return 2.0 * conf * cons / (conf + cons + EPS)
    if mode == "min":
        return np.minimum(conf, cons)
    if mode == "max":
        return np.maximum(conf, cons)
    if mode == "log":
        return np.log(conf) + np.log(cons)
    raise ValueError(f"Unknown MAF fusion mode: {mode}")


def distance_variant_score(distances: np.ndarray, variant: str) -> np.ndarray:
    d = np.sort(np.asarray(distances, dtype=float), axis=1)
    nearest = d[:, 0]
    second = d[:, 1] if d.shape[1] > 1 else d[:, 0]
    variant = variant.lower()
    if variant == "nearest_mahalanobis":
        return -nearest
    if variant == "distance_margin":
        return second - nearest
    if variant == "distance_ratio":
        return -(nearest / (second + EPS))
    if variant == "entropy":
        p = np.exp(-d)
        p = p / p.sum(axis=1, keepdims=True)
        return np.sum(p * np.log(p + EPS), axis=1)
    if variant == "gini":
        p = np.exp(-d)
        p = p / p.sum(axis=1, keepdims=True)
        return -np.sum(p * (1.0 - p), axis=1)
    if variant == "renyi_entropy":
        p = np.exp(-d)
        p = p / p.sum(axis=1, keepdims=True)
        return -np.log(np.sum(p**2, axis=1) + EPS)
    if variant == "topk_distance_entropy":
        top = d[:, : min(3, d.shape[1])]
        p = np.exp(-top)
        p = p / p.sum(axis=1, keepdims=True)
        return np.sum(p * np.log(p + EPS), axis=1)
    raise ValueError(f"Unknown distance variant: {variant}")
