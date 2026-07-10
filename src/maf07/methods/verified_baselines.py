from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp, softmax
from scipy.stats import weibull_min
from sklearn.covariance import EmpiricalCovariance
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

EPS = 1e-12


class UnsupportedVerifiedBaseline(ValueError):
    """Raised when a paper-faithful score cannot be computed from cached features."""


@dataclass(frozen=True)
class RidgeLinearProbe:
    """Deterministic linear probe used by every classifier-dependent baseline."""

    weight: np.ndarray
    bias: np.ndarray
    classes: np.ndarray

    def logits(self, features: np.ndarray) -> np.ndarray:
        x = np.asarray(features, dtype=np.float32)
        return x @ self.weight.T + self.bias


def fit_ridge_linear_probe(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    *,
    reg: float | None = None,
    device: str | None = None,
) -> RidgeLinearProbe:
    """Fit the closed-form ridge probe used by the MAF07 evaluation protocol."""

    import torch

    x_np = np.asarray(train_features, dtype=np.float32)
    y_np = np.asarray(train_labels, dtype=int)
    classes = np.asarray(sorted(np.unique(y_np)), dtype=int)
    class_to_col = {int(cls): i for i, cls in enumerate(classes)}
    y_cols = np.asarray([class_to_col[int(v)] for v in y_np], dtype=np.int64)

    device_name = device or os.environ.get("MAF07_TORCH_DEVICE")
    if device_name is None:
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device_name)
    x = torch.as_tensor(x_np, dtype=torch.float32, device=torch_device)
    y_idx = torch.as_tensor(y_cols, dtype=torch.long, device=torch_device)
    targets = torch.nn.functional.one_hot(y_idx, num_classes=len(classes)).to(torch.float32)

    x_aug = torch.cat(
        [x, torch.ones((x.shape[0], 1), dtype=x.dtype, device=torch_device)],
        dim=1,
    )
    ridge = float(reg if reg is not None else os.environ.get("MAF07_RIDGE_LAMBDA", "1e-3"))
    gram = x_aug.T @ x_aug
    penalty = torch.eye(gram.shape[0], dtype=x.dtype, device=torch_device)
    penalty[-1, -1] = 0.0
    coefficients = torch.linalg.solve(gram + ridge * penalty, x_aug.T @ targets)
    weight = coefficients[:-1].T.detach().cpu().numpy()
    bias = coefficients[-1].detach().cpu().numpy()
    return RidgeLinearProbe(weight=weight, bias=bias, classes=classes)


def score_fingerprint(scores: np.ndarray) -> str:
    values = np.ascontiguousarray(np.asarray(scores, dtype="<f8"))
    return hashlib.sha256(values.tobytes()).hexdigest()


def msp_score(logits: np.ndarray) -> np.ndarray:
    return softmax(np.asarray(logits, dtype=np.float64), axis=1).max(axis=1)


def entropy_score(logits: np.ndarray) -> np.ndarray:
    probs = softmax(np.asarray(logits, dtype=np.float64), axis=1)
    return np.sum(probs * np.log(probs + EPS), axis=1)


def energy_score(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64) / float(temperature)
    return float(temperature) * logsumexp(z, axis=1)


def maxlogit_score(logits: np.ndarray) -> np.ndarray:
    return np.asarray(logits, dtype=np.float64).max(axis=1)


def gen_score(logits: np.ndarray, gamma: float = 0.1, top_m: int = 100) -> np.ndarray:
    """Official GEN score: negative top-M generalized entropy."""

    probs = softmax(np.asarray(logits, dtype=np.float64), axis=1)
    m = min(max(1, int(top_m)), probs.shape[1])
    top = np.sort(probs, axis=1)[:, -m:]
    return -np.sum((top**float(gamma)) * ((1.0 - top) ** float(gamma)), axis=1)


def gradnorm_score(
    features: np.ndarray,
    logits: np.ndarray,
    *,
    temperature: float = 1.0,
) -> np.ndarray:
    """Per-sample L1 norm of the official GradNorm final-layer gradient.

    The reference loss is ``-sum_c log softmax(z / T)_c``. Its gradient with
    respect to the final-layer weight is the outer product of ``C*p - 1`` and
    the penultimate feature, so the L1 norm can be evaluated without autograd.
    """

    x = np.asarray(features, dtype=np.float64)
    z = np.asarray(logits, dtype=np.float64) / float(temperature)
    probs = softmax(z, axis=1)
    class_term = np.sum(np.abs(probs * probs.shape[1] - 1.0), axis=1)
    feature_term = np.sum(np.abs(x), axis=1)
    return class_term * feature_term / float(temperature)


def kl_matching_score(
    logits: np.ndarray,
    train_logits: np.ndarray,
    train_labels: np.ndarray,
) -> np.ndarray:
    probs = softmax(np.asarray(logits, dtype=np.float64), axis=1)
    train_probs = softmax(np.asarray(train_logits, dtype=np.float64), axis=1)
    labels = np.asarray(train_labels, dtype=int)
    templates = np.vstack(
        [train_probs[labels == cls].mean(axis=0) for cls in sorted(np.unique(labels))]
    )
    templates = np.clip(templates, EPS, 1.0)
    kl = np.sum(
        probs[:, None, :] * (np.log(probs[:, None, :] + EPS) - np.log(templates[None, :, :])),
        axis=2,
    )
    return -kl.min(axis=1)


def _class_means(features: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    classes = np.asarray(sorted(np.unique(labels)), dtype=int)
    means = np.vstack([features[labels == cls].mean(axis=0) for cls in classes])
    return classes, means


def _fit_tied_precision(features: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    _, means = _class_means(features, labels)
    mean_by_label = {int(cls): means[i] for i, cls in enumerate(sorted(np.unique(labels)))}
    residuals = np.vstack(
        [
            features[labels == cls] - mean_by_label[int(cls)]
            for cls in sorted(np.unique(labels))
        ]
    )
    precision = EmpiricalCovariance(assume_centered=True).fit(residuals).precision_
    return means, precision


def _quadratic_distance(
    features: np.ndarray,
    mean: np.ndarray,
    precision: np.ndarray,
) -> np.ndarray:
    centered = np.asarray(features, dtype=np.float64) - np.asarray(mean, dtype=np.float64)
    return np.einsum("nd,de,ne->n", centered, precision, centered, optimize=True)


def _minimum_quadratic_distance(
    features: np.ndarray,
    means: np.ndarray,
    precision: np.ndarray,
) -> np.ndarray:
    best = np.full(len(features), np.inf, dtype=np.float64)
    for mean in means:
        best = np.minimum(best, _quadratic_distance(features, mean, precision))
    return np.maximum(best, 0.0)


def _stable_denominator(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    fallback = np.where(values < 0.0, -EPS, EPS)
    return np.where(np.abs(values) < EPS, fallback, values)


def mahalanobis_score(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    *,
    l2_normalize: bool = False,
) -> np.ndarray:
    train_x = np.asarray(train_features, dtype=np.float64)
    test_x = np.asarray(test_features, dtype=np.float64)
    if l2_normalize:
        train_x = normalize(train_x)
        test_x = normalize(test_x)
    means, precision = _fit_tied_precision(train_x, np.asarray(train_labels, dtype=int))
    return -_minimum_quadratic_distance(test_x, means, precision)


def rmd_score(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
) -> np.ndarray:
    train_x = np.asarray(train_features, dtype=np.float64)
    test_x = np.asarray(test_features, dtype=np.float64)
    labels = np.asarray(train_labels, dtype=int)
    class_means, class_precision = _fit_tied_precision(train_x, labels)
    class_distance = _minimum_quadratic_distance(test_x, class_means, class_precision)

    global_mean = train_x.mean(axis=0)
    global_residuals = train_x - global_mean
    global_precision = EmpiricalCovariance(assume_centered=True).fit(global_residuals).precision_
    global_distance = _quadratic_distance(test_x, global_mean, global_precision)
    return global_distance - class_distance


def knn_score(train_features: np.ndarray, test_features: np.ndarray, k: int = 50) -> np.ndarray:
    train_x = normalize(np.asarray(train_features, dtype=np.float64))
    test_x = normalize(np.asarray(test_features, dtype=np.float64))
    neighbors = min(max(1, int(k)), len(train_x))
    nn = NearestNeighbors(n_neighbors=neighbors, metric="euclidean").fit(train_x)
    distances, _ = nn.kneighbors(test_x)
    return -distances[:, -1]


def vim_score(
    train_features: np.ndarray,
    test_features: np.ndarray,
    probe: RidgeLinearProbe,
    *,
    principal_dim: int | None = None,
) -> np.ndarray:
    train_x = np.asarray(train_features, dtype=np.float64)
    test_x = np.asarray(test_features, dtype=np.float64)
    weight = np.asarray(probe.weight, dtype=np.float64)
    bias = np.asarray(probe.bias, dtype=np.float64)
    origin = -(np.linalg.pinv(weight) @ bias)
    centered = train_x - origin
    covariance = EmpiricalCovariance(assume_centered=True).fit(centered).covariance_
    eigvals, eigvecs = np.linalg.eigh(covariance)
    order = np.argsort(eigvals)[::-1]
    if principal_dim is None:
        principal_dim = 512 if train_x.shape[1] >= 768 else max(1, train_x.shape[1] // 2)
    principal_dim = min(max(1, int(principal_dim)), train_x.shape[1] - 1)
    null_space = np.ascontiguousarray(eigvecs[:, order[principal_dim:]])
    train_residual = np.linalg.norm(centered @ null_space, axis=1)
    train_logits = probe.logits(train_x)
    alpha = float(np.mean(np.max(train_logits, axis=1)) / (np.mean(train_residual) + EPS))
    test_logits = probe.logits(test_x)
    virtual_logit = alpha * np.linalg.norm((test_x - origin) @ null_space, axis=1)
    return energy_score(test_logits) - virtual_logit


def react_score(
    train_features: np.ndarray,
    test_features: np.ndarray,
    probe: RidgeLinearProbe,
    *,
    quantile: float = 0.9,
) -> np.ndarray:
    threshold = float(np.quantile(np.asarray(train_features), float(quantile)))
    clipped = np.clip(np.asarray(test_features), a_min=None, a_max=threshold)
    return energy_score(probe.logits(clipped))


def _topk_mask(features: np.ndarray, percentile: float) -> tuple[np.ndarray, np.ndarray, int]:
    x = np.asarray(features, dtype=np.float64)
    n = x.shape[1]
    k = max(1, n - int(np.round(n * float(percentile) / 100.0)))
    indices = np.argpartition(x, kth=n - k, axis=1)[:, -k:]
    rows = np.arange(len(x))[:, None]
    return x, indices, k


def ash_features(features: np.ndarray, *, variant: str, percentile: float = 95.0) -> np.ndarray:
    x, indices, k = _topk_mask(features, percentile)
    rows = np.arange(len(x))[:, None]
    original_sum = x.sum(axis=1)
    kept = x[rows, indices]
    out = np.zeros_like(x)
    variant = variant.lower()
    if variant == "p":
        out[rows, indices] = kept
    elif variant == "b":
        out[rows, indices] = (original_sum / float(k))[:, None]
    elif variant == "s":
        out[rows, indices] = kept
        kept_sum = kept.sum(axis=1)
        ratio = original_sum / _stable_denominator(kept_sum)
        out *= np.exp(np.clip(ratio, -50.0, 50.0))[:, None]
    else:
        raise ValueError(f"Unknown ASH variant: {variant}")
    return out


def ash_score(
    test_features: np.ndarray,
    probe: RidgeLinearProbe,
    *,
    variant: str,
    percentile: float = 95.0,
) -> np.ndarray:
    shaped = ash_features(test_features, variant=variant, percentile=percentile)
    return energy_score(probe.logits(shaped))


def dice_score(
    train_features: np.ndarray,
    test_features: np.ndarray,
    probe: RidgeLinearProbe,
    *,
    percentile: float = 90.0,
) -> np.ndarray:
    mean_activation = np.asarray(train_features, dtype=np.float64).mean(axis=0)
    contribution = np.asarray(probe.weight, dtype=np.float64) * mean_activation[None, :]
    threshold = float(np.percentile(contribution, float(percentile)))
    masked_weight = np.asarray(probe.weight, dtype=np.float64) * (contribution > threshold)
    logits = np.asarray(test_features, dtype=np.float64) @ masked_weight.T + probe.bias
    return energy_score(logits)


def scale_score(
    test_features: np.ndarray,
    probe: RidgeLinearProbe,
    *,
    percentile: float = 85.0,
) -> np.ndarray:
    x, indices, _ = _topk_mask(test_features, percentile)
    rows = np.arange(len(x))[:, None]
    total = x.sum(axis=1)
    kept = x[rows, indices].sum(axis=1)
    ratio = total / _stable_denominator(kept)
    scaled = x * np.exp(np.clip(ratio, -50.0, 50.0))[:, None]
    return energy_score(probe.logits(scaled))


def nci_score(
    train_features: np.ndarray,
    test_features: np.ndarray,
    probe: RidgeLinearProbe,
    *,
    alpha: float = 1e-4,
) -> np.ndarray:
    x = np.asarray(test_features, dtype=np.float64)
    centered = x - np.asarray(train_features, dtype=np.float64).mean(axis=0)
    pred = np.argmax(probe.logits(x), axis=1)
    direction = np.sum(probe.weight[pred] * centered, axis=1) / (
        np.linalg.norm(centered, axis=1) + EPS
    )
    return direction + float(alpha) * np.linalg.norm(x, ord=1, axis=1)


def _eucos_distance(rows: np.ndarray, mean: np.ndarray) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    euclidean = np.linalg.norm(rows - mean, axis=1) / 200.0
    denom = np.linalg.norm(rows, axis=1) * np.linalg.norm(mean)
    cosine = 1.0 - (rows @ mean) / np.maximum(denom, EPS)
    return euclidean + cosine


@dataclass
class OpenMaxModel:
    means: np.ndarray
    weibull_shape: np.ndarray
    weibull_scale: np.ndarray
    alpha_rank: int

    def id_scores(self, logits: np.ndarray) -> np.ndarray:
        z = np.asarray(logits, dtype=np.float64)
        class_count = z.shape[1]
        alpha = min(max(1, int(self.alpha_rank)), class_count)
        ranks = np.argsort(z, axis=1)[:, ::-1]
        omega = np.zeros_like(z)
        rank_weights = (alpha - np.arange(alpha, dtype=np.float64)) / float(alpha)
        omega[np.arange(len(z))[:, None], ranks[:, :alpha]] = rank_weights[None, :]

        wscore = np.empty_like(z)
        for cls in range(class_count):
            distance = _eucos_distance(z, self.means[cls])
            wscore[:, cls] = weibull_min.cdf(
                distance,
                self.weibull_shape[cls],
                loc=0.0,
                scale=self.weibull_scale[cls],
            )
        revised = z * (1.0 - omega * wscore)
        unknown = np.sum(z - revised, axis=1, keepdims=True)
        extended = np.concatenate([revised, unknown], axis=1)
        probs = softmax(extended, axis=1)
        return 1.0 - probs[:, -1]


def fit_openmax(
    train_logits: np.ndarray,
    train_labels: np.ndarray,
    *,
    tail_size: int = 20,
    alpha_rank: int = 10,
) -> OpenMaxModel:
    logits = np.asarray(train_logits, dtype=np.float64)
    labels = np.asarray(train_labels, dtype=int)
    classes = np.asarray(sorted(np.unique(labels)), dtype=int)
    predictions = np.argmax(logits, axis=1)
    means = []
    shapes = []
    scales = []
    for cls in classes:
        rows = logits[(labels == cls) & (predictions == cls)]
        if len(rows) < 2:
            rows = logits[labels == cls]
        mean = rows.mean(axis=0)
        distances = _eucos_distance(rows, mean)
        tail = np.sort(np.maximum(distances, EPS))[-min(max(2, int(tail_size)), len(distances)) :]
        try:
            shape, _, scale = weibull_min.fit(tail, floc=0.0)
        except (ValueError, FloatingPointError):
            shape, scale = 1.0, float(np.mean(tail) + EPS)
        means.append(mean)
        shapes.append(max(float(shape), EPS))
        scales.append(max(float(scale), EPS))
    return OpenMaxModel(
        means=np.vstack(means),
        weibull_shape=np.asarray(shapes),
        weibull_scale=np.asarray(scales),
        alpha_rank=int(alpha_rank),
    )


VERIFIED_METHODS = frozenset(
    {
        "msp",
        "entropy",
        "energy",
        "maxlogit",
        "gen",
        "gradnorm",
        "kl_matching",
        "mahalanobis",
        "mahalanobispp",
        "rmd",
        "knn",
        "vim",
        "react",
        "ashp",
        "ashb",
        "ashs",
        "dice",
        "scale",
        "nci",
        "openmax",
    }
)

UNSUPPORTED_DINOV2_METHODS = frozenset(
    {
        "odin",
        "mcm",
        "clip_zeroshot_msp",
        "clip_text_energy",
        "tip_adapter",
    }
)


@dataclass
class VerifiedBaselineSuite:
    train_features: np.ndarray
    train_labels: np.ndarray
    val_features: np.ndarray | None = None
    probe: RidgeLinearProbe | None = None

    def __post_init__(self) -> None:
        self.train_features = np.asarray(self.train_features, dtype=np.float32)
        self.train_labels = np.asarray(self.train_labels, dtype=int)
        self.val_features = (
            None
            if self.val_features is None
            else np.asarray(self.val_features, dtype=np.float32)
        )
        if self.probe is None:
            self.probe = fit_ridge_linear_probe(self.train_features, self.train_labels)
        self.train_logits = self.probe.logits(self.train_features)
        self._openmax: OpenMaxModel | None = None
        self._tied_raw: tuple[np.ndarray, np.ndarray] | None = None
        self._tied_l2: tuple[np.ndarray, np.ndarray] | None = None
        self._global_raw: tuple[np.ndarray, np.ndarray] | None = None
        self._eval_object_id: int | None = None
        self._eval_logits: np.ndarray | None = None

    def _logits(self, features: np.ndarray) -> np.ndarray:
        object_id = id(features)
        if object_id != self._eval_object_id or self._eval_logits is None:
            self._eval_object_id = object_id
            self._eval_logits = self.probe.logits(features)
        return self._eval_logits

    def score(self, method: str, test_features: np.ndarray) -> np.ndarray:
        name = method.lower()
        if name in UNSUPPORTED_DINOV2_METHODS:
            raise UnsupportedVerifiedBaseline(
                f"{name} cannot be reported as its original method from cached DINOv2 features"
            )
        if name == "mah_mindist":
            raise UnsupportedVerifiedBaseline(
                "mah_mindist is an alias of nearest-class Mahalanobis "
                "and is not a separate baseline"
            )
        if name not in VERIFIED_METHODS:
            raise ValueError(f"Unknown verified baseline: {name}")

        x = np.asarray(test_features, dtype=np.float32)
        logits = self._logits(test_features)
        if name == "msp":
            return msp_score(logits)
        if name == "entropy":
            return entropy_score(logits)
        if name == "energy":
            return energy_score(logits)
        if name == "maxlogit":
            return maxlogit_score(logits)
        if name == "gen":
            return gen_score(logits)
        if name == "gradnorm":
            return gradnorm_score(x, logits)
        if name == "kl_matching":
            return kl_matching_score(logits, self.train_logits, self.train_labels)
        if name == "mahalanobis":
            if self._tied_raw is None:
                self._tied_raw = _fit_tied_precision(self.train_features, self.train_labels)
            means, precision = self._tied_raw
            return -_minimum_quadratic_distance(x, means, precision)
        if name == "mahalanobispp":
            train_l2 = normalize(self.train_features)
            if self._tied_l2 is None:
                self._tied_l2 = _fit_tied_precision(train_l2, self.train_labels)
            means, precision = self._tied_l2
            return -_minimum_quadratic_distance(normalize(x), means, precision)
        if name == "rmd":
            if self._tied_raw is None:
                self._tied_raw = _fit_tied_precision(self.train_features, self.train_labels)
            class_means, class_precision = self._tied_raw
            class_distance = _minimum_quadratic_distance(x, class_means, class_precision)
            if self._global_raw is None:
                global_mean = self.train_features.mean(axis=0)
                residuals = self.train_features - global_mean
                global_precision = EmpiricalCovariance(assume_centered=True).fit(
                    residuals
                ).precision_
                self._global_raw = global_mean, global_precision
            global_mean, global_precision = self._global_raw
            return _quadratic_distance(x, global_mean, global_precision) - class_distance
        if name == "knn":
            return knn_score(self.train_features, x)
        if name == "vim":
            return vim_score(self.train_features, x, self.probe)
        if name == "react":
            return react_score(self.train_features, x, self.probe)
        if name in {"ashp", "ashb", "ashs"}:
            return ash_score(x, self.probe, variant=name[-1])
        if name == "dice":
            return dice_score(self.train_features, x, self.probe)
        if name == "scale":
            return scale_score(x, self.probe)
        if name == "nci":
            return nci_score(self.train_features, x, self.probe)
        if name == "openmax":
            if self._openmax is None:
                self._openmax = fit_openmax(self.train_logits, self.train_labels)
            return self._openmax.id_scores(logits)
        raise AssertionError(f"Unreachable verified baseline branch: {name}")
