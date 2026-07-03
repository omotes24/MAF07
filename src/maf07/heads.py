from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, normalize


@dataclass
class HeadResult:
    classes: list[str]
    logits: np.ndarray
    y_true: np.ndarray


def _fit_logistic(train_x: np.ndarray, train_y: np.ndarray, *, cosine: bool = False) -> LogisticRegression:
    x = normalize(train_x) if cosine else train_x
    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced", n_jobs=1)
    clf.fit(x, train_y)
    return clf


def linear_probe_logits(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> np.ndarray:
    clf = _fit_logistic(train_x, train_y, cosine=False)
    return clf.decision_function(test_x)


def cosine_classifier_logits(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> np.ndarray:
    clf = _fit_logistic(train_x, train_y, cosine=True)
    return clf.decision_function(normalize(test_x))


def mlp_head_logits(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> np.ndarray:
    clf = MLPClassifier(hidden_layer_sizes=(512,), alpha=1e-4, max_iter=300, random_state=0)
    clf.fit(train_x, train_y)
    return clf.predict_proba(test_x)


def nearest_class_mean_logits(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> np.ndarray:
    classes = np.asarray(sorted(np.unique(train_y)))
    means = np.vstack([train_x[train_y == cls].mean(axis=0) for cls in classes])
    diffs = test_x[:, None, :] - means[None, :, :]
    return -np.sqrt(np.sum(diffs**2, axis=2))


def fit_closed_head(
    method: str,
    train_x: np.ndarray,
    train_labels: np.ndarray,
    test_x: np.ndarray,
) -> np.ndarray:
    method = method.lower()
    if method == "linear_probe":
        return linear_probe_logits(train_x, train_labels, test_x)
    if method == "mlp_head":
        return mlp_head_logits(train_x, train_labels, test_x)
    if method == "cosine_classifier":
        return cosine_classifier_logits(train_x, train_labels, test_x)
    if method == "nearest_class_mean":
        return nearest_class_mean_logits(train_x, train_labels, test_x)
    if method in {"clip_zero_shot", "clip_prompt_ensemble"}:
        return nearest_class_mean_logits(normalize(train_x), train_labels, normalize(test_x))
    raise ValueError(f"Unknown closed method: {method}")


def encode_labels(labels: np.ndarray | list[str]) -> tuple[np.ndarray, list[str], LabelEncoder]:
    enc = LabelEncoder()
    y = enc.fit_transform(np.asarray(labels))
    return y, [str(c) for c in enc.classes_], enc


def probabilities_from_logits(logits: np.ndarray) -> np.ndarray:
    arr = np.asarray(logits, dtype=float)
    if arr.ndim == 1:
        arr = np.stack([-arr, arr], axis=1)
    return softmax(arr, axis=1)

