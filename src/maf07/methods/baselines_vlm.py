from __future__ import annotations

import numpy as np
from scipy.special import logsumexp, softmax
from sklearn.preprocessing import normalize


def clip_logits(image_features: np.ndarray, text_features: np.ndarray, logit_scale: float = 100.0) -> np.ndarray:
    return logit_scale * normalize(image_features) @ normalize(text_features).T


def mcm_score(image_features: np.ndarray, text_features: np.ndarray, logit_scale: float = 100.0) -> np.ndarray:
    return softmax(clip_logits(image_features, text_features, logit_scale), axis=1).max(axis=1)


def clip_zero_shot_msp(
    image_features: np.ndarray,
    text_features: np.ndarray,
    logit_scale: float = 100.0,
) -> np.ndarray:
    return mcm_score(image_features, text_features, logit_scale)


def clip_text_energy(
    image_features: np.ndarray,
    text_features: np.ndarray,
    logit_scale: float = 100.0,
    temperature: float = 1.0,
) -> np.ndarray:
    logits = clip_logits(image_features, text_features, logit_scale) / temperature
    return temperature * logsumexp(logits, axis=1)


def tip_adapter_score(
    image_features: np.ndarray,
    cache_features: np.ndarray,
    cache_onehot: np.ndarray,
    text_features: np.ndarray,
    *,
    beta: float = 5.5,
    alpha: float = 1.0,
    logit_scale: float = 100.0,
) -> np.ndarray:
    img = normalize(image_features)
    cache = normalize(cache_features)
    affinity = img @ cache.T
    cache_logits = np.exp(-beta * (1.0 - affinity)) @ cache_onehot
    zshot_logits = clip_logits(img, text_features, logit_scale)
    logits = zshot_logits + alpha * cache_logits
    return softmax(logits, axis=1).max(axis=1)

