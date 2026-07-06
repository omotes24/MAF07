from __future__ import annotations

import numpy as np

from maf07.methods.pals import PALSDetector


def _normalize(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def _images(rng: np.random.Generator, centers: np.ndarray, n: int, k: int) -> np.ndarray:
    choices = rng.integers(0, len(centers), size=(n, k))
    noise = 0.08 * rng.normal(size=(n, k, centers.shape[1]))
    return _normalize(centers[choices] + noise).astype(np.float16)


def test_pals_scores_matching_patch_distribution_as_more_id() -> None:
    rng = np.random.default_rng(31)
    dim = 8
    k = 10
    centers0 = np.eye(dim)[:2]
    centers1 = np.eye(dim)[2:4]
    centers_ood = np.eye(dim)[4:6]
    class0 = _images(rng, centers0, 40, k)
    class1 = _images(rng, centers1, 40, k)
    ood = _images(rng, centers_ood, 12, k)
    patches = np.concatenate([class0, class1, ood], axis=0)
    idx0_train = np.arange(0, 24)
    idx0_cal = np.arange(24, 34)
    idx1_train = np.arange(40, 64)
    idx1_cal = np.arange(64, 74)
    id_eval = np.concatenate([np.arange(34, 40), np.arange(74, 80)])
    ood_eval = np.arange(80, 92)
    detector = PALSDetector(
        num_prototypes=4,
        patches_per_image_for_codebook=4,
        kmeans_iter=3,
        lowrank_dim=2,
        max_codebook_images=24,
        device="cpu",
        image_batch_size=8,
        kmeans_batch_size=16,
    )
    model0 = detector.fit_class_model("a", patches, idx0_train, idx0_cal, seed=1)
    model1 = detector.fit_class_model("b", patches, idx1_train, idx1_cal, seed=2)
    eval_idx = np.concatenate([id_eval, ood_eval])
    scores = detector.id_scores(patches, eval_idx, [model0, model1], variant="full")
    assert scores.shape == (24,)
    assert np.isfinite(scores).all()
    assert float(scores[:12].mean()) > float(scores[12:].mean())
