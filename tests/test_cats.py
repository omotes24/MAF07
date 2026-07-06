from __future__ import annotations

import numpy as np

from maf07.methods.cats import CATSDetector


def test_cats_scores_off_support_samples_as_more_ood() -> None:
    rng = np.random.default_rng(23)
    t0 = rng.normal(size=(80, 1))
    x0 = np.hstack([t0, 0.15 * rng.normal(size=(80, 1)), 0.05 * rng.normal(size=(80, 4))])
    t1 = rng.normal(size=(80, 1))
    x1 = np.hstack([0.15 * rng.normal(size=(80, 1)), t1, 0.05 * rng.normal(size=(80, 4))]) + 2.0
    train_x = np.vstack([x0[:55], x1[:55]])
    val_x = np.vstack([x0[55:], x1[55:]])
    train_y = np.array([0] * 55 + [1] * 55)
    val_y = np.array([0] * 25 + [1] * 25)
    id_x = val_x.copy()
    ood_x = np.vstack([
        np.hstack([rng.normal(size=(25, 1)), rng.normal(loc=2.0, size=(25, 1)), rng.normal(size=(25, 4))]),
        np.hstack([rng.normal(loc=2.0, size=(25, 1)), rng.normal(size=(25, 1)), rng.normal(size=(25, 4))]),
    ])
    eval_x = np.vstack([id_x, ood_x])
    detector = CATSDetector(
        anchors_per_class=12,
        local_k=12,
        tangent_dim=2,
        candidate_anchors=4,
        score_batch=13,
        device="cpu",
    ).fit(train_x, train_y, z_cal=val_x, y_cal=val_y)
    ood_scores = detector.score_samples(eval_x)
    id_scores = detector.id_scores(eval_x)
    assert ood_scores.shape == (100,)
    assert float(ood_scores[:50].mean()) < float(ood_scores[50:].mean())
    assert float(id_scores[:50].mean()) > float(id_scores[50:].mean())
