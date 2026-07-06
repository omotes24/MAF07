from __future__ import annotations

import numpy as np

from maf07.methods.csr import CSRDetector


def test_csr_scores_off_manifold_samples_as_more_ood() -> None:
    rng = np.random.default_rng(17)
    x0 = rng.normal(loc=-2.0, scale=[0.2, 0.4, 0.3, 0.5], size=(90, 4))
    x1 = rng.normal(loc=2.0, scale=[0.5, 0.3, 0.6, 0.2], size=(90, 4))
    train_x = np.vstack([x0[:60], x1[:60]])
    val_x = np.vstack([x0[60:], x1[60:]])
    train_y = np.array([0] * 60 + [1] * 60)
    val_y = np.array([0] * 30 + [1] * 30)
    id_x = val_x.copy()
    ood_x = np.vstack([
        rng.normal(loc=-2.0, scale=1.8, size=(30, 4)),
        rng.normal(loc=2.0, scale=1.8, size=(30, 4)),
    ])
    eval_x = np.vstack([id_x, ood_x])
    logits = np.column_stack([-np.linalg.norm(eval_x + 2.0, axis=1), -np.linalg.norm(eval_x - 2.0, axis=1)])
    detector = CSRDetector(r=2, topq=2, score_batch=17, device="cpu").fit(
        train_x,
        train_y,
        z_cal=val_x,
        y_cal=val_y,
    )
    ood_scores = detector.score_samples(eval_x, logits)
    id_scores = detector.id_scores(eval_x, logits)
    assert ood_scores.shape == (120,)
    assert float(ood_scores[:60].mean()) < float(ood_scores[60:].mean())
    assert float(id_scores[:60].mean()) > float(id_scores[60:].mean())
