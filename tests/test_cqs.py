from __future__ import annotations

import numpy as np

from maf07.methods.cqs import CQSDetector


def test_cqs_scores_class_conditional_off_manifold_samples() -> None:
    rng = np.random.default_rng(13)
    x0 = rng.normal(loc=-2.0, scale=0.25, size=(80, 4))
    x1 = rng.normal(loc=2.0, scale=0.60, size=(80, 4))
    train_x = np.vstack([x0[:60], x1[:60]])
    val_x = np.vstack([x0[60:], x1[60:]])
    train_y = np.array([0] * 60 + [1] * 60)
    val_y = np.array([0] * 20 + [1] * 20)
    id_x = val_x.copy()
    ood_x = np.vstack([
        rng.normal(loc=-2.0, scale=1.5, size=(20, 4)),
        rng.normal(loc=2.0, scale=2.0, size=(20, 4)),
    ])
    eval_x = np.vstack([id_x, ood_x])
    logits = np.column_stack([-np.linalg.norm(eval_x + 2.0, axis=1), -np.linalg.norm(eval_x - 2.0, axis=1)])
    detector = CQSDetector(k=5, topq=2, normalize=False).fit(train_x, train_y, z_cal=val_x, y_cal=val_y)
    ood_scores = detector.score_samples(eval_x, logits)
    id_scores = detector.id_scores(eval_x, logits)
    assert ood_scores.shape == (80,)
    assert float(ood_scores[:40].mean()) < float(ood_scores[40:].mean())
    assert float(id_scores[:40].mean()) > float(id_scores[40:].mean())
