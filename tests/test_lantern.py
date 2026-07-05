from __future__ import annotations

import numpy as np

from maf07.methods.lantern import LANTERNDetector


def test_lantern_scores_are_higher_for_off_manifold_samples() -> None:
    rng = np.random.default_rng(7)
    x0 = np.column_stack(
        [
            rng.normal(-2.0, 0.6, 80),
            rng.normal(0.0, 0.03, 80),
            rng.normal(0.0, 0.03, 80),
            rng.normal(0.0, 0.03, 80),
        ]
    )
    x1 = np.column_stack(
        [
            rng.normal(2.0, 0.6, 80),
            rng.normal(0.0, 0.03, 80),
            rng.normal(0.0, 0.03, 80),
            rng.normal(0.0, 0.03, 80),
        ]
    )
    train_x = np.vstack([x0[:60], x1[:60]])
    val_x = np.vstack([x0[60:], x1[60:]])
    train_y = np.array([0] * 60 + [1] * 60)
    val_y = np.array([0] * 20 + [1] * 20)
    train_logits = np.column_stack([-train_x[:, 0], train_x[:, 0]])
    val_logits = np.column_stack([-val_x[:, 0], val_x[:, 0]])
    id_x = val_x.copy()
    ood_x = np.column_stack(
        [
            rng.normal(0.0, 0.5, 40),
            rng.normal(3.0, 0.2, 40),
            rng.normal(3.0, 0.2, 40),
            rng.normal(3.0, 0.2, 40),
        ]
    )
    eval_x = np.vstack([id_x, ood_x])
    eval_logits = np.column_stack([-eval_x[:, 0], eval_x[:, 0]])
    detector = LANTERNDetector(
        n_per_patch=30,
        max_patches_per_class=4,
        min_patch_samples=10,
        max_rank=2,
        topq=2,
        candidate_patches=1,
        normalize=False,
    ).fit(
        train_x,
        train_y,
        logits_train=train_logits,
        z_cal=val_x,
        y_cal=val_y,
        logits_cal=val_logits,
    )
    ood_scores = detector.score_samples(eval_x, eval_logits)
    id_scores = detector.id_scores(eval_x, eval_logits)
    assert ood_scores.shape == (80,)
    assert id_scores.shape == (80,)
    assert float(ood_scores[:40].mean()) < float(ood_scores[40:].mean())
    assert float(id_scores[:40].mean()) > float(id_scores[40:].mean())
