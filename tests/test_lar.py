from __future__ import annotations

import numpy as np

from maf07.methods.lar import LARScorer


def test_lar_scores_are_higher_for_off_manifold_samples() -> None:
    rng = np.random.default_rng(9)
    train = np.column_stack(
        [
            rng.normal(0.0, 2.0, 90),
            rng.normal(0.0, 0.03, 90),
            rng.normal(0.0, 0.03, 90),
        ]
    )
    id_x = np.column_stack(
        [
            rng.normal(0.0, 2.0, 30),
            rng.normal(0.0, 0.03, 30),
            rng.normal(0.0, 0.03, 30),
        ]
    )
    ood_x = np.column_stack(
        [
            rng.normal(0.0, 2.0, 30),
            rng.normal(1.5, 0.05, 30),
            rng.normal(1.5, 0.05, 30),
        ]
    )
    scorer = LARScorer(k=30, gamma=1e-2, beta=1.0, normalize=False, device="cpu", score_batch=8).fit(train)
    scores = scorer.score_samples(np.vstack([id_x, ood_x]))
    assert scores.shape == (60,)
    assert float(scores[:30].mean()) < float(scores[30:].mean())
