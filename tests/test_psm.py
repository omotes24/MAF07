from __future__ import annotations

import numpy as np

from maf07.methods.psm import PSMScorer


def test_psm_scores_are_higher_for_shifted_samples() -> None:
    rng = np.random.default_rng(17)
    train = rng.normal(size=(160, 12))
    train[:, :3] *= 2.0
    train[:, 3:] *= 0.2
    id_x = rng.normal(size=(40, 12))
    id_x[:, :3] *= 2.0
    id_x[:, 3:] *= 0.2
    ood_x = id_x.copy()
    ood_x[:, 5:8] += 1.0
    scorer = PSMScorer(r=4, k=30, gamma_local=1e-2, gamma_tail=1e-2, device="cpu", score_batch=16).fit(train)
    scores = scorer.score_samples(np.vstack([id_x, ood_x]))
    assert scores.shape == (80,)
    assert float(scores[:40].mean()) < float(scores[40:].mean())
