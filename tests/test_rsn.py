from __future__ import annotations

import numpy as np

from maf07.methods.diagcard import DiagCARDDetector
from maf07.methods.rsn import RSNDetector


def test_rsn_matches_legacy_diag_huber_raw_scores() -> None:
    rng = np.random.default_rng(47)
    x0 = rng.normal(scale=[1.0, 0.2, 0.4, 0.8], size=(70, 4))
    x1 = rng.normal(scale=[0.5, 1.0, 0.3, 0.6], size=(70, 4)) + 3.0
    train_x = np.vstack([x0[:50], x1[:50]])
    val_x = np.vstack([x0[50:60], x1[50:60]])
    train_y = np.array([0] * 50 + [1] * 50)
    val_y = np.array([0] * 10 + [1] * 10)
    eval_x = np.vstack([x0[60:], x1[60:], rng.normal(size=(20, 4)) + 5.0])

    legacy = DiagCARDDetector(k=8, delta=1.345, device="cpu", score_batch=7).fit(
        train_x, train_y, z_cal=val_x, y_cal=val_y
    )
    rsn = RSNDetector(k=8, delta=1.345, device="cpu", score_batch=7).fit(
        train_x, train_y, z_cal=val_x, y_cal=val_y
    )
    np.testing.assert_allclose(
        rsn.id_scores(eval_x),
        legacy.id_scores(eval_x, delta=1.345, calibrated=False),
        rtol=1e-6,
        atol=1e-6,
    )
