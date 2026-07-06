from __future__ import annotations

import numpy as np

from maf07.methods.diagcard import DiagCARDDetector


def test_diagcard_scores_far_samples_as_less_id() -> None:
    rng = np.random.default_rng(41)
    x0 = rng.normal(scale=[1.0, 0.2, 0.4, 0.8], size=(70, 4))
    x1 = rng.normal(scale=[0.5, 1.0, 0.3, 0.6], size=(70, 4)) + 3.0
    train_x = np.vstack([x0[:50], x1[:50]])
    val_x = np.vstack([x0[50:60], x1[50:60]])
    train_y = np.array([0] * 50 + [1] * 50)
    val_y = np.array([0] * 10 + [1] * 10)
    id_x = np.vstack([x0[60:], x1[60:]])
    ood_x = np.vstack([
        rng.normal(size=(10, 4)) + np.array([0.0, 4.0, 0.0, 0.0]),
        rng.normal(size=(10, 4)) + np.array([6.0, 3.0, 3.0, 3.0]),
    ])
    detector = DiagCARDDetector(k=8, device="cpu", score_batch=7).fit(train_x, train_y, z_cal=val_x, y_cal=val_y)
    scores = detector.id_scores(np.vstack([id_x, ood_x]), delta=None, calibrated=True)
    raw = detector.id_scores(np.vstack([id_x, ood_x]), delta=None, calibrated=False)
    huber = detector.id_scores(np.vstack([id_x, ood_x]), delta=1.345, calibrated=True)
    huber_raw = detector.id_scores(np.vstack([id_x, ood_x]), delta=1.345, calibrated=False)
    assert scores.shape == (40,)
    assert np.isfinite(scores).all()
    assert np.isfinite(raw).all()
    assert np.isfinite(huber).all()
    assert np.isfinite(huber_raw).all()
    assert float(scores[:20].mean()) > float(scores[20:].mean())
