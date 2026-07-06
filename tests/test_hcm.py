from __future__ import annotations

import numpy as np

from maf07.methods.hcm import HCMDetector


def test_hcm_scores_broad_shift_as_more_ood_than_one_axis_outlier() -> None:
    rng = np.random.default_rng(17)
    train_x = rng.normal(size=(120, 8))
    val_x = rng.normal(size=(40, 8))
    train_y = np.zeros(120, dtype=int)
    val_y = np.zeros(40, dtype=int)

    detector = HCMDetector(device="cpu", score_batch=11).fit(train_x, train_y, z_cal=val_x, y_cal=val_y)
    stats = detector.class_stats_[0]
    mu = stats["mu"].detach().cpu().numpy()
    u = stats["U"].detach().cpu().numpy()
    lam_sqrt = stats["lam_sqrt"].detach().cpu().numpy()
    one_axis_t = np.zeros((30, 8))
    one_axis_t[:, 0] = 8.0
    broad_t = np.full((30, 8), 2.5)
    one_axis = mu + (one_axis_t * lam_sqrt) @ u.T
    broad = mu + (broad_t * lam_sqrt) @ u.T
    one_axis_scores = detector.id_scores(one_axis, delta=1.345, calibrated=False)
    broad_scores = detector.id_scores(broad, delta=1.345, calibrated=False)
    assert one_axis_scores.shape == (30,)
    assert float(one_axis_scores.mean()) > float(broad_scores.mean())


def test_hcm_calibrated_scores_have_expected_shape() -> None:
    rng = np.random.default_rng(19)
    x0 = rng.normal(size=(70, 6))
    x1 = rng.normal(size=(70, 6)) + 2.5
    detector = HCMDetector(device="cpu", topq=2, score_batch=9).fit(
        np.vstack([x0[:50], x1[:50]]),
        np.array([0] * 50 + [1] * 50),
        z_cal=np.vstack([x0[50:], x1[50:]]),
        y_cal=np.array([0] * 20 + [1] * 20),
    )
    scores = detector.id_scores(np.vstack([x0[50:55], x1[50:55]]), delta=1.345, calibrated=True)
    assert scores.shape == (10,)
    assert np.isfinite(scores).all()
