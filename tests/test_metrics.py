from __future__ import annotations

import numpy as np

from maf07.methods.baselines_logit import gradnorm, kl_matching
from maf07.methods.verified_baselines import fit_ridge_linear_probe
from maf07.metrics import fpr95, logits_to_scores, ood_metrics, oscr
from maf07.runner import _torch_ridge_logits


def test_fpr95_definition() -> None:
    id_scores = np.array([0.9] * 95 + [0.1] * 5)
    ood_scores = np.array([0.2] * 3 + [0.05] * 7)
    assert fpr95(id_scores, ood_scores) == 0.3


def test_ood_metrics_higher_is_id_like() -> None:
    labels = np.array([1, 1, 1, 0, 0, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
    metrics = ood_metrics(labels, scores)
    assert metrics["AUROC"] == 1.0
    assert metrics["FPR95"] == 0.0


def test_binary_logits_are_promoted_to_two_columns() -> None:
    logits = np.array([-2.0, 0.0, 2.0])
    assert logits_to_scores(logits, "msp").shape == (3,)
    assert logits_to_scores(logits, "odin").shape == (3,)
    features = np.array([[1.0, -1.0], [0.5, 0.25], [-0.2, 0.8]])
    assert gradnorm(logits, features).shape == (3,)
    train = {
        0: np.array([-2.0, -1.0, -0.5]),
        1: np.array([0.5, 1.0, 2.0]),
    }
    assert kl_matching(logits, train).shape == (3,)


def test_gen_score_is_higher_for_confident_id_like_logits() -> None:
    confident = np.array([[10.0, -10.0]])
    uncertain = np.array([[0.0, 0.0]])
    scores = logits_to_scores(np.vstack([confident, uncertain]), "gen")
    assert scores[0] > scores[1]


def test_oscr_uses_supported_trapezoid_integration() -> None:
    id_scores = np.array([0.9, 0.8, 0.7])
    ood_scores = np.array([0.3, 0.2, 0.1])
    assert oscr(id_scores, ood_scores) >= 0.0


def test_torch_ridge_logits_shape(monkeypatch) -> None:
    monkeypatch.setenv("MAF07_TORCH_DEVICE", "cpu")
    train_x = np.array([[1, 0], [0, 1], [1, 1], [-1, 0], [0, -1], [-1, -1]], dtype=float)
    train_y = np.array([0, 0, 0, 1, 1, 1])
    test_x = np.array([[0.5, 0.5], [-0.5, -0.5]], dtype=float)
    logits, by_class = _torch_ridge_logits(train_x, train_y, test_x)
    assert logits.shape == (2, 2)
    assert sorted(by_class) == [0, 1]
    probe = fit_ridge_linear_probe(train_x, train_y, device="cpu")
    np.testing.assert_allclose(probe.logits(test_x), logits, rtol=1e-6, atol=1e-6)
