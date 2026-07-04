from __future__ import annotations

import numpy as np

from maf07.methods.baselines_logit import gradnorm, kl_matching
from maf07.metrics import fpr95, logits_to_scores, ood_metrics, oscr


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
    assert gradnorm(logits).shape == (3,)
    train = {
        0: np.array([-2.0, -1.0, -0.5]),
        1: np.array([0.5, 1.0, 2.0]),
    }
    assert kl_matching(logits, train).shape == (3,)


def test_oscr_uses_supported_trapezoid_integration() -> None:
    id_scores = np.array([0.9, 0.8, 0.7])
    ood_scores = np.array([0.3, 0.2, 0.1])
    assert oscr(id_scores, ood_scores) >= 0.0
