from __future__ import annotations

import numpy as np

from maf07.metrics import fpr95, ood_metrics


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

