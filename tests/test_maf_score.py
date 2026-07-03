from __future__ import annotations

import numpy as np

from maf07.methods.maf import MAFScorer, maf_fusion
from maf07.metrics import fpr95, ood_metrics


def test_maf_score_shape_and_orientation() -> None:
    rng = np.random.default_rng(0)
    x0 = rng.normal(loc=-2.0, scale=0.4, size=(40, 4))
    x1 = rng.normal(loc=2.0, scale=0.4, size=(40, 4))
    train_x = np.vstack([x0, x1])
    train_y = np.array([0] * len(x0) + [1] * len(x1))
    id_x = np.vstack([
        rng.normal(loc=-2.0, scale=0.4, size=(20, 4)),
        rng.normal(loc=2.0, scale=0.4, size=(20, 4)),
    ])
    ood_x = rng.normal(loc=0.0, scale=4.0, size=(40, 4))
    scorer = MAFScorer().fit(train_x, train_y)
    id_scores = scorer.score(id_x)
    ood_scores = scorer.score(ood_x)
    assert id_scores.shape == (40,)
    assert float(np.mean(id_scores)) > float(np.mean(ood_scores))


def test_product_and_sqrt_product_have_same_auroc_fpr95() -> None:
    rng = np.random.default_rng(1)
    s_conf = rng.uniform(0.1, 1.0, size=100)
    s_cons = rng.uniform(0.1, 1.0, size=100)
    labels = np.array([1] * 50 + [0] * 50)
    product = maf_fusion(s_conf, s_cons, mode="product")
    sqrt_product = maf_fusion(s_conf, s_cons, mode="sqrt_product")
    m1 = ood_metrics(labels, product)
    m2 = ood_metrics(labels, sqrt_product)
    assert m1["AUROC"] == m2["AUROC"]
    assert fpr95(product[labels == 1], product[labels == 0]) == fpr95(
        sqrt_product[labels == 1], sqrt_product[labels == 0]
    )

