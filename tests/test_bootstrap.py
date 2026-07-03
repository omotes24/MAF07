from __future__ import annotations

import numpy as np
import pandas as pd

from maf07.bootstrap import bootstrap_metric_ci, stratified_paired_sample_indices


def test_stratified_paired_sample_keeps_strata_counts() -> None:
    strata = np.array(["id:a"] * 5 + ["id:b"] * 3 + ["ood:c"] * 4)
    idx = stratified_paired_sample_indices(strata, np.random.default_rng(0))
    sampled = pd.Series(strata[idx]).value_counts().to_dict()
    assert sampled == {"id:a": 5, "ood:c": 4, "id:b": 3}


def test_bootstrap_returns_ci_rows() -> None:
    frame = pd.DataFrame(
        {
            "is_id": [1, 1, 1, 0, 0, 0],
            "class_name": ["a", "a", "b", "c", "c", "d"],
            "score": [0.9, 0.8, 0.7, 0.2, 0.1, 0.0],
        }
    )
    ci = bootstrap_metric_ci(frame, "score", n_boot=10, seed=0)
    assert {"metric", "mean", "ci_low", "ci_high", "n_boot"} <= set(ci.columns)
    assert "AUROC" in set(ci["metric"])

