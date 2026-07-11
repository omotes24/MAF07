from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

from maf07.methods.rsn_factorial import (
    FactorSpec,
    FactorialNeighborSuite,
    nnguide_score,
    reviewer_factor_specs,
)


def _toy_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [10.0, 10.0],
            [11.0, 10.0],
            [10.0, 11.0],
        ],
        dtype=np.float32,
    )
    labels = np.asarray([0, 0, 0, 1, 1, 1])
    query = np.asarray([[0.5, 0.0], [10.5, 10.0], [5.0, 5.0]], dtype=np.float32)
    return train, labels, query


def test_reviewer_grid_contains_requested_factor_controls() -> None:
    specs = reviewer_factor_specs()
    names = {spec.name for spec in specs}
    assert len(names) == len(specs)
    assert {
        "knn_raw_pooled_k150_kth",
        "raw_pooled_huber_k150",
        "raw_classwise_huber_k150",
        "global_std_huber_k150",
        "class_std_squared_k150",
        "rsn_class_std_huber_k150",
        "class_mad_huber_k150",
        "class_iqr_huber_k150",
        "rsn_k50",
        "rsn_delta200",
        "rsn_tau1e2",
    }.issubset(names)


def test_pooled_raw_kth_matches_bruteforce() -> None:
    train, labels, query = _toy_data()
    spec = FactorSpec(
        "pooled",
        scale="none",
        aggregation="euclidean",
        bank_mode="pooled",
        neighbor_reduce="kth",
        k=2,
    )
    score = FactorialNeighborSuite(train, labels, device="cpu").score_many(query, [spec])[
        "pooled"
    ]
    distances = np.linalg.norm(query[:, None, :] - train[None, :, :], axis=2)
    expected = -np.sort(distances, axis=1)[:, 1]
    np.testing.assert_allclose(score, expected, atol=1e-5)


def test_class_scales_and_huber_scores_are_finite() -> None:
    train, labels, query = _toy_data()
    specs = [
        FactorSpec("std", scale="class_std", aggregation="huber", k=2),
        FactorSpec("mad", scale="class_mad", aggregation="huber", k=2),
        FactorSpec("iqr", scale="class_iqr", aggregation="huber", k=2),
        FactorSpec("sq", scale="class_std", aggregation="squared", k=2),
    ]
    scores = FactorialNeighborSuite(train, labels, device="cpu").score_many(query, specs)
    assert set(scores) == {spec.name for spec in specs}
    assert all(np.isfinite(value).all() for value in scores.values())
    assert scores["std"][0] > scores["std"][2]
    assert scores["std"][1] > scores["std"][2]


def test_nnguide_matches_released_formula() -> None:
    train, _, query = _toy_data()
    train_logits = np.asarray(
        [[2.0, 0.0], [1.5, 0.0], [1.0, 0.0], [0.0, 2.0], [0.0, 1.5], [0.0, 1.0]],
        dtype=np.float32,
    )
    query_logits = np.asarray([[1.5, 0.0], [0.0, 1.5], [0.2, 0.2]], dtype=np.float32)
    score = nnguide_score(
        train,
        train_logits,
        query,
        query_logits,
        k=2,
        batch_size=2,
        device="cpu",
    )
    train_energy = logsumexp(train_logits, axis=1)
    query_energy = logsumexp(query_logits, axis=1)
    similarity = query @ (train * train_energy[:, None]).T
    expected = np.sort(similarity, axis=1)[:, -2:].mean(axis=1) * query_energy
    np.testing.assert_allclose(score, expected, rtol=1e-5, atol=1e-5)
