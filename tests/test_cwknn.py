from __future__ import annotations

import numpy as np
import pytest

from maf07.methods.cwknn import CWKNNMeanDetector, cwknn_mean_from_env
from maf07.methods.rsn_factorial import FactorSpec, FactorialNeighborSuite
from maf07.runner import _score_method


def _data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def _brute_force(
    train: np.ndarray,
    labels: np.ndarray,
    query: np.ndarray,
    k: int,
) -> np.ndarray:
    columns = []
    for class_id in sorted(np.unique(labels)):
        bank = train[labels == class_id]
        distances = np.linalg.norm(query[:, None, :] - bank[None, :, :], axis=2)
        columns.append(np.sort(distances, axis=1)[:, : min(k, len(bank))].mean(axis=1))
    return np.column_stack(columns)


def test_cwknn_matches_definition() -> None:
    train, labels, query = _data()
    detector = CWKNNMeanDetector(k=2, device="cpu", score_batch=2).fit(train, labels)
    classes, distances = detector.class_distances(query)
    expected = _brute_force(train, labels, query, k=2)

    np.testing.assert_array_equal(classes, [0, 1])
    np.testing.assert_allclose(distances, expected, atol=1e-5)
    np.testing.assert_allclose(detector.ood_scores(query), expected.min(axis=1), atol=1e-5)
    np.testing.assert_allclose(detector.id_scores(query), -expected.min(axis=1), atol=1e-5)
    np.testing.assert_array_equal(detector.predict(query), [0, 1, 0])


def test_cwknn_matches_factorial_result_condition() -> None:
    train, labels, query = _data()
    dedicated = CWKNNMeanDetector(k=2, device="cpu").fit(train, labels).id_scores(query)
    spec = FactorSpec(
        "knn_raw_classwise_k150_mean",
        normalize=False,
        scale="none",
        aggregation="euclidean",
        bank_mode="classwise",
        neighbor_reduce="mean",
        k=2,
    )
    factorial = FactorialNeighborSuite(train, labels, device="cpu").score_many(
        query, [spec]
    )[spec.name]
    np.testing.assert_allclose(dedicated, factorial, atol=1e-5)


def test_cwknn_candidates_and_small_banks() -> None:
    train, labels, query = _data()
    detector = CWKNNMeanDetector(k=150, device="cpu").fit(train, labels)
    classes, distances = detector.class_distances(query, candidate_classes=[1])
    expected = _brute_force(train, labels, query, k=150)[:, 1:2]
    np.testing.assert_array_equal(classes, [1])
    np.testing.assert_allclose(distances, expected, atol=1e-5)
    np.testing.assert_array_equal(detector.predict(query, candidate_classes=[1]), [1, 1, 1])


def test_cwknn_validation_and_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    train, labels, query = _data()
    with pytest.raises(ValueError, match="positive"):
        CWKNNMeanDetector(k=0)
    with pytest.raises(RuntimeError, match="not fitted"):
        CWKNNMeanDetector(device="cpu").id_scores(query)

    detector = CWKNNMeanDetector(device="cpu").fit(train, labels)
    with pytest.raises(ValueError, match="Unknown candidate"):
        detector.id_scores(query, candidate_classes=[2])
    with pytest.raises(ValueError, match="dimension"):
        detector.id_scores(np.zeros((1, 3), dtype=np.float32))

    monkeypatch.setenv("MAF07_CWKNN_K", "17")
    monkeypatch.setenv("MAF07_CWKNN_DEVICE", "cpu")
    monkeypatch.setenv("MAF07_CWKNN_SCORE_BATCH", "11")
    configured = cwknn_mean_from_env()
    assert configured.k == 17
    assert configured.device == "cpu"
    assert configured.score_batch == 11


def test_cwknn_runner_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    train, labels, query = _data()
    monkeypatch.setenv("MAF07_CWKNN_K", "2")
    monkeypatch.setenv("MAF07_CWKNN_DEVICE", "cpu")
    score = _score_method(
        "cwknn_mean",
        train,
        labels,
        np.empty((0, train.shape[1]), dtype=np.float32),
        np.empty(0, dtype=int),
        query,
    )
    expected = CWKNNMeanDetector(k=2, device="cpu").fit(train, labels).id_scores(query)
    np.testing.assert_allclose(score, expected, atol=1e-5)
