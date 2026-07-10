from __future__ import annotations

import numpy as np
import pytest
from scipy.special import softmax

from maf07.methods.rsn import rsn_from_env, rsn_uses_topq
from maf07.methods.verified_baselines import (
    VERIFIED_METHODS,
    UnsupportedVerifiedBaseline,
    VerifiedBaselineSuite,
    _quadratic_distance,
    ash_features,
    gen_score,
    gradnorm_score,
)


def test_gen_matches_official_reference_formula() -> None:
    logits = np.array([[2.0, 0.5, -1.0], [0.1, 0.2, 0.3]])
    probs = softmax(logits, axis=1)
    expected = -np.sum(probs**0.1 * (1.0 - probs) ** 0.1, axis=1)
    np.testing.assert_allclose(gen_score(logits), expected)


def test_gradnorm_closed_form_matches_autograd() -> None:
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(13)
    features = rng.normal(size=(4, 5))
    weight = rng.normal(size=(3, 5))
    bias = rng.normal(size=3)
    logits = features @ weight.T + bias

    expected = []
    for row in features:
        x = torch.tensor(row, dtype=torch.float64)
        layer = torch.nn.Linear(5, 3, bias=True, dtype=torch.float64)
        with torch.no_grad():
            layer.weight.copy_(torch.tensor(weight, dtype=torch.float64))
            layer.bias.copy_(torch.tensor(bias, dtype=torch.float64))
        loss = -torch.log_softmax(layer(x), dim=0).sum()
        loss.backward()
        expected.append(float(layer.weight.grad.abs().sum()))
    np.testing.assert_allclose(gradnorm_score(features, logits), expected, rtol=1e-10, atol=1e-10)


def test_quadratic_distance_uses_off_diagonal_precision() -> None:
    features = np.array([[1.0, -1.0]])
    precision = np.array([[2.0, 1.5], [1.5, 2.0]])
    result = _quadratic_distance(features, np.zeros(2), precision)
    np.testing.assert_allclose(result, [1.0])
    assert result[0] != 4.0


def test_ash_variants_match_reference_operations() -> None:
    features = np.array([[1.0, 2.0, 3.0, 4.0]])
    np.testing.assert_allclose(ash_features(features, variant="p", percentile=50), [[0, 0, 3, 4]])
    np.testing.assert_allclose(ash_features(features, variant="b", percentile=50), [[0, 0, 5, 5]])
    expected_scale = np.exp(10.0 / 7.0)
    np.testing.assert_allclose(
        ash_features(features, variant="s", percentile=50),
        [[0, 0, 3 * expected_scale, 4 * expected_scale]],
    )


def test_verified_suite_produces_finite_scores(monkeypatch) -> None:
    monkeypatch.setenv("MAF07_TORCH_DEVICE", "cpu")
    rng = np.random.default_rng(17)
    train = np.vstack([rng.normal(loc=cls * 1.5, size=(20, 6)) for cls in range(3)])
    labels = np.repeat(np.arange(3), 20)
    test = rng.normal(size=(11, 6))
    suite = VerifiedBaselineSuite(train, labels)
    for method in sorted(VERIFIED_METHODS):
        scores = suite.score(method, test)
        assert scores.shape == (len(test),), method
        assert np.isfinite(scores).all(), method


def test_unsupported_dinov2_names_fail_closed(monkeypatch) -> None:
    monkeypatch.setenv("MAF07_TORCH_DEVICE", "cpu")
    train = np.array([[1.0, 0.0], [0.8, 0.1], [0.0, 1.0], [0.1, 0.8]])
    labels = np.array([0, 0, 1, 1])
    suite = VerifiedBaselineSuite(train, labels)
    with pytest.raises(UnsupportedVerifiedBaseline):
        suite.score("odin", train)
    with pytest.raises(UnsupportedVerifiedBaseline):
        suite.score("mcm", train)
    with pytest.raises(UnsupportedVerifiedBaseline):
        suite.score("mah_mindist", train)


def test_rsn_profiles_make_paper_and_reported_settings_explicit(monkeypatch) -> None:
    for name in [
        "MAF07_RSN_PROFILE",
        "MAF07_RSN_K",
        "MAF07_RSN_NORMALIZE",
        "MAF07_RSN_USE_TOPQ",
        "MAF07_DIAGCARD_K",
        "MAF07_DIAGCARD_NORMALIZE",
    ]:
        monkeypatch.delenv(name, raising=False)
    paper = rsn_from_env()
    assert paper.k == 10
    assert paper.normalize is True
    assert rsn_uses_topq() is True

    monkeypatch.setenv("MAF07_RSN_PROFILE", "reported_20260708")
    reported = rsn_from_env()
    assert reported.k == 150
    assert reported.normalize is False
    assert rsn_uses_topq() is False
