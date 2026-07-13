import numpy as np

from audit_pulse_locked_proxies import (
    anchor_formulas,
    cauchy_envelope_formulas,
    cauchy_stage_confidence,
    proxy_transform,
    quantile_envelope_formulas,
    quantile_envelope_references,
)


def test_radial_extrapolation_uses_class_mean() -> None:
    values = np.array([[2.0, 4.0], [10.0, 20.0]], dtype=np.float32)
    labels = np.array([0, 1])
    means = np.array([[1.0, 2.0], [8.0, 16.0]], dtype=np.float32)
    actual = proxy_transform(values, labels, means, "radial_extrapolation")
    np.testing.assert_allclose(actual, means + 1.8 * (values - means))


def test_channel_mask_respects_stage_boundaries() -> None:
    values = np.ones((2, 6), dtype=np.float32)
    labels = np.array([0, 1])
    means = np.ones((2, 6), dtype=np.float32)
    actual = proxy_transform(values, labels, means, "channel_mask", stage_dims=(3, 3))
    expected = np.full((2, 6), 1.0 / 0.9, dtype=np.float32)
    expected[:, (0, 3)] = 0.0
    np.testing.assert_allclose(actual, expected)


def test_class_direction_uses_other_class_mean() -> None:
    values = np.array([[1.0, 1.0], [3.0, 3.0]], dtype=np.float32)
    labels = np.array([0, 1])
    means = np.array([[2.0, 2.0], [4.0, 4.0]], dtype=np.float32)
    actual = proxy_transform(values, labels, means, "class_direction_extrapolation")
    expected = np.array([[-0.5, -0.5], [4.5, 4.5]], dtype=np.float32)
    np.testing.assert_allclose(actual, expected)


def test_anchor_correction_requires_three_positive_evidences() -> None:
    scores = {
        "MSPS": np.array([-1.0, -1.0]),
        "_ood_shift": np.array([1.0, 1.0]),
        "_ood_uniform": np.array([2.0, 2.0]),
        "_ood_localized": np.array([0.0, 3.0]),
        "_ood_knn": np.array([4.0, 4.0]),
    }
    formulas = anchor_formulas(scores, msps_center=0.0, msps_scale=1.0)
    # Shift/uniform/localized consensus is zero until all three are positive.
    np.testing.assert_allclose(formulas["AnchorMin-SUL"], np.array([-1.0, -2.0]))
    np.testing.assert_allclose(formulas["AnchorHarmonic-SUL"][0], -1.0)


def test_cauchy_stage_confidence_is_monotone() -> None:
    scores = {
        f"_stage_support_{layer}": np.array([0.0, 2.0]) for layer in range(4)
    }
    references = tuple(np.array([-1.0, 0.0, 1.0]) for _ in range(4))
    confidence = cauchy_stage_confidence(scores, references)
    assert confidence[1] > confidence[0]


def test_envelope_selects_larger_ood_branch() -> None:
    scores = {
        "CauchyStageTail": np.array([-1.0, -3.0]),
        "_ood_shift": np.array([3.0, 0.0]),
        "_ood_uniform": np.array([3.0, 0.0]),
        "_ood_localized": np.array([3.0, 0.0]),
        "_ood_knn": np.array([3.0, 0.0]),
    }
    formulas = cauchy_envelope_formulas(scores, anchor_center=0.0, anchor_scale=1.0)
    expected = -np.maximum(np.array([1.0, 3.0]), np.array([np.sqrt(27.0), 0.0]))
    np.testing.assert_allclose(formulas["EnvelopeSum-SUL"], expected)


def test_quantile_envelope_aligns_branch_scales() -> None:
    fit = {
        "CauchyStageTail": np.array([3.0, 2.0, 1.0]),
        "_ood_shift": np.array([0.0, 1.0, 2.0]),
        "_ood_uniform": np.array([0.0, 1.0, 2.0]),
        "_ood_localized": np.array([0.0, 1.0, 2.0]),
        "_ood_knn": np.array([0.0, 1.0, 2.0]),
    }
    references = quantile_envelope_references(fit)
    scores = {name: value[-1:] for name, value in fit.items()}
    formulas = quantile_envelope_formulas(scores, references)
    assert formulas["QuantileEnvelope-SUL"][0] < -0.9
