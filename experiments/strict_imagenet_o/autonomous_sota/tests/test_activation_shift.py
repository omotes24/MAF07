import numpy as np
import torch

from methods.activation_shift import (
    ShiftRCState,
    activation_shift_statistics,
    adascale_confidence,
    low_gradient_perturb,
)


def test_low_gradient_perturb_changes_only_smallest_gradient_coordinates():
    image = torch.zeros(1, 1, 1, 4)
    gradient = torch.tensor([[[[4.0, -1.0, 3.0, 2.0]]]])
    output = low_gradient_perturb(image, gradient, fraction=0.25, magnitude=0.5)
    assert torch.equal(output, torch.tensor([[[[0.0, -0.5, 0.0, 0.0]]]]))


def test_activation_shift_statistics_detects_larger_change():
    feature = torch.arange(1.0, 101.0).repeat(2, 1)
    perturbed = feature.clone()
    perturbed[0, -5:] -= 0.1
    perturbed[1, -5:] -= 2.0
    result = activation_shift_statistics(feature, perturbed)
    assert result["shift_ratio"][1] > result["shift_ratio"][0]
    assert result["qprime_eval"].shape == (2,)


def test_shift_rc_prefers_id_like_atoms():
    state = ShiftRCState(rc_center=-0.8, rc_scale=0.1, shift_center=0.0, shift_scale=0.1)
    typical = state.confidence(np.array([0.8]), np.array([0.0]))
    shifted = state.confidence(np.array([0.5]), np.array([0.5]))
    assert typical[0] > shifted[0]


def test_adascale_confidence_has_expected_shape_and_direction():
    features = np.ones((2, 20), dtype=np.float32)
    features[1] *= 2.0
    qprime = np.array([0.5, 0.5])
    reference = np.linspace(0.0, 1.0, 10)
    weight = np.ones((3, 20), dtype=np.float32)
    bias = np.zeros(3, dtype=np.float32)
    confidence = adascale_confidence(
        features, qprime, reference, weight, bias, logit_scaling=True
    )
    assert confidence.shape == (2,)
    assert confidence[1] > confidence[0]
