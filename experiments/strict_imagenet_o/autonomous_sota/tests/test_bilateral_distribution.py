import numpy as np

from methods.bilateral_distribution import COMPONENT_ORIENTATION, bilateral_components


def test_uniform_support_is_less_id_like_than_peaked_support():
    prototypes = np.eye(3, dtype=np.float32)
    uniform = np.array([[1.0, 1.0, 1.0]], dtype=np.float32)
    peaked = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    values = bilateral_components(np.concatenate([uniform, peaked]), prototypes, device="cpu")
    assert set(values) == set(COMPONENT_ORIENTATION)
    assert values["prototype_spread"][0] < values["prototype_spread"][1]
    assert values["top_background_contrast"][0] < values["top_background_contrast"][1]
    assert values["uniform_cosine"][0] > values["uniform_cosine"][1]
    assert all(value.shape == (2,) and np.isfinite(value).all() for value in values.values())


def test_invalid_shapes_are_rejected():
    try:
        bilateral_components(np.ones((2, 3)), np.ones((2, 4)), device="cpu")
    except ValueError:
        pass
    else:
        raise AssertionError("incompatible features and prototypes must be rejected")
