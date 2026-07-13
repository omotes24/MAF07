from __future__ import annotations

import numpy as np

from methods.atomic_components import class_empirical_cdf


def test_class_empirical_cdf_uses_candidate_class_reference() -> None:
    references = np.asarray(
        [
            [1.0, 2.0, 3.0],
            [10.0, 20.0, 30.0],
        ],
        dtype=np.float32,
    )
    values = np.asarray([[2.5, 25.0]], dtype=np.float32)
    classes = np.asarray([[0, 1]], dtype=np.int64)
    actual = class_empirical_cdf(values, classes, references)
    np.testing.assert_allclose(actual, [[0.75, 0.75]])
