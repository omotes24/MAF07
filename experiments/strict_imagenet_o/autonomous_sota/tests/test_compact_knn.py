from __future__ import annotations

import numpy as np

from methods.compact_knn import normalize_rows


def test_normalize_rows_produces_unit_vectors() -> None:
    values = np.array([[3.0, 4.0], [1.0, -1.0]], dtype=np.float32)
    result = normalize_rows(values)
    np.testing.assert_allclose(np.linalg.norm(result, axis=1), 1.0, rtol=1e-6)


def test_normalize_rows_keeps_zero_vector_finite() -> None:
    result = normalize_rows(np.zeros((1, 4), dtype=np.float32))
    assert np.isfinite(result).all()
    np.testing.assert_array_equal(result, np.zeros((1, 4), dtype=np.float32))
