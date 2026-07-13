from pathlib import Path

import numpy as np

from methods.pulse import PulseState, fit_pulse_state


def test_pulse_formula_and_orientation() -> None:
    state = PulseState(0, 1, 0, 1, 0, 1, 0, 1)
    score = state.score(
        np.array([0.0, np.e - 1.0]),
        np.array([0.0, 1.0]),
        np.array([1.0, 0.0]),
        np.array([1.0, 0.0]),
    )
    assert np.allclose(score, [-2.0, 2.0])
    assert score[1] > score[0]


def test_fit_uses_higher_is_id_confidences() -> None:
    state = fit_pulse_state(
        shift_center=1.0,
        shift_scale=2.0,
        uniform_cosine=np.array([0.0, 1.0, 2.0]),
        localized_prototype_confidence=np.array([2.0, 3.0, 4.0]),
        compact_knn_confidence=np.array([5.0, 6.0, 7.0]),
    )
    assert state.localized_center < 0
    assert state.knn_center < 0


def test_state_round_trip(tmp_path: Path) -> None:
    state = PulseState(1, 2, 3, 4, 5, 6, 7, 8)
    path = tmp_path / "state.npz"
    state.save(path)
    assert PulseState.load(path) == state
