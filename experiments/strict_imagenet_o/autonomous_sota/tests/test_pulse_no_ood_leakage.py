from pathlib import Path

import pytest

from reproduce_nnguide import assert_no_final_access


def test_prelock_guard_rejects_final_benchmarks() -> None:
    with pytest.raises(PermissionError):
        assert_no_final_access([Path("/cache/NINCO/features.npz")])
    with pytest.raises(PermissionError):
        assert_no_final_access([Path("/cache/ssb-hard/features.npz")])


def test_pulse_fit_entrypoint_has_no_ood_argument() -> None:
    source = (Path(__file__).parents[1] / "fit_pulse_state.py").read_text(encoding="utf-8")
    assert "--ood" not in source
    assert "DATASETS" not in source
    assert "target_ood_used_for_fit_or_selection\": False" in source
