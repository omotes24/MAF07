import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from final_lock import sha256_file, sha256_model_state, verify_locked_config


def write_lock(tmp_path: Path) -> tuple[Path, Path]:
    code_root = tmp_path / "code"
    code_root.mkdir()
    code = code_root / "method.py"
    code.write_text("VALUE = 1\n", encoding="utf-8")
    artifact = tmp_path / "state.npz"
    np.savez(artifact, value=np.asarray([1.0]))
    config = tmp_path / "locked.json"
    config.write_text(
        json.dumps(
            {
                "lock_status": "locked_before_final_access",
                "final_evaluation_executed": False,
                "code_root": str(code_root),
                "code_sha256": {"method.py": sha256_file(code)},
                "artifacts": {
                    "state": {"path": str(artifact), "sha256": sha256_file(artifact)}
                },
                "method": {
                    "name": "PULSE",
                    "constraints": {
                        "major_score_components": 4,
                        "free_hyperparameters": 3,
                        "target_ood_fit_or_calibration": False,
                        "test_image_sharing": False,
                    },
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    config.with_suffix(".sha256").write_text(
        f"{sha256_file(config)}  {config.name}\n", encoding="utf-8"
    )
    return config, artifact


def test_locked_config_verifies_all_hashes(tmp_path: Path) -> None:
    config, _ = write_lock(tmp_path)
    payload, observed = verify_locked_config(config)
    assert payload["method"]["name"] == "PULSE"
    assert observed == sha256_file(config)


def test_locked_config_rejects_artifact_tampering(tmp_path: Path) -> None:
    config, artifact = write_lock(tmp_path)
    artifact.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="artifact state changed"):
        verify_locked_config(config)


def test_locked_config_rejects_code_tampering(tmp_path: Path) -> None:
    config, _ = write_lock(tmp_path)
    (tmp_path / "code" / "method.py").write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="locked code changed"):
        verify_locked_config(config)


def test_model_state_hash_is_order_stable_and_value_sensitive() -> None:
    first = torch.nn.Linear(3, 2)
    second = torch.nn.Linear(3, 2)
    second.load_state_dict(first.state_dict())
    assert sha256_model_state(first) == sha256_model_state(second)
    with torch.no_grad():
        second.weight[0, 0] += 1
    assert sha256_model_state(first) != sha256_model_state(second)
