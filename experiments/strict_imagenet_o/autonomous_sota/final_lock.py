"""Integrity checks shared by the one-shot locked final evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_model_state(model: Any) -> str:
    """Hash model tensor names, metadata, and values in a stable order."""
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        metadata = json.dumps(
            {
                "dtype": str(value.dtype),
                "name": name,
                "shape": list(value.shape),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(metadata).to_bytes(8, "little"))
        digest.update(metadata)
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _verify_sidecar(config_path: Path) -> str:
    sidecar = config_path.with_suffix(".sha256")
    if not sidecar.is_file():
        raise FileNotFoundError(f"missing lock hash sidecar: {sidecar}")
    expected = sidecar.read_text(encoding="utf-8").split()[0]
    observed = sha256_file(config_path)
    if observed != expected:
        raise RuntimeError(
            f"locked config hash mismatch: expected={expected} observed={observed}"
        )
    return observed


def verify_locked_config(config_path: Path) -> tuple[dict[str, Any], str]:
    """Verify the immutable config and every code/data artifact it names."""
    config_path = config_path.resolve()
    config_sha = _verify_sidecar(config_path)
    lock = json.loads(config_path.read_text(encoding="utf-8"))
    if lock.get("lock_status") != "locked_before_final_access":
        raise RuntimeError("final evaluation requires a pre-access locked config")
    if lock.get("final_evaluation_executed") is not False:
        raise RuntimeError("locked config must record an untouched final evaluation")
    if lock.get("method", {}).get("name") != "PULSE":
        raise RuntimeError("unexpected locked method")
    constraints = lock.get("method", {}).get("constraints", {})
    if int(constraints.get("major_score_components", -1)) > 4:
        raise RuntimeError("locked method exceeds the component budget")
    if int(constraints.get("free_hyperparameters", -1)) > 3:
        raise RuntimeError("locked method exceeds the hyperparameter budget")
    if constraints.get("target_ood_fit_or_calibration") is not False:
        raise RuntimeError("target-OOD fitting is forbidden")
    if constraints.get("test_image_sharing") is not False:
        raise RuntimeError("test-image sharing is forbidden")

    for name, artifact in lock.get("artifacts", {}).items():
        path = Path(artifact["path"])
        if not path.is_file():
            raise FileNotFoundError(f"locked artifact {name} is missing: {path}")
        observed = sha256_file(path)
        if observed != artifact["sha256"]:
            raise RuntimeError(
                f"locked artifact {name} changed: expected={artifact['sha256']} "
                f"observed={observed}"
            )

    code_root = Path(lock["code_root"])
    for relative, expected in lock.get("code_sha256", {}).items():
        path = code_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"locked code is missing: {path}")
        observed = sha256_file(path)
        if observed != expected:
            raise RuntimeError(
                f"locked code changed: {relative} expected={expected} observed={observed}"
            )
    return lock, config_sha


def artifact_path(lock: dict[str, Any], name: str) -> Path:
    return Path(lock["artifacts"][name]["path"])
