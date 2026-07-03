from __future__ import annotations

import json
from pathlib import Path

from .config import resolve_path
from .data import verify_dataset
from .jobs import audit_coverage


def write_dataset_audit(
    dataset_config: str | Path = "configs/dataset.yaml",
    output_path: str | Path = "results/logs/dataset_audit.json",
) -> dict[str, object]:
    report = verify_dataset(dataset_config)
    out = resolve_path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def write_coverage_audit(strict: bool = False) -> dict[str, object]:
    return audit_coverage(strict=strict)

