#!/usr/bin/env python3
"""Fit and hash the PULSE scalar state from independent ID validation only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from methods.pulse import fit_pulse_state
from reproduce_nnguide import assert_no_final_access


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--activation-state", type=Path, required=True)
    parser.add_argument("--bilateral-validation", type=Path, required=True)
    parser.add_argument("--localized-validation", type=Path, required=True)
    parser.add_argument("--compact-knn-validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load(path: Path, key: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        return data[key].astype(np.float64)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    args = parse_args()
    inputs = (
        args.activation_state,
        args.bilateral_validation,
        args.localized_validation,
        args.compact_knn_validation,
    )
    assert_no_final_access([*inputs, args.output])
    with np.load(args.activation_state, allow_pickle=False) as data:
        shift_center = float(data["shift_center"])
        shift_scale = float(data["shift_scale"])
    localized = load(args.localized_validation, "prototype_max").max(axis=1)
    state = fit_pulse_state(
        shift_center=shift_center,
        shift_scale=shift_scale,
        uniform_cosine=load(args.bilateral_validation, "uniform_cosine"),
        localized_prototype_confidence=localized,
        compact_knn_confidence=load(args.compact_knn_validation, "mean_feature"),
    )
    state.save(args.output)
    metadata = {
        "method": "PULSE",
        "fit_data": "independent ImageNet ID validation only",
        "target_ood_used_for_fit_or_selection": False,
        "final_benchmark_accessed": False,
        "score_components": 4,
        "learned_component_weights": False,
        "input_sha256": {str(path): sha256(path) for path in inputs},
        "state_sha256": sha256(args.output),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
