#!/usr/bin/env python3
"""Evaluate a frozen PULSE state without target-OOD fitting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from audit_symmetric_support_proxy import metric_row
from evaluate_activation_shift import DATASETS, load_rc, load_shift
from methods.pulse import PulseState
from reproduce_nnguide import assert_no_final_access, expand


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--feature", action="append", required=True, help="NAME=glob")
    parser.add_argument("--bilateral-dir", type=Path, required=True)
    parser.add_argument("--localized-dir", type=Path, required=True)
    parser.add_argument("--compact-knn-dir", type=Path, required=True)
    parser.add_argument("--legacy-atomic-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load(path: Path, key: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        return data[key].astype(np.float64)


def main() -> None:
    args = parse_args()
    specs = dict(item.split("=", 1) for item in args.feature)
    paths = {name: expand([pattern]) for name, pattern in specs.items()}
    if set(paths) != {"id", *DATASETS}:
        raise ValueError(f"expected id and {DATASETS}, got {sorted(paths)}")
    all_paths = [path for group in paths.values() for path in group]
    assert_no_final_access(
        all_paths
        + [
            args.state,
            args.bilateral_dir,
            args.localized_dir,
            args.compact_knn_dir,
            args.legacy_atomic_dir,
            args.output,
        ]
    )
    state = PulseState.load(args.state)
    scores, baseline = {}, {}
    args.output.mkdir(parents=True, exist_ok=True)
    for name in ("id", *DATASETS):
        shift = load_shift(paths[name])["shift_ratio"]
        uniform = load(args.bilateral_dir / f"{name}.npz", "uniform_cosine")
        localized = load(args.localized_dir / f"{name}.npz", "prototype_max").max(axis=1)
        knn = load(args.compact_knn_dir / f"{name}.npz", "mean_feature")
        scores[name] = state.score(shift, uniform, localized, knn)
        baseline[name] = load_rc(args.legacy_atomic_dir / f"{name}.npz").astype(np.float64)
        np.savez_compressed(
            args.output / f"{name}_scores.npz",
            pulse_ood_score=scores[name].astype(np.float32),
            rc_msps_confidence=baseline[name].astype(np.float32),
        )

    rows = []
    for dataset in DATASETS:
        rows.append(
            {
                "method": "PULSE",
                "dataset": dataset,
                **metric_row(-scores["id"], -scores[dataset]),
            }
        )
        rows.append(
            {
                "method": "RC-MSPS",
                "dataset": dataset,
                **metric_row(baseline["id"], baseline[dataset]),
            }
        )
    per_dataset = pd.DataFrame(rows)
    summary = per_dataset.groupby("method", as_index=False).agg(
        Macro_AUROC=("AUROC", "mean"),
        Macro_FPR95=("FPR95", "mean"),
        Macro_AUPR_OUT=("AUPR_OUT", "mean"),
    )
    per_dataset.to_csv(args.output / "per_dataset.csv", index=False)
    summary.to_csv(args.output / "summary.csv", index=False)
    (args.output / "provenance.json").write_text(
        json.dumps(
            {
                "state": str(args.state),
                "same_configuration_all_datasets": True,
                "test_image_sharing": False,
                "target_ood_used_for_fit_or_calibration": False,
                "final_benchmark_accessed": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(per_dataset.to_string(index=False))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
