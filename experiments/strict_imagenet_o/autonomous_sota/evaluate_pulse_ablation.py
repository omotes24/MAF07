#!/usr/bin/env python3
"""Frozen leave-one-component-out ablation for PULSE."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from audit_symmetric_support_proxy import metric_row
from evaluate_activation_shift import DATASETS, load_shift
from methods.pulse import PulseState
from reproduce_nnguide import assert_no_final_access, expand


COMPONENTS = ("shift", "uniform", "localized", "knn")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--feature", action="append", required=True, help="NAME=glob")
    parser.add_argument("--bilateral-dir", type=Path, required=True)
    parser.add_argument("--localized-dir", type=Path, required=True)
    parser.add_argument("--compact-knn-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load(path: Path, key: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        return data[key].astype(np.float64)


def formulas(component: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    output = {"PULSE": sum(component.values())}
    for removed in COMPONENTS:
        output[f"PULSE-minus-{removed}"] = sum(
            score for name, score in component.items() if name != removed
        )
    output.update({f"PULSE-only-{name}": score for name, score in component.items()})
    return output


def main() -> None:
    args = parse_args()
    specs = dict(item.split("=", 1) for item in args.feature)
    paths = {name: expand([pattern]) for name, pattern in specs.items()}
    if set(paths) != {"id", *DATASETS}:
        raise ValueError(f"expected id and {DATASETS}, got {sorted(paths)}")
    assert_no_final_access(
        [
            *(path for group in paths.values() for path in group),
            args.state,
            args.bilateral_dir,
            args.localized_dir,
            args.compact_knn_dir,
            args.output,
        ]
    )
    state = PulseState.load(args.state)
    scores = {}
    for dataset in ("id", *DATASETS):
        component = state.components(
            load_shift(paths[dataset])["shift_ratio"],
            load(args.bilateral_dir / f"{dataset}.npz", "uniform_cosine"),
            load(args.localized_dir / f"{dataset}.npz", "prototype_max").max(axis=1),
            load(args.compact_knn_dir / f"{dataset}.npz", "mean_feature"),
        )
        scores[dataset] = formulas(component)
    rows = []
    for method in scores["id"]:
        for dataset in DATASETS:
            rows.append(
                {
                    "method": method,
                    "dataset": dataset,
                    **metric_row(-scores["id"][method], -scores[dataset][method]),
                }
            )
    per_dataset = pd.DataFrame(rows)
    summary = per_dataset.groupby("method", as_index=False).agg(
        Macro_AUROC=("AUROC", "mean"),
        Macro_FPR95=("FPR95", "mean"),
        Macro_AUPR_OUT=("AUPR_OUT", "mean"),
    )
    full = summary.set_index("method").loc["PULSE"]
    summary["delta_AUROC_vs_full"] = summary["Macro_AUROC"] - full["Macro_AUROC"]
    summary["delta_FPR95_vs_full"] = summary["Macro_FPR95"] - full["Macro_FPR95"]
    args.output.mkdir(parents=True, exist_ok=True)
    per_dataset.to_csv(args.output / "per_dataset.csv", index=False)
    summary.to_csv(args.output / "summary.csv", index=False)
    (args.output / "provenance.json").write_text(
        json.dumps(
            {
                "state": str(args.state),
                "state_refit_per_ablation": False,
                "target_ood_used_for_fit_or_selection": False,
                "final_benchmark_accessed": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(summary.sort_values("Macro_AUROC", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
