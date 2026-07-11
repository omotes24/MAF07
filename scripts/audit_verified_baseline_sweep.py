#!/usr/bin/env python
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from maf07.config import load_yaml, resolve_path

from run_verified_baseline_audit import DEFAULT_METHODS, _parse_values


KEY_COLUMNS = ["id_size", "backbone", "seed", "id_set", "method"]
METRIC_COLUMNS = ["AUROC", "FPR95", "AUPR_OUT"]


def _expected_frame(
    id_sizes: list[int],
    backbones: list[str],
    seeds: list[int],
    methods: list[str],
) -> pd.DataFrame:
    classes = list(load_yaml("configs/dataset.yaml")["classes"])
    rows = []
    for id_size in id_sizes:
        for backbone, seed, id_set, method in itertools.product(
            backbones,
            seeds,
            itertools.combinations(classes, id_size),
            methods,
        ):
            rows.append(
                {
                    "id_size": int(id_size),
                    "backbone": backbone,
                    "seed": int(seed),
                    "id_set": "|".join(id_set),
                    "method": method,
                }
            )
    return pd.DataFrame(rows, columns=KEY_COLUMNS)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--id-sizes", default="2,3,4,5,6,7")
    parser.add_argument("--methods", default=DEFAULT_METHODS)
    parser.add_argument("--backbones", default="dinov2_vitb14,dinov2_vitl14")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--report", required=True)
    parser.add_argument("--missing-output", required=True)
    args = parser.parse_args(argv)

    input_path = resolve_path(args.input)
    report_path = resolve_path(args.report)
    missing_path = resolve_path(args.missing_output)
    id_sizes = [int(value) for value in _parse_values(args.id_sizes)]
    methods = [value.lower() for value in _parse_values(args.methods)]
    backbones = _parse_values(args.backbones)
    seeds = [int(value) for value in _parse_values(args.seeds)]

    expected = _expected_frame(id_sizes, backbones, seeds, methods)
    if not input_path.exists() or input_path.stat().st_size == 0:
        actual = pd.DataFrame(columns=KEY_COLUMNS + METRIC_COLUMNS)
    else:
        actual = pd.read_csv(input_path)

    missing_columns = sorted(set(KEY_COLUMNS + METRIC_COLUMNS) - set(actual.columns))
    if missing_columns:
        raise SystemExit(f"Input is missing required columns: {missing_columns}")

    actual = actual.copy()
    actual["id_size"] = pd.to_numeric(actual["id_size"], errors="raise").astype(int)
    actual["seed"] = pd.to_numeric(actual["seed"], errors="raise").astype(int)
    actual["method"] = actual["method"].astype(str).str.lower()
    target = actual[
        actual["id_size"].isin(id_sizes)
        & actual["backbone"].isin(backbones)
        & actual["seed"].isin(seeds)
        & actual["method"].isin(methods)
    ].copy()

    duplicate_rows = int(target.duplicated(KEY_COLUMNS, keep=False).sum())
    unique_actual = target.drop_duplicates(KEY_COLUMNS, keep="last")
    merged = expected.merge(unique_actual[KEY_COLUMNS], on=KEY_COLUMNS, how="left", indicator=True)
    missing = merged.loc[merged["_merge"] == "left_only", KEY_COLUMNS].copy()

    expected_keys = set(expected.itertuples(index=False, name=None))
    actual_keys = set(unique_actual[KEY_COLUMNS].itertuples(index=False, name=None))
    extra_keys = actual_keys - expected_keys
    nonfinite_rows = int(
        (~np.isfinite(unique_actual[METRIC_COLUMNS].to_numpy(dtype=np.float64))).any(axis=1).sum()
    )

    per_id_size = []
    for id_size in id_sizes:
        expected_n = int((expected["id_size"] == id_size).sum())
        completed_n = int(
            len(
                set(
                    unique_actual.loc[
                        unique_actual["id_size"] == id_size,
                        KEY_COLUMNS,
                    ].itertuples(index=False, name=None)
                )
                & expected_keys
            )
        )
        per_id_size.append(
            {
                "id_size": id_size,
                "folds": math.comb(8, id_size) * len(seeds) * len(backbones),
                "expected_jobs": expected_n,
                "completed_jobs": completed_n,
                "missing_jobs": expected_n - completed_n,
            }
        )

    completed_jobs = len(expected_keys & actual_keys)
    report = {
        "complete": bool(
            completed_jobs == len(expected)
            and missing.empty
            and duplicate_rows == 0
            and not extra_keys
            and nonfinite_rows == 0
        ),
        "expected_jobs": int(len(expected)),
        "completed_jobs": int(completed_jobs),
        "missing_jobs": int(len(missing)),
        "duplicate_rows": duplicate_rows,
        "extra_jobs": int(len(extra_keys)),
        "nonfinite_rows": nonfinite_rows,
        "id_sizes": id_sizes,
        "methods": methods,
        "backbones": backbones,
        "seeds": seeds,
        "per_id_size": per_id_size,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    missing_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    missing.to_csv(missing_path, index=False)
    print(json.dumps(report, sort_keys=True))
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
