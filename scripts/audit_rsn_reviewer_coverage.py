#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from maf07.config import load_yaml, resolve_path
from maf07.methods.rsn_factorial import reviewer_factor_specs


SPECIAL_METHODS = [
    "nnguide_k10",
    "nnguide_k50",
    "rsn_class_std_huber_calibrated",
]
KEYS = ["backbone", "seed", "id_size", "id_set", "method"]


def _values(raw: str) -> list[str]:
    return [value.strip() for value in raw.replace(" ", ",").split(",") if value.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--per-ood-input", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--missing-output", required=True)
    parser.add_argument("--backbones", default="dinov2_vitb14,dinov2_vitl14")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--id-sizes", default="2,3,4,5,6,7")
    parser.add_argument("--methods", default="all")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    classes = list(load_yaml("configs/dataset.yaml").get("classes", []))
    backbones = _values(args.backbones)
    seeds = [int(value) for value in _values(args.seeds)]
    id_sizes = [int(value) for value in _values(args.id_sizes)]
    available = [spec.name for spec in reviewer_factor_specs()] + SPECIAL_METHODS
    methods = available if args.methods == "all" else _values(args.methods)
    invalid = sorted(set(methods) - set(available))
    if invalid:
        raise SystemExit(f"Unknown methods: {invalid}")

    expected_rows = []
    expected_per_ood = 0
    for backbone, seed, id_size, method in itertools.product(
        backbones, seeds, id_sizes, methods
    ):
        for id_set in itertools.combinations(classes, id_size):
            expected_rows.append(
                {
                    "backbone": backbone,
                    "seed": seed,
                    "id_size": id_size,
                    "id_set": "|".join(id_set),
                    "method": method,
                }
            )
            expected_per_ood += len(classes) - id_size
    expected = pd.DataFrame(expected_rows)
    result_path = resolve_path(args.input)
    completed_raw = pd.read_csv(result_path) if result_path.exists() else pd.DataFrame(columns=KEYS)
    if not completed_raw.empty:
        completed_raw = completed_raw[
            completed_raw["backbone"].isin(backbones)
            & completed_raw["seed"].isin(seeds)
            & completed_raw["id_size"].isin(id_sizes)
            & completed_raw["method"].isin(methods)
        ].copy()
    duplicate_jobs = int(completed_raw.duplicated(KEYS, keep=False).sum())
    completed = completed_raw.drop_duplicates(KEYS, keep="last")
    merged = expected.merge(completed[KEYS], on=KEYS, how="left", indicator=True)
    missing = merged[merged["_merge"].eq("left_only")][KEYS].copy()
    extra = completed.merge(expected, on=KEYS, how="left", indicator=True)
    extra_count = int(extra["_merge"].eq("left_only").sum())

    per_ood_path = resolve_path(args.per_ood_input)
    if per_ood_path.exists():
        per_ood_raw = pd.read_csv(per_ood_path)
        per_ood_raw = per_ood_raw[
            per_ood_raw["backbone"].isin(backbones)
            & per_ood_raw["seed"].isin(seeds)
            & per_ood_raw["id_size"].isin(id_sizes)
            & per_ood_raw["method"].isin(methods)
        ].copy()
        per_ood_unique = per_ood_raw.drop_duplicates([*KEYS, "ood_class"], keep="last")
        per_ood_rows = int(len(per_ood_unique))
        per_ood_duplicates = int(
            per_ood_raw.duplicated([*KEYS, "ood_class"], keep=False).sum()
        )
    else:
        per_ood_rows = 0
        per_ood_duplicates = 0

    missing_path = resolve_path(args.missing_output)
    missing_path.parent.mkdir(parents=True, exist_ok=True)
    missing.to_csv(missing_path, index=False)
    report = {
        "expected_jobs": int(len(expected)),
        "completed_jobs": int(len(expected) - len(missing)),
        "missing_jobs": int(len(missing)),
        "extra_jobs": extra_count,
        "duplicate_job_rows": duplicate_jobs,
        "expected_per_ood_rows": int(expected_per_ood),
        "completed_per_ood_rows": per_ood_rows,
        "duplicate_per_ood_rows": per_ood_duplicates,
        "method_count": len(methods),
        "strict_pass": bool(
            missing.empty
            and extra_count == 0
            and per_ood_rows == expected_per_ood
        ),
    }
    report_path = resolve_path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True))
    if args.strict and not report["strict_pass"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
