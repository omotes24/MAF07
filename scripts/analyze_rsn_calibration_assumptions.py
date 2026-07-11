#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from maf07.config import resolve_path
from maf07.metrics import ood_metrics


RAW = "rsn_class_std_huber_k150"
CALIBRATED = "rsn_class_std_huber_calibrated"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-samples", required=True)
    parser.add_argument(
        "--split", default="results/splits_content_cleaned_conservative/splits_seed0.csv"
    )
    parser.add_argument("--id-set", default="cheetah|jaguar")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output = resolve_path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    samples = pd.read_csv(resolve_path(args.score_samples))
    block = samples[samples["method"].isin([RAW, CALIBRATED])].copy()
    pivot = block.pivot(
        index=["image_id", "class_name", "role", "ood_label"],
        columns="method",
        values="score",
    ).dropna()
    if RAW not in pivot or CALIBRATED not in pivot:
        raise SystemExit("Score sample file does not contain raw and calibrated RSN")

    raw = pivot[RAW].to_numpy()
    calibrated = pivot[CALIBRATED].to_numpy()
    tau = kendalltau(raw, calibrated)
    rho = spearmanr(raw, calibrated)
    raw_rank = pd.Series(raw).rank(method="average").to_numpy()
    calibrated_rank = pd.Series(calibrated).rank(method="average").to_numpy()
    displacement = np.abs(raw_rank - calibrated_rank)
    y_is_id = (pivot.index.get_level_values("ood_label").to_numpy() == 0).astype(int)
    raw_metrics = ood_metrics(y_is_id, raw)
    calibrated_metrics = ood_metrics(y_is_id, calibrated)

    order = np.argsort(raw)
    raw_margin = np.diff(raw[order])
    calibrated_difference = np.diff(calibrated[order])
    adjacent = pd.DataFrame(
        {
            "raw_margin": raw_margin,
            "calibrated_reversal": calibrated_difference < 0,
        }
    )
    bin_count = min(10, int(adjacent["raw_margin"].nunique()))
    if bin_count >= 1:
        adjacent["margin_bin"] = pd.qcut(
            adjacent["raw_margin"], q=bin_count, duplicates="drop"
        ).astype(str)
        margin_summary = (
            adjacent.groupby("margin_bin", observed=True)
            .agg(
                n=("calibrated_reversal", "size"),
                raw_margin_mean=("raw_margin", "mean"),
                reversal_rate=("calibrated_reversal", "mean"),
            )
            .reset_index()
        )
    else:
        margin_summary = pd.DataFrame(
            columns=["margin_bin", "n", "raw_margin_mean", "reversal_rate"]
        )
    margin_summary.to_csv(output / "score_margin_rank_reversal.csv", index=False)

    split = pd.read_csv(resolve_path(args.split))
    id_classes = args.id_set.split("|")
    val_counts = (
        split[split["class_name"].isin(id_classes) & split["split"].eq("val")]
        .groupby("class_name")
        .size()
    )
    class_count = int(len(id_classes))
    n_min = int(val_counts.min())
    epsilon = np.linspace(0.01, 0.20, 20)
    union = np.minimum(1.0, 2.0 * class_count * np.exp(-2.0 * n_min * epsilon**2))
    pd.DataFrame(
        {
            "epsilon": epsilon,
            "class_count": class_count,
            "min_validation_n": n_min,
            "dkw_union_bound": union,
        }
    ).to_csv(output / "dkw_union_bound.csv", index=False)

    report = {
        "sample_n": int(len(pivot)),
        "id_set": id_classes,
        "validation_counts": {str(k): int(v) for k, v in val_counts.items()},
        "spearman_rho": float(rho.statistic),
        "kendall_tau": float(tau.statistic),
        "discordant_pair_fraction_from_tau": float((1.0 - tau.statistic) / 2.0),
        "mean_absolute_rank_displacement": float(displacement.mean()),
        "max_absolute_rank_displacement": float(displacement.max()),
        "raw_metrics": {key: float(raw_metrics[key]) for key in ("AUROC", "FPR95", "AUPR_OUT")},
        "calibrated_metrics": {
            key: float(calibrated_metrics[key]) for key in ("AUROC", "FPR95", "AUPR_OUT")
        },
        "claim_scope": (
            "Empirical rank-equivalence diagnostic only. DKW controls CDF error; "
            "it does not by itself imply AUROC degradation."
        ),
    }
    (output / "calibration_assumption_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
