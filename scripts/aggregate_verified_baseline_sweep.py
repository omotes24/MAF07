#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


KEY_COLUMNS = ["backbone", "seed", "id_size", "id_set_id", "id_set", "method"]
METRICS = ["AUROC", "FPR95", "AUPR_OUT"]
MAIN_RANKING_EXCLUDES = {"rsn_paper"}
DEFAULT_VERIFIED_METHODS = (
    "rsn_reported,rsn_paper,knn,mahalanobis,mahalanobispp,rmd,"
    "msp,entropy,energy,maxlogit,gen,gradnorm,kl_matching,vim,react,"
    "ashp,ashb,ashs,dice,scale,nci,openmax"
)


def _parse_ints(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.replace(" ", ",").split(",") if item.strip()]


def _parse_strings(raw: str) -> list[str]:
    return [item.strip().lower() for item in raw.replace(" ", ",").split(",") if item.strip()]


def _summarize(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_columns = ["backbone", "id_size", "method"]
    for keys, block in frame.groupby(group_columns, sort=True):
        rows.append(
            {
                "backbone": keys[0],
                "id_size": int(keys[1]),
                "method": keys[2],
                "n": int(len(block)),
                **{metric: float(block[metric].mean()) for metric in METRICS},
            }
        )
    for keys, block in frame.groupby(["id_size", "method"], sort=True):
        rows.append(
            {
                "backbone": "ALL",
                "id_size": int(keys[0]),
                "method": keys[1],
                "n": int(len(block)),
                **{metric: float(block[metric].mean()) for metric in METRICS},
            }
        )
    summary = pd.DataFrame(rows)
    for metric in METRICS:
        summary[metric] = summary[metric].round(6)
    return summary.sort_values(["id_size", "backbone", "method"]).reset_index(drop=True)


def _rankings(summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pooled = summary[
        (summary["backbone"] == "ALL")
        & ~summary["method"].isin(MAIN_RANKING_EXCLUDES)
    ].copy()
    ranked = []
    for id_size, block in pooled.groupby("id_size", sort=True):
        block = block.sort_values(
            ["AUROC", "FPR95", "AUPR_OUT", "method"],
            ascending=[False, True, False, True],
        ).copy()
        block.insert(0, "rank", np.arange(1, len(block) + 1))
        ranked.append(block)
    ranking = pd.concat(ranked, ignore_index=True)
    mean_ranking = (
        ranking.groupby("method", sort=False)
        .agg(
            id_sizes=("id_size", "nunique"),
            mean_rank=("rank", "mean"),
            mean_AUROC=("AUROC", "mean"),
            mean_FPR95=("FPR95", "mean"),
            mean_AUPR_OUT=("AUPR_OUT", "mean"),
        )
        .reset_index()
        .sort_values(["mean_rank", "mean_AUROC"], ascending=[True, False])
        .reset_index(drop=True)
    )
    mean_ranking.insert(0, "rank", np.arange(1, len(mean_ranking) + 1))
    for column in ["mean_rank", "mean_AUROC", "mean_FPR95", "mean_AUPR_OUT"]:
        mean_ranking[column] = mean_ranking[column].round(6)
    return ranking, mean_ranking


def _metric_rankings(summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pooled = summary[
        (summary["backbone"] == "ALL")
        & ~summary["method"].isin(MAIN_RANKING_EXCLUDES)
    ].copy()
    rows = []
    for id_size in sorted(pooled["id_size"].unique()):
        block = pooled[pooled["id_size"] == id_size]
        for metric in METRICS:
            ranked = block.sort_values(
                [metric, "method"],
                ascending=[metric == "FPR95", True],
            ).reset_index(drop=True)
            for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
                rows.append(
                    {
                        "id_size": int(id_size),
                        "metric": metric,
                        "rank": rank,
                        "method": row["method"],
                        "n": int(row["n"]),
                        "value": float(row[metric]),
                    }
                )
    metric_ranking = pd.DataFrame(rows)
    mean_metric_rank = (
        metric_ranking.groupby(["metric", "method"], sort=False)
        .agg(
            id_sizes=("id_size", "nunique"),
            mean_rank=("rank", "mean"),
            mean_value=("value", "mean"),
        )
        .reset_index()
        .sort_values(["metric", "mean_rank", "method"])
        .reset_index(drop=True)
    )
    mean_metric_rank["mean_rank"] = mean_metric_rank["mean_rank"].round(6)
    mean_metric_rank["mean_value"] = mean_metric_rank["mean_value"].round(6)
    return metric_ranking, mean_metric_rank


def _paired_rsn_vs_all(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["backbone", "seed", "id_size", "id_set_id"]
    methods = sorted(set(frame["method"]) - {"rsn_reported", "rsn_paper"})
    rows = []
    for id_size in sorted(frame["id_size"].unique()):
        for method in methods:
            block = frame[
                frame["id_size"].eq(id_size)
                & frame["method"].isin(["rsn_reported", method])
            ]
            pivot = block.pivot(index=keys, columns="method", values=METRICS)
            for metric in METRICS:
                values = pivot[[(metric, "rsn_reported"), (metric, method)]].dropna()
                values.columns = ["rsn", "baseline"]
                diff = values["rsn"].to_numpy() - values["baseline"].to_numpy()
                if metric == "FPR95":
                    diff = -diff
                n = len(diff)
                mean = float(diff.mean())
                half = float(stats.t.ppf(0.975, n - 1) * stats.sem(diff)) if n > 1 else np.nan
                rows.append(
                    {
                        "id_size": int(id_size),
                        "method_a": "rsn_reported",
                        "method_b": method,
                        "metric": metric,
                        "n": n,
                        "mean_diff": mean,
                        "ci_low": mean - half,
                        "ci_high": mean + half,
                        "good_rate": float((diff > 0).mean()),
                    }
                )
    paired = pd.DataFrame(rows)
    for column in ["mean_diff", "ci_low", "ci_high", "good_rate"]:
        paired[column] = paired[column].round(6)
    return paired


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verified-fold", required=True)
    parser.add_argument("--psm-fold", required=True)
    parser.add_argument("--id-sizes", default="2,3,4,5,6,7")
    parser.add_argument("--verified-methods", default=DEFAULT_VERIFIED_METHODS)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)

    verified = pd.read_csv(args.verified_fold)
    psm_source = pd.read_csv(args.psm_fold)
    id_sizes = _parse_ints(args.id_sizes)
    expected_methods = _parse_strings(args.verified_methods) + ["psm"]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    required_verified = set(KEY_COLUMNS + METRICS + ["implementation_version"])
    required_psm = set(KEY_COLUMNS + METRICS)
    if missing := sorted(required_verified - set(verified.columns)):
        raise SystemExit(f"Verified fold CSV is missing columns: {missing}")
    if missing := sorted(required_psm - set(psm_source.columns)):
        raise SystemExit(f"PSM fold CSV is missing columns: {missing}")

    verified = verified[
        verified["id_size"].isin(id_sizes)
        & verified["implementation_version"].eq("verified_v1")
    ].copy()
    psm = psm_source[
        psm_source["id_size"].isin(id_sizes)
        & psm_source["method"].eq("psm")
    ].copy()
    verified["source"] = "verified_v1"
    psm["source"] = "cleaned_psm"
    combined_columns = KEY_COLUMNS + METRICS + ["source"]
    combined = pd.concat(
        [verified[combined_columns], psm[combined_columns]],
        ignore_index=True,
    )

    duplicate_rows = int(combined.duplicated(KEY_COLUMNS, keep=False).sum())
    nonfinite_rows = int(
        (~np.isfinite(combined[METRICS].to_numpy(dtype=np.float64))).any(axis=1).sum()
    )
    combined = combined.drop_duplicates(KEY_COLUMNS, keep="last")
    methods = sorted(combined["method"].unique())
    expected_folds = {
        id_size: math.comb(8, id_size) * 3 * 2
        for id_size in id_sizes
    }
    coverage = (
        combined.groupby(["id_size", "method"])
        .size()
        .rename("completed_folds")
        .reset_index()
    )
    coverage["expected_folds"] = coverage["id_size"].map(expected_folds)
    coverage["complete"] = coverage["completed_folds"] == coverage["expected_folds"]
    expected_pairs = {(id_size, method) for id_size in id_sizes for method in expected_methods}
    actual_pairs = set(coverage[["id_size", "method"]].itertuples(index=False, name=None))
    missing_pairs = sorted(expected_pairs - actual_pairs)
    extra_methods = sorted(set(methods) - set(expected_methods))
    complete = bool(
        not missing_pairs
        and not extra_methods
        and coverage["complete"].all()
        and duplicate_rows == 0
        and nonfinite_rows == 0
    )

    summary = _summarize(combined)
    ranking, mean_ranking = _rankings(summary)
    metric_ranking, mean_metric_rank = _metric_rankings(summary)
    paired = _paired_rsn_vs_all(combined)
    combined.to_csv(output_dir / "fold_level_verified_with_psm.csv", index=False)
    summary.to_csv(output_dir / "summary_verified_with_psm.csv", index=False)
    ranking.to_csv(output_dir / "ranking_by_id_size.csv", index=False)
    mean_ranking.to_csv(output_dir / "mean_ranking_m2_m7.csv", index=False)
    metric_ranking.to_csv(output_dir / "metric_ranking_by_id_size.csv", index=False)
    mean_metric_rank.to_csv(output_dir / "mean_metric_rank_m2_m7.csv", index=False)
    paired.to_csv(output_dir / "paired_rsn_vs_all_by_id_size.csv", index=False)
    coverage.to_csv(output_dir / "coverage_by_method_id_size.csv", index=False)

    report = {
        "complete": complete,
        "id_sizes": id_sizes,
        "methods": methods,
        "method_count": len(methods),
        "fold_rows": int(len(combined)),
        "duplicate_rows": duplicate_rows,
        "nonfinite_rows": nonfinite_rows,
        "missing_method_id_size_pairs": [
            {"id_size": int(id_size), "method": method}
            for id_size, method in missing_pairs
        ],
        "extra_methods": extra_methods,
    }
    (output_dir / "aggregation_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, sort_keys=True))
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
