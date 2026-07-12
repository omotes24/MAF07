#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


CORE_METHODS = [
    "knn_l2_pooled_k50_kth",
    "knn_raw_pooled_k150_kth",
    "knn_raw_pooled_k150_mean",
    "raw_pooled_huber_k150",
    "knn_raw_classwise_k150_mean",
    "raw_classwise_squared_k150",
    "raw_classwise_huber_k150",
    "global_std_huber_k150",
    "class_std_squared_k150",
    "rsn_class_std_huber_k150",
    "class_mad_huber_k150",
    "class_iqr_huber_k150",
    "rsn_equalbank3000",
    "rsn_class_std_huber_calibrated",
    "nnguide_k10",
    "nnguide_k50",
]

KEY_METHODS = [
    "knn_raw_classwise_k150_mean",
    "raw_classwise_squared_k150",
    "raw_classwise_huber_k150",
    "rsn_class_std_huber_k150",
    "knn_l2_pooled_k50_kth",
    "nnguide_k10",
]

SENSITIVITY_METHODS = [
    "rsn_k25",
    "rsn_k50",
    "rsn_k100",
    "rsn_class_std_huber_k150",
    "rsn_k300",
    "rsn_delta075",
    "rsn_delta100",
    "rsn_delta200",
    "rsn_delta300",
    "rsn_tau1e4",
    "rsn_tau1e2",
    "rsn_tau1e1",
]


def _value(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6f}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def _markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    columns = [str(column) for column in frame.columns]
    rows = ["| " + " | ".join(columns) + " |"]
    rows.append("|" + "|".join(["---"] * len(columns)) + "|")
    for values in frame.itertuples(index=False, name=None):
        rows.append("| " + " | ".join(_value(value) for value in values) + " |")
    return "\n".join(rows)


def _equal_m_backbone(fold: pd.DataFrame) -> pd.DataFrame:
    block = fold[fold["method"].isin(KEY_METHODS)]
    by_m = (
        block.groupby(["backbone", "id_size", "method"])[
            ["AUROC", "FPR95", "AUPR_OUT_BALANCED"]
        ]
        .mean()
        .reset_index()
    )
    return (
        by_m.groupby(["backbone", "method"])[
            ["AUROC", "FPR95", "AUPR_OUT_BALANCED"]
        ]
        .mean()
        .reset_index()
        .sort_values(["backbone", "AUROC"], ascending=[True, False])
    )


def _equal_m_species(per_ood: pd.DataFrame) -> pd.DataFrame:
    methods = [
        "rsn_class_std_huber_k150",
        "knn_raw_classwise_k150_mean",
        "knn_l2_pooled_k50_kth",
    ]
    block = per_ood[per_ood["method"].isin(methods)]
    by_stratum = (
        block.groupby(["method", "ood_class", "id_size", "backbone"])[
            ["AUROC", "FPR95", "AUPR_OUT_BALANCED"]
        ]
        .mean()
        .reset_index()
    )
    values = (
        by_stratum.groupby(["method", "ood_class"])[
            ["AUROC", "FPR95", "AUPR_OUT_BALANCED"]
        ]
        .mean()
        .reset_index()
    )
    pivot = values.pivot(index="ood_class", columns="method")
    result = pd.DataFrame(
        {
            "ood_class": pivot.index,
            "RSN_AUROC": pivot["AUROC"]["rsn_class_std_huber_k150"],
            "dAUROC_vs_raw_classwise": (
                pivot["AUROC"]["rsn_class_std_huber_k150"]
                - pivot["AUROC"]["knn_raw_classwise_k150_mean"]
            ),
            "RSN_FPR95": pivot["FPR95"]["rsn_class_std_huber_k150"],
            "FPR95_improvement_vs_raw": (
                pivot["FPR95"]["knn_raw_classwise_k150_mean"]
                - pivot["FPR95"]["rsn_class_std_huber_k150"]
            ),
            "RSN_balanced_AUPR": pivot["AUPR_OUT_BALANCED"][
                "rsn_class_std_huber_k150"
            ],
        }
    )
    return result.sort_values("RSN_AUROC")


def _baseline_table(baseline: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, block in baseline.groupby("method"):
        row: dict[str, object] = {"method": method}
        for metric in ("AUROC", "FPR95", "AUPR_OUT"):
            value = block[block["metric"].eq(metric)].iloc[0]
            row[metric] = value["mean_equal_id_size"]
            row[f"{metric}_95CI"] = (
                f"[{value['ci_low']:.6f}, {value['ci_high']:.6f}]"
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("AUROC", ascending=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    root = Path(args.input_dir)
    analysis = root / "analysis"
    factor = pd.read_csv(analysis / "factor_summary.csv")
    fold = pd.read_csv(root / "fold_level.csv").drop_duplicates("job_id", keep="last")
    per_ood = pd.read_csv(root / "per_ood_class.csv").drop_duplicates(
        ["job_id", "ood_class"], keep="last"
    )
    paired = pd.read_csv(analysis / "paired_idset_cluster_bootstrap.csv")
    baseline = pd.read_csv(analysis / "baseline_21_cluster_ci.csv")
    confounds = pd.read_csv(analysis / "m_confound_table.csv")
    runtime = pd.read_csv(root / "representative_fold" / "runtime_memory.csv")
    confusion = pd.read_csv(
        root / "representative_fold" / "species_confusion_matrix.csv", index_col=0
    )
    calibration = json.loads(
        (root / "calibration_assumptions" / "calibration_assumption_report.json").read_text()
    )
    core_coverage = json.loads((root / "core_coverage_report.json").read_text())
    sensitivity_coverage = json.loads(
        (root / "sensitivity_coverage_report.json").read_text()
    )

    equal = factor[factor["scope"].eq("m2_m7_equal_weight")]
    core = equal[equal["method"].isin(CORE_METHODS)].sort_values(
        "AUROC", ascending=False
    )
    all_variants = equal.sort_values("AUROC", ascending=False)

    by_m = factor[
        factor["id_size"].astype(str).isin(["2", "3", "4", "5", "6", "7"])
        & factor["method"].isin(KEY_METHODS)
    ].sort_values(["id_size", "AUROC"], ascending=[True, False])

    restricted = factor[
        factor["id_size"].astype(str).isin(["2", "4", "7"])
        & factor["method"].isin(SENSITIVITY_METHODS)
    ]
    sensitivity = (
        restricted.groupby("method")[["AUROC", "FPR95", "AUPR_OUT_BALANCED"]]
        .mean()
        .reset_index()
        .sort_values("AUROC", ascending=False)
    )

    matched_methods = [
        "knn_raw_classwise_k150_mean",
        "raw_classwise_squared_k150",
        "raw_classwise_huber_k150",
        "global_std_euclidean_k150",
        "global_std_squared_k150",
        "global_std_huber_k150",
        "class_std_euclidean_k150",
        "class_std_squared_k150",
        "rsn_class_std_huber_k150",
        "class_mad_huber_k150",
        "class_iqr_huber_k150",
        "l2_class_std_euclidean_k150",
        "l2_class_std_squared_k150",
        "l2_class_std_huber_k150",
        "rsn_equalbank3000",
        "rsn_class_std_huber_calibrated",
    ]
    matched = factor[
        factor["id_size"].astype(str).isin(["2", "4", "7"])
        & factor["method"].isin(matched_methods)
    ]
    matched = (
        matched.groupby("method")[["AUROC", "FPR95", "AUPR_OUT_BALANCED"]]
        .mean()
        .reset_index()
        .sort_values("AUROC", ascending=False)
    )

    effect_methods = [
        "knn_l2_pooled_k50_kth",
        "knn_raw_classwise_k150_mean",
        "raw_classwise_squared_k150",
        "raw_classwise_huber_k150",
        "class_std_squared_k150",
        "class_mad_huber_k150",
        "class_iqr_huber_k150",
        "rsn_equalbank3000",
        "rsn_class_std_huber_calibrated",
        "nnguide_k10",
    ]
    effects = paired[
        paired["scope"].eq("m2_m7_equal_weight")
        & paired["method_b"].isin(effect_methods)
        & paired["metric"].isin(["AUROC", "FPR95", "AUPR_OUT_BALANCED"])
    ][
        [
            "method_b",
            "metric",
            "mean_effect",
            "ci_low",
            "ci_high",
            "permutation_p",
            "good_rate",
        ]
    ].sort_values(["method_b", "metric"])

    confound_columns = [
        "id_size",
        "id_train_n_mean",
        "id_n_mean",
        "ood_n_mean",
        "ood_prevalence_mean",
        "AUROC_mean",
        "FPR95_mean",
        "AUPR_OUT_mean",
        "AUPR_OUT_BALANCED_mean",
        "MACRO_AUROC_mean",
    ]
    confound_table = confounds[confound_columns]

    calibration_table = pd.DataFrame(
        [
            {"variant": "raw", **calibration["raw_metrics"]},
            {"variant": "calibrated", **calibration["calibrated_metrics"]},
        ]
    )
    calibration_rank = pd.DataFrame(
        [
            {
                "sample_n": calibration["sample_n"],
                "spearman_rho": calibration["spearman_rho"],
                "kendall_tau": calibration["kendall_tau"],
                "discordant_pair_fraction": calibration[
                    "discordant_pair_fraction_from_tau"
                ],
                "mean_abs_rank_displacement": calibration[
                    "mean_absolute_rank_displacement"
                ],
                "max_abs_rank_displacement": calibration[
                    "max_absolute_rank_displacement"
                ],
            }
        ]
    )

    coverage = pd.DataFrame(
        [
            {"block": "core", **core_coverage},
            {"block": "sensitivity", **sensitivity_coverage},
        ]
    )
    coverage = coverage[
        [
            "block",
            "method_count",
            "expected_jobs",
            "completed_jobs",
            "missing_jobs",
            "duplicate_job_rows",
            "expected_per_ood_rows",
            "completed_per_ood_rows",
            "strict_pass",
        ]
    ]

    report = [
        "# RSN Reviewer Suite: Complete Available Results",
        "",
        "Generated from the cleaned 105,554-image experiment output. Core results use "
        "equal weight across m=2,...,7. Sensitivity tables use equal weight across "
        "m={2,4,7}; they must not be compared directly with the six-m aggregate.",
        "",
        "## Coverage",
        "",
        _markdown(coverage),
        "",
        f"Unique fold-method rows: **{len(fold):,}**. Per-OOD rows: **{len(per_ood):,}**.",
        "",
        "## Core 16-condition ranking (m=2,...,7 equal weight; n=1,476 each)",
        "",
        _markdown(core[["method", "n", "AUROC", "FPR95", "AUPR_OUT", "AUPR_OUT_BALANCED"]]),
        "",
        "## All 33 registered conditions",
        "",
        "Rows with n=1,476 cover all six m values. Rows with n=636 are sensitivity "
        "conditions covering m={2,4,7} only.",
        "",
        _markdown(all_variants[["method", "n", "AUROC", "FPR95", "AUPR_OUT", "AUPR_OUT_BALANCED"]]),
        "",
        "## Key methods by ID class count",
        "",
        _markdown(by_m[["id_size", "method", "n", "AUROC", "FPR95", "AUPR_OUT_BALANCED"]]),
        "",
        "## Matched factor decomposition (m={2,4,7} equal weight)",
        "",
        _markdown(matched),
        "",
        "## Independent k, delta, and tau sensitivity (m={2,4,7} equal weight)",
        "",
        _markdown(sensitivity),
        "",
        "## Backbone consistency (m=2,...,7 equal weight)",
        "",
        _markdown(_equal_m_backbone(fold)),
        "",
        "## Per-OOD-species results (m and backbone equal weight)",
        "",
        _markdown(_equal_m_species(per_ood)),
        "",
        "## ID-set cluster bootstrap and sign-flip effects",
        "",
        "Positive effect means RSN is better. FPR95 is direction-adjusted, so positive "
        "means RSN has lower FPR95.",
        "",
        _markdown(effects),
        "",
        "## Verified 21-baseline ranking with ID-set cluster 95% CIs",
        "",
        _markdown(_baseline_table(baseline)),
        "",
        "## m confounds and prevalence",
        "",
        _markdown(confound_table),
        "",
        "## Calibration diagnostic (representative cheetah|jaguar fold)",
        "",
        _markdown(calibration_table),
        "",
        _markdown(calibration_rank),
        "",
        "The representative fold improves after calibration, but the all-fold core "
        "aggregate deteriorates from AUROC 0.870392 / FPR95 0.566419 to AUROC "
        "0.864502 / FPR95 0.637795.",
        "",
        "## Runtime and peak GPU memory (representative fold)",
        "",
        _markdown(runtime),
        "",
        "## Closed-set species confusion matrix",
        "",
        _markdown(confusion.reset_index().rename(columns={"index": "true_class"})),
        "",
        f"Macro diagonal accuracy: **{float(np.diag(confusion).mean()):.6f}**.",
        "",
        "## Source files",
        "",
        "- `fold_level.csv`: every fold-method result",
        "- `per_ood_class.csv`: every fold-method-OOD-species result",
        "- `analysis/factor_summary.csv`: aggregate factor table",
        "- `analysis/paired_idset_cluster_bootstrap.csv`: dependence-aware comparisons",
        "- `analysis/baseline_21_cluster_ci.csv`: verified baseline intervals",
        "- `representative_fold/`: qualitative panels, score distributions, dimensions, "
        "  confusion matrix, and runtime",
        "- `calibration_assumptions/`: rank reversals and DKW union-bound diagnostics",
        "",
        "Candidate-count counterfactual, data audits, external backbones, and saliency "
        "ablation are queued separately and are not included in this frozen report.",
        "",
    ]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"output": str(output), "fold_rows": len(fold), "per_ood_rows": len(per_ood)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
