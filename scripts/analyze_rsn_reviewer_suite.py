#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRICS = ["AUROC", "FPR95", "AUPR_OUT", "AUPR_OUT_BALANCED"]
MAIN_METHOD = "rsn_class_std_huber_k150"
CORE_METHODS = [
    "knn_l2_pooled_k50_kth",
    "knn_raw_pooled_k150_kth",
    "raw_pooled_huber_k150",
    "knn_raw_classwise_k150_mean",
    "raw_classwise_huber_k150",
    "global_std_huber_k150",
    "class_std_squared_k150",
    MAIN_METHOD,
    "class_mad_huber_k150",
    "class_iqr_huber_k150",
    "rsn_equalbank3000",
    "rsn_class_std_huber_calibrated",
    "nnguide_k10",
]


def _effect(a: np.ndarray, b: np.ndarray, metric: str) -> np.ndarray:
    difference = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return -difference if metric == "FPR95" else difference


def _cluster_differences(
    frame: pd.DataFrame,
    method_a: str,
    method_b: str,
    metric: str,
    id_size: int,
) -> np.ndarray:
    keys = ["id_size", "id_set", "seed", "backbone"]
    block = frame[
        frame["id_size"].eq(id_size)
        & frame["method"].isin([method_a, method_b])
    ]
    pivot = block.pivot(index=keys, columns="method", values=metric).dropna()
    if method_a not in pivot or method_b not in pivot:
        return np.empty(0, dtype=float)
    pivot["effect"] = _effect(pivot[method_a], pivot[method_b], metric)
    return pivot.groupby(level=["id_size", "id_set"])["effect"].mean().to_numpy()


def _bootstrap_mean(
    values: np.ndarray,
    *,
    rng: np.random.Generator,
    n_boot: int,
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    draws = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(values.mean()), float(low), float(high)


def _sign_flip_p(
    values: np.ndarray,
    *,
    rng: np.random.Generator,
    n_perm: int,
) -> float:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return np.nan
    observed = abs(float(values.mean()))
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=(n_perm, len(values)))
    null = np.abs((signs * values[None, :]).mean(axis=1))
    return float((1.0 + np.sum(null >= observed)) / (1.0 + n_perm))


def paired_cluster_statistics(
    frame: pd.DataFrame,
    *,
    n_boot: int,
    n_perm: int,
    seed: int,
) -> pd.DataFrame:
    methods = sorted(set(frame["method"]) - {MAIN_METHOD})
    rows = []
    for method in methods:
        for metric in METRICS:
            by_size = {}
            for id_size in sorted(frame["id_size"].unique()):
                values = _cluster_differences(
                    frame, MAIN_METHOD, method, metric, int(id_size)
                )
                by_size[int(id_size)] = values
                rng = np.random.default_rng(seed + 1000 * int(id_size) + len(rows))
                mean, low, high = _bootstrap_mean(values, rng=rng, n_boot=n_boot)
                rows.append(
                    {
                        "scope": f"id_size={int(id_size)}",
                        "method_a": MAIN_METHOD,
                        "method_b": method,
                        "metric": metric,
                        "cluster_unit": "id_set_after_seed_backbone_mean",
                        "n_clusters": int(len(values)),
                        "mean_effect": mean,
                        "ci_low": low,
                        "ci_high": high,
                        "permutation_p": _sign_flip_p(
                            values, rng=rng, n_perm=n_perm
                        ),
                        "good_rate": float(np.mean(values > 0)) if len(values) else np.nan,
                    }
                )

            # Equal weight for every ID-size stratum, then resample ID-set clusters
            # independently inside each stratum.
            rng = np.random.default_rng(seed + 100_000 + len(rows))
            valid = [values for values in by_size.values() if len(values)]
            if valid:
                draws = np.column_stack(
                    [
                        rng.choice(values, size=(n_boot, len(values)), replace=True).mean(
                            axis=1
                        )
                        for values in valid
                    ]
                ).mean(axis=1)
                size_means = np.asarray([values.mean() for values in valid])
                pooled_mean = float(size_means.mean())
                low, high = np.quantile(draws, [0.025, 0.975])
                concatenated = np.concatenate(valid)
                rows.append(
                    {
                        "scope": "m2_m7_equal_weight",
                        "method_a": MAIN_METHOD,
                        "method_b": method,
                        "metric": metric,
                        "cluster_unit": "id_set_stratified_by_id_size",
                        "n_clusters": int(sum(len(values) for values in valid)),
                        "mean_effect": pooled_mean,
                        "ci_low": float(low),
                        "ci_high": float(high),
                        "permutation_p": _sign_flip_p(
                            concatenated, rng=rng, n_perm=n_perm
                        ),
                        "good_rate": float(np.mean(concatenated > 0)),
                    }
                )
    return pd.DataFrame(rows)


def confound_table(frame: pd.DataFrame) -> pd.DataFrame:
    main = frame[frame["method"].eq(MAIN_METHOD)].copy()
    columns = [
        "id_train_n",
        "id_n",
        "ood_n",
        "ood_prevalence",
        "id_class_count",
        "ood_class_count",
        "min_class_bank_n",
        "max_class_bank_n",
        "mean_class_bank_n",
        "AUROC",
        "FPR95",
        "AUPR_OUT",
        "AUPR_OUT_BALANCED",
        "MACRO_AUROC",
        "MACRO_AUPR_OUT",
    ]
    columns = [column for column in columns if column in main]
    summary = main.groupby("id_size")[columns].agg(["mean", "std"])
    summary.columns = [f"{column}_{stat}" for column, stat in summary.columns]
    return summary.reset_index()


def factor_summary(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = [metric for metric in METRICS if metric in frame]
    rows = []
    for (id_size, method), block in frame.groupby(["id_size", "method"]):
        row = {"scope": f"id_size={int(id_size)}", "id_size": int(id_size), "method": method, "n": len(block)}
        row.update({metric: float(block[metric].mean()) for metric in metrics})
        rows.append(row)
    for method, block in frame.groupby("method"):
        by_size = block.groupby("id_size")[metrics].mean()
        row = {"scope": "m2_m7_equal_weight", "id_size": "ALL", "method": method, "n": len(block)}
        row.update({metric: float(by_size[metric].mean()) for metric in metrics})
        rows.append(row)
    return pd.DataFrame(rows)


def species_macro_summary(per_ood: pd.DataFrame) -> pd.DataFrame:
    metrics = [metric for metric in METRICS if metric in per_ood]
    by_species = (
        per_ood.groupby(["method", "ood_class"])[metrics]
        .mean()
        .reset_index()
    )
    macro = by_species.groupby("method")[metrics].mean().reset_index()
    macro.insert(1, "ood_class", "MACRO_8_SPECIES")
    return pd.concat([by_species, macro], ignore_index=True)


def baseline_cluster_intervals(
    baseline: pd.DataFrame,
    *,
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    baseline = baseline[
        ~baseline["method"].isin(["rsn_paper"])
    ].copy()
    rows = []
    for method, block in baseline.groupby("method"):
        for metric in ("AUROC", "FPR95", "AUPR_OUT"):
            cluster = (
                block.groupby(["id_size", "id_set"])[metric]
                .mean()
                .reset_index()
            )
            rng = np.random.default_rng(seed + len(rows))
            strata = [
                values[metric].to_numpy(dtype=float)
                for _, values in cluster.groupby("id_size")
            ]
            draws = np.column_stack(
                [
                    rng.choice(values, size=(n_boot, len(values)), replace=True).mean(
                        axis=1
                    )
                    for values in strata
                ]
            ).mean(axis=1)
            point = float(cluster.groupby("id_size")[metric].mean().mean())
            low, high = np.quantile(draws, [0.025, 0.975])
            rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "mean_equal_id_size": point,
                    "ci_low": float(low),
                    "ci_high": float(high),
                    "cluster_unit": "id_set_after_seed_backbone_mean",
                }
            )
    return pd.DataFrame(rows)


def plot_full_ranking(frame: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 8.5))
    titles = {
        "AUROC": "AUROC (higher is better)",
        "FPR95": "FPR95 (lower is better)",
        "AUPR_OUT": "AUPR-OUT (higher is better)",
    }
    for ax, metric in zip(axes, ("AUROC", "FPR95", "AUPR_OUT"), strict=True):
        block = frame[frame["metric"].eq(metric)].copy()
        block = block.sort_values(
            "mean_equal_id_size", ascending=(metric == "FPR95")
        ).reset_index(drop=True)
        y = np.arange(len(block))
        x = block["mean_equal_id_size"].to_numpy()
        xerr = np.vstack([x - block["ci_low"], block["ci_high"] - x])
        colors = ["#007f73" if name == "rsn_reported" else "#7a7a7a" for name in block["method"]]
        ax.barh(y, x, color=colors, alpha=0.9)
        ax.errorbar(x, y, xerr=xerr, fmt="none", ecolor="#222222", capsize=2, linewidth=0.8)
        ax.set_yticks(y, block["method"], fontsize=7)
        ax.invert_yaxis()
        ax.set_title(titles[metric], weight="bold")
        ax.grid(axis="x", alpha=0.2)
    fig.suptitle("All verified baselines: equal-ID-size means and ID-set cluster 95% CIs", weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_factor_ablation(summary: pd.DataFrame, output: Path) -> None:
    block = summary[
        summary["scope"].eq("m2_m7_equal_weight")
        & summary["method"].isin(CORE_METHODS)
    ].copy()
    block = block.sort_values("AUROC", ascending=True)
    fig, axes = plt.subplots(1, 3, figsize=(14, 6.8))
    for ax, metric in zip(axes, ("AUROC", "FPR95", "AUPR_OUT_BALANCED"), strict=True):
        ordered = block.sort_values(metric, ascending=(metric != "FPR95"))
        colors = ["#007f73" if method == MAIN_METHOD else "#777777" for method in ordered["method"]]
        ax.barh(ordered["method"], ordered[metric], color=colors)
        ax.set_title(metric, weight="bold")
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(axis="x", alpha=0.2)
    fig.suptitle("RSN factor decomposition", weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold-results", required=True)
    parser.add_argument("--per-ood-results", required=True)
    parser.add_argument("--baseline-results", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--permutations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260712)
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    fold = pd.read_csv(args.fold_results).drop_duplicates("job_id", keep="last")
    per_ood = pd.read_csv(args.per_ood_results).drop_duplicates(
        ["job_id", "ood_class"], keep="last"
    )
    baseline = pd.read_csv(args.baseline_results).drop_duplicates("job_id", keep="last")

    paired = paired_cluster_statistics(
        fold, n_boot=args.bootstrap, n_perm=args.permutations, seed=args.seed
    )
    confounds = confound_table(fold)
    summary = factor_summary(fold)
    species = species_macro_summary(per_ood)
    baseline_ci = baseline_cluster_intervals(
        baseline, n_boot=args.bootstrap, seed=args.seed
    )

    paired.to_csv(output / "paired_idset_cluster_bootstrap.csv", index=False)
    confounds.to_csv(output / "m_confound_table.csv", index=False)
    summary.to_csv(output / "factor_summary.csv", index=False)
    species.to_csv(output / "per_ood_species_macro.csv", index=False)
    baseline_ci.to_csv(output / "baseline_21_cluster_ci.csv", index=False)
    plot_full_ranking(baseline_ci, output / "fig_baseline_21_cluster_ci.png")
    plot_factor_ablation(summary, output / "fig_factor_ablation.png")

    report = {
        "fold_rows": int(len(fold)),
        "methods": sorted(fold["method"].unique()),
        "id_sizes": sorted(int(value) for value in fold["id_size"].unique()),
        "paired_cluster_unit": "id_set_after_seed_backbone_mean",
        "bootstrap_replicates": int(args.bootstrap),
        "permutation_replicates": int(args.permutations),
        "uses_fold_independence_assumption": False,
        "aupr_out_balanced_prevalence": 0.5,
    }
    (output / "analysis_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
