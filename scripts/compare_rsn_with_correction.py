#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t, ttest_1samp, wilcoxon


def _existing_path(candidates: list[str]) -> Path:
    for item in candidates:
        path = Path(item)
        if path.exists():
            return path
    return Path(candidates[0])


def _holm_adjust(pvalues: np.ndarray) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, idx in enumerate(order):
        value = min(1.0, (m - rank) * p[idx])
        running = max(running, value)
        adjusted[idx] = running
    return adjusted


def _safe_wilcoxon(diff: np.ndarray) -> float:
    if np.allclose(diff, 0.0):
        return 1.0
    try:
        return float(wilcoxon(diff, zero_method="wilcox", alternative="two-sided").pvalue)
    except ValueError:
        return 1.0


def _load_rsn(path: Path, protocol: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["protocol"].astype(str) == protocol].copy()
    if "variant" in df:
        df = df[df["variant"].astype(str).isin(["main", "diag_huber_raw"])]
    df = df.drop_duplicates("job_id", keep="last")
    return df


def _load_baselines(path: Path, protocol: str, methods: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["protocol"].astype(str) == protocol].copy()
    df = df[df["method"].astype(str).isin(methods)].copy()
    if "variant" in df:
        df = df[df["variant"].astype(str).isin(["main", ""])]
    df = df.drop_duplicates("job_id", keep="last")
    return df


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rsn-results",
        default=str(
            _existing_path(
                [
                    "results/quick/rsn_full_results.csv",
                    "results/quick/diagcard_full_huber_raw_results.csv",
                ]
            )
        ),
    )
    parser.add_argument("--baseline-results", default="results/ood/fair/summary_by_setting.csv")
    parser.add_argument("--protocol", default="fair")
    parser.add_argument("--baselines", default="knn,card")
    parser.add_argument("--output", default="results/analysis/rsn_vs_baselines_corrected_stats.csv")
    args = parser.parse_args()

    baselines = [item.strip() for item in args.baselines.split(",") if item.strip()]
    rsn = _load_rsn(Path(args.rsn_results), args.protocol)
    base = _load_baselines(Path(args.baseline_results), args.protocol, baselines)
    keys = ["protocol", "seed", "backbone", "id_size", "id_set"]
    rows: list[dict[str, object]] = []
    for baseline in baselines:
        sub = base[base["method"].astype(str) == baseline]
        joined = rsn.merge(sub, on=keys, suffixes=("_rsn", "_base"), how="inner")
        for id_size, block in joined.groupby("id_size", sort=True):
            for metric in ["AUROC", "FPR95", "AUPR_OUT"]:
                diff = (block[f"{metric}_rsn"].astype(float) - block[f"{metric}_base"].astype(float)).to_numpy()
                n = int(len(diff))
                mean = float(np.mean(diff))
                sd = float(np.std(diff, ddof=1)) if n > 1 else 0.0
                se = sd / np.sqrt(n) if n > 1 else 0.0
                crit = float(t.ppf(0.975, n - 1)) if n > 1 else np.nan
                ci_low = mean - crit * se if n > 1 else mean
                ci_high = mean + crit * se if n > 1 else mean
                p_t = float(ttest_1samp(diff, 0.0, alternative="two-sided").pvalue) if n > 1 else 1.0
                p_w = _safe_wilcoxon(diff)
                good = diff < 0 if metric == "FPR95" else diff > 0
                rows.append(
                    {
                        "protocol": args.protocol,
                        "baseline": baseline,
                        "id_size": int(id_size),
                        "metric": metric,
                        "n": n,
                        "mean_diff": mean,
                        "median_diff": float(np.median(diff)),
                        "ci_low": float(ci_low),
                        "ci_high": float(ci_high),
                        "paired_t_p": p_t,
                        "wilcoxon_p": p_w,
                        "good_rate": float(np.mean(good)),
                    }
                )

    out = pd.DataFrame(rows)
    if not out.empty:
        for col in ["paired_t_p", "wilcoxon_p"]:
            p = out[col].to_numpy(dtype=float)
            out[f"{col}_bonferroni"] = np.minimum(1.0, p * len(p))
            out[f"{col}_holm"] = _holm_adjust(p)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
