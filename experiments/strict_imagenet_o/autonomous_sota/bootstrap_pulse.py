#!/usr/bin/env python3
"""Paired image-level bootstrap for frozen PULSE versus RC-MSPS."""

from __future__ import annotations

import argparse
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd

from audit_symmetric_support_proxy import metric_row
from evaluate_activation_shift import DATASETS


SCORES: dict[str, dict[str, np.ndarray]] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=16)
    return parser.parse_args()


def initialize(scores_dir: str) -> None:
    global SCORES
    root = Path(scores_dir)
    loaded = {}
    for name in ("id", *DATASETS):
        with np.load(root / f"{name}_scores.npz", allow_pickle=False) as data:
            loaded[name] = {
                "pulse": -data["pulse_ood_score"].astype(np.float64),
                "rc": data["rc_msps_confidence"].astype(np.float64),
            }
    SCORES = loaded


def one_draw(seed: int) -> list[dict[str, float | int | str]]:
    rng = np.random.default_rng(seed)
    id_index = rng.integers(0, len(SCORES["id"]["pulse"]), len(SCORES["id"]["pulse"]))
    rows, macro = [], {method: [] for method in ("pulse", "rc")}
    for dataset in DATASETS:
        count = len(SCORES[dataset]["pulse"])
        ood_index = rng.integers(0, count, count)
        values = {}
        for method in ("pulse", "rc"):
            values[method] = metric_row(
                SCORES["id"][method][id_index], SCORES[dataset][method][ood_index]
            )
            macro[method].append(values[method])
        for metric in ("AUROC", "FPR95", "AUPR_OUT"):
            rows.append(
                {
                    "dataset": dataset,
                    "metric": metric,
                    "pulse": values["pulse"][metric],
                    "rc_msps": values["rc"][metric],
                    "diff": values["pulse"][metric] - values["rc"][metric],
                }
            )
    for metric in ("AUROC", "FPR95", "AUPR_OUT"):
        pulse = float(np.mean([row[metric] for row in macro["pulse"]]))
        rc = float(np.mean([row[metric] for row in macro["rc"]]))
        rows.append(
            {
                "dataset": "macro",
                "metric": metric,
                "pulse": pulse,
                "rc_msps": rc,
                "diff": pulse - rc,
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    if args.iterations < 1000:
        raise ValueError("at least 1000 bootstrap iterations are required")
    seeds = np.random.SeedSequence(args.seed).generate_state(args.iterations).tolist()
    context = mp.get_context("fork")
    with context.Pool(
        processes=args.workers,
        initializer=initialize,
        initargs=(str(args.scores_dir),),
    ) as pool:
        draws = pool.map(one_draw, seeds)
    rows = []
    for iteration, iteration_rows in enumerate(draws):
        for row in iteration_rows:
            rows.append({"iteration": iteration, **row})
    table = pd.DataFrame(rows)

    summaries = []
    for (dataset, metric), group in table.groupby(["dataset", "metric"], sort=False):
        difference = group["diff"].to_numpy()
        higher_is_better = metric != "FPR95"
        summaries.append(
            {
                "dataset": dataset,
                "metric": metric,
                "iterations": len(group),
                "pulse_mean": group["pulse"].mean(),
                "rc_msps_mean": group["rc_msps"].mean(),
                "mean_diff": difference.mean(),
                "ci_low": np.quantile(difference, 0.025),
                "ci_high": np.quantile(difference, 0.975),
                "good_rate": float(np.mean(difference > 0 if higher_is_better else difference < 0)),
            }
        )
    summary = pd.DataFrame(summaries)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_dir / "paired_bootstrap_draws.csv.gz", index=False)
    summary.to_csv(args.output_dir / "paired_bootstrap.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
