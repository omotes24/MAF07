#!/usr/bin/env python3
"""Locked paired bootstrap for PULSE versus RC-MSPS on the final suite."""

from __future__ import annotations

import argparse
import multiprocessing as mp
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from audit_symmetric_support_proxy import metric_row
from evaluate_final_locked import load_dataset_scores, load_scores
from final_lock import artifact_path, verify_locked_config


SCORES: dict[str, Any] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--locked-config", type=Path, required=True)
    return parser.parse_args()


def initialize(config_path: str) -> None:
    global SCORES
    lock, _ = verify_locked_config(Path(config_path))
    SCORES = {
        "id": load_scores(artifact_path(lock, "id_scores")),
        "datasets": {
            str(spec["name"]): load_dataset_scores(lock, str(spec["key"]))
            for spec in lock["final_suite"]["datasets"]
        },
    }


def one_draw(seed: int) -> list[dict[str, float | str]]:
    rng = np.random.default_rng(seed)
    id_count = len(SCORES["id"]["pulse"])
    id_index = rng.integers(0, id_count, id_count)
    rows = []
    macro = {method: [] for method in ("pulse", "rc")}
    for dataset, scores in SCORES["datasets"].items():
        count = len(scores["pulse"])
        ood_index = rng.integers(0, count, count)
        values = {
            "pulse": metric_row(
                -SCORES["id"]["pulse"][id_index], -scores["pulse"][ood_index]
            ),
            "rc": metric_row(
                SCORES["id"]["rc"][id_index], scores["rc"][ood_index]
            ),
        }
        for method in macro:
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
    lock, _ = verify_locked_config(args.locked_config)
    iterations = int(lock["evaluation"]["paired_bootstrap_replicates"])
    if iterations < 2000:
        raise RuntimeError("the final lock must require at least 2000 bootstrap draws")
    seed = int(lock["evaluation"]["bootstrap_seed"])
    seeds = np.random.SeedSequence(seed).generate_state(iterations).tolist()
    context = mp.get_context("fork")
    with context.Pool(
        processes=int(lock["execution"]["bootstrap_workers"]),
        initializer=initialize,
        initargs=(str(args.locked_config.resolve()),),
    ) as pool:
        draws = pool.map(one_draw, seeds)
    table = pd.DataFrame(
        [
            {"iteration": iteration, **row}
            for iteration, iteration_rows in enumerate(draws)
            for row in iteration_rows
        ]
    )
    summaries = []
    for (dataset, metric), group in table.groupby(["dataset", "metric"], sort=False):
        difference = group["diff"].to_numpy()
        good = difference < 0 if metric == "FPR95" else difference > 0
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
                "good_rate": float(np.mean(good)),
            }
        )
    summary = pd.DataFrame(summaries)
    output = Path(lock["outputs"]["result_dir"])
    output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output / "paired_bootstrap_draws.csv.gz", index=False)
    summary.to_csv(output / "paired_bootstrap.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
