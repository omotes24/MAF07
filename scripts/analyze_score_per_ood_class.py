#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from maf07.metrics import ood_metrics


JOB_COLUMNS = ["job_id", "protocol", "seed", "backbone", "method", "variant", "id_size", "id_set", "ood_set"]


def _score_paths(input_path: Path) -> list[Path]:
    if input_path.is_dir():
        return sorted(input_path.glob("*.parquet"))
    return [input_path]


def _per_job(frame: pd.DataFrame) -> list[pd.DataFrame]:
    if "job_id" in frame and frame["job_id"].nunique(dropna=False) > 1:
        return [sub.copy() for _, sub in frame.groupby("job_id", sort=False, dropna=False)]
    return [frame]


def _metrics_for_job(frame: pd.DataFrame) -> list[dict[str, object]]:
    if "is_id" not in frame:
        frame = frame.copy()
        frame["is_id"] = (frame["ood_label"] == 0).astype(int)
    id_rows = frame[frame["is_id"].astype(int) == 1]
    ood_rows = frame[frame["is_id"].astype(int) == 0]
    rows: list[dict[str, object]] = []
    if id_rows.empty or ood_rows.empty:
        return rows

    meta = {col: frame[col].iloc[0] for col in JOB_COLUMNS if col in frame}
    for ood_class, ood_class_rows in ood_rows.groupby("class_name", sort=True, dropna=False):
        block = pd.concat([id_rows, ood_class_rows], ignore_index=True)
        metrics = ood_metrics(block["is_id"].to_numpy(), block["score"].to_numpy())
        rows.append(
            {
                **meta,
                "ood_class": ood_class,
                "id_n": int(len(id_rows)),
                "ood_n": int(len(ood_class_rows)),
                **metrics,
            }
        )
    return rows


def _summarize(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return results
    metrics = ["AUROC", "FPR95", "AUPR_OUT", "DetectionError", "OSCR"]
    metrics = [metric for metric in metrics if metric in results]
    return (
        results.groupby(["protocol", "id_size", "backbone", "method", "variant", "ood_class"], dropna=False)
        .agg(
            n=("AUROC", "size"),
            id_n_mean=("id_n", "mean"),
            ood_n_mean=("ood_n", "mean"),
            **{f"{metric}_mean": (metric, "mean") for metric in metrics},
            **{f"{metric}_std": (metric, "std") for metric in metrics},
        )
        .reset_index()
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/ood/fair/scores/jobs")
    parser.add_argument("--output", default="results/analysis/per_ood_class_results.csv")
    parser.add_argument("--summary-output", default="results/analysis/per_ood_class_summary.csv")
    parser.add_argument("--max-files", type=int, default=None)
    args = parser.parse_args()

    paths = _score_paths(Path(args.input))
    if args.max_files is not None:
        paths = paths[: args.max_files]
    rows: list[dict[str, object]] = []
    for path in paths:
        frame = pd.read_parquet(path)
        for job_frame in _per_job(frame):
            rows.extend(_metrics_for_job(job_frame))

    results = pd.DataFrame(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output, index=False)
    summary = _summarize(results)
    summary_output = Path(args.summary_output)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_output, index=False)
    print(output)
    print(summary_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
