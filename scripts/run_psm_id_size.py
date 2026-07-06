#!/usr/bin/env python
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from maf07 import TARGET_CLASSES
from maf07.config import load_yaml, resolve_path
from maf07.features import feature_frame_for_split
from maf07.jobs import _stable_job_id, append_csv_rows
from maf07.methods.psm import psm_from_env
from maf07.metrics import ood_metrics
from maf07.runner import _load_split
from maf07.splits import make_ood_eval_frame


def _parse_csv_values(raw: str | None, default: list[str]) -> list[str]:
    if not raw:
        return default
    return [item.strip() for chunk in raw.split(",") for item in chunk.split() if item.strip()]


def _jobs(backbones: list[str], seeds: list[int], classes: list[str], id_size: int, protocol: str) -> pd.DataFrame:
    rows = []
    for seed, backbone, id_set in itertools.product(seeds, backbones, itertools.combinations(classes, id_size)):
        id_classes = list(id_set)
        row = {
            "job_kind": "quick_psm",
            "protocol": protocol,
            "seed": int(seed),
            "backbone": backbone,
            "method": "psm",
            "variant": "main",
            "id_size": len(id_classes),
            "id_set": "|".join(id_classes),
            "ood_set": "|".join(c for c in classes if c not in set(id_classes)),
        }
        row["job_id"] = _stable_job_id(row)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["backbone", "seed", "id_set"]).reset_index(drop=True)


def _base_jobs(backbones: list[str], seeds: list[int], classes: list[str], id_size: int) -> pd.DataFrame:
    rows = []
    for seed, backbone, id_set in itertools.product(seeds, backbones, itertools.combinations(classes, id_size)):
        rows.append({"seed": int(seed), "backbone": backbone, "id_set": "|".join(id_set)})
    return pd.DataFrame(rows).sort_values(["backbone", "seed", "id_set"]).reset_index(drop=True)


def _summarize(results_path: Path, summary_path: Path) -> None:
    if not results_path.exists():
        return
    df = pd.read_csv(results_path)
    if df.empty:
        return
    df = df.drop_duplicates("job_id", keep="last")
    rows = []
    for (protocol, id_size), sub0 in df.groupby(["protocol", "id_size"], dropna=False):
        by_backbone = (
            sub0.groupby(["backbone", "method", "variant"], dropna=False)
            .agg(
                n=("AUROC", "size"),
                AUROC_mean=("AUROC", "mean"),
                FPR95_mean=("FPR95", "mean"),
                AUPR_OUT_mean=("AUPR_OUT", "mean"),
            )
            .reset_index()
        )
        allg = (
            sub0.groupby(["method", "variant"], dropna=False)
            .agg(
                n=("AUROC", "size"),
                AUROC_mean=("AUROC", "mean"),
                FPR95_mean=("FPR95", "mean"),
                AUPR_OUT_mean=("AUPR_OUT", "mean"),
            )
            .reset_index()
        )
        allg.insert(0, "backbone", "ALL")
        block = pd.concat([by_backbone, allg], ignore_index=True)
        block.insert(0, "id_size", int(id_size))
        block.insert(0, "protocol", protocol)
        rows.extend(block.to_dict("records"))
    summary = pd.DataFrame(rows)
    for col in ["AUROC_mean", "FPR95_mean", "AUPR_OUT_mean"]:
        summary[col] = summary[col].round(6)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.sort_values(["protocol", "id_size", "backbone"]).to_csv(summary_path, index=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--id-size", type=int, default=2)
    parser.add_argument("--protocol", default="fair", choices=["fair", "oracle"])
    parser.add_argument("--backbones", default=os.environ.get("MAF07_BACKBONES", "dinov2_vitb14,dinov2_vitl14"))
    parser.add_argument("--seeds", default=os.environ.get("MAF07_SEEDS", "0,1,2"))
    parser.add_argument("--worker-index", type=int, default=int(os.environ.get("MAF07_JOB_WORKER_INDEX", "0")))
    parser.add_argument("--worker-count", type=int, default=int(os.environ.get("MAF07_JOB_WORKER_COUNT", "1")))
    parser.add_argument("--max-base-jobs", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--summary-output", default=None)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args(argv)

    protocol = args.protocol
    output = resolve_path(args.output or f"results/quick/psm_id_size{args.id_size}_{protocol}_results.csv")
    summary_output = resolve_path(args.summary_output or f"results/quick/psm_id_size{args.id_size}_{protocol}_summary.csv")
    if args.summarize_only:
        _summarize(output, summary_output)
        return 0

    dcfg = load_yaml("configs/dataset.yaml")
    classes = list(dcfg.get("classes", TARGET_CLASSES))
    backbones = _parse_csv_values(args.backbones, ["dinov2_vitb14", "dinov2_vitl14"])
    seeds = [int(v) for v in _parse_csv_values(args.seeds, ["0", "1", "2"])]
    all_jobs = _jobs(backbones, seeds, classes, args.id_size, protocol)
    done: set[str] = set()
    if output.exists():
        existing = pd.read_csv(output, usecols=["job_id"])
        done = set(existing["job_id"].astype(str))

    base_jobs = _base_jobs(backbones, seeds, classes, args.id_size)
    if args.worker_count > 1:
        start = len(base_jobs) * args.worker_index // args.worker_count
        end = len(base_jobs) * (args.worker_index + 1) // args.worker_count
        base_jobs = base_jobs.iloc[start:end].reset_index(drop=True)
    if args.max_base_jobs is not None:
        base_jobs = base_jobs.head(args.max_base_jobs)

    completed = 0
    for _, base in base_jobs.iterrows():
        id_classes = str(base["id_set"]).split("|")
        job_rows = all_jobs[
            (all_jobs["seed"] == int(base["seed"]))
            & (all_jobs["backbone"] == str(base["backbone"]))
            & (all_jobs["id_set"] == str(base["id_set"]))
        ]
        job_rows = job_rows[~job_rows["job_id"].astype(str).isin(done)]
        if job_rows.empty:
            continue

        split = _load_split(int(base["seed"]))
        eval_frame = make_ood_eval_frame(split, id_classes, protocol=protocol)
        rows, features = feature_frame_for_split(eval_frame, str(base["backbone"]))
        train_mask = rows["role"] == "id_train"
        eval_mask = rows["role"].isin(["id_test", "ood_test"])
        train_x = features[train_mask.to_numpy()]
        eval_x = features[eval_mask.to_numpy()]
        scores = psm_from_env().fit(train_x).id_scores(eval_x)
        score_rows = rows.loc[eval_mask, ["ood_label"]].copy()
        is_id = (score_rows["ood_label"].to_numpy() == 0).astype(int)
        metrics = ood_metrics(is_id, scores)

        for _, job in job_rows.iterrows():
            result = {**job.to_dict(), **metrics}
            append_csv_rows(pd.DataFrame([result]), output)
            completed += 1
            print(json.dumps({"completed": completed, "job_id": str(job["job_id"])}, sort_keys=True), flush=True)

    _summarize(output, summary_output)
    print(json.dumps({"completed": completed, "output": str(output), "summary": str(summary_output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
