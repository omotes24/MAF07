#!/usr/bin/env python
from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from maf07 import TARGET_CLASSES
from maf07.config import load_yaml, resolve_path
from maf07.features import feature_frame_for_split
from maf07.jobs import _stable_job_id, append_csv_rows
from maf07.methods.cats import cats_from_env
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
            "job_kind": "quick_cats",
            "protocol": protocol,
            "seed": int(seed),
            "backbone": backbone,
            "method": "cats",
            "variant": "main",
            "id_size": len(id_classes),
            "id_set": "|".join(id_classes),
            "ood_set": "|".join(c for c in classes if c not in set(id_classes)),
        }
        row["job_id"] = _stable_job_id(row)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["backbone", "seed", "id_set"]).reset_index(drop=True)


def _summarize(results_path: Path, summary_path: Path) -> None:
    if not results_path.exists():
        return
    df = pd.read_csv(results_path)
    if df.empty:
        return
    df = df.drop_duplicates("job_id", keep="last")
    rows = []
    for scope, sub in df.groupby("protocol", dropna=False):
        grouped = sub.groupby(["backbone", "method"], dropna=False).agg(
            n=("AUROC", "size"),
            AUROC_mean=("AUROC", "mean"),
            FPR95_mean=("FPR95", "mean"),
            AUPR_OUT_mean=("AUPR_OUT", "mean"),
        ).reset_index()
        allg = sub.groupby(["method"], dropna=False).agg(
            n=("AUROC", "size"),
            AUROC_mean=("AUROC", "mean"),
            FPR95_mean=("FPR95", "mean"),
            AUPR_OUT_mean=("AUPR_OUT", "mean"),
        ).reset_index()
        allg.insert(0, "backbone", "ALL")
        out = pd.concat([grouped, allg], ignore_index=True)
        out.insert(0, "scope", scope)
        rows.extend(out.to_dict("records"))
    summary = pd.DataFrame(rows)
    for col in ["AUROC_mean", "FPR95_mean", "AUPR_OUT_mean"]:
        if col in summary:
            summary[col] = summary[col].round(6)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--id-size", type=int, default=2)
    parser.add_argument("--protocol", default="fair", choices=["fair", "oracle"])
    parser.add_argument("--backbones", default=os.environ.get("MAF07_BACKBONES", "dinov2_vitb14,dinov2_vitl14"))
    parser.add_argument("--seeds", default=os.environ.get("MAF07_SEEDS", "0,1,2"))
    parser.add_argument("--worker-index", type=int, default=int(os.environ.get("MAF07_JOB_WORKER_INDEX", "0")))
    parser.add_argument("--worker-count", type=int, default=int(os.environ.get("MAF07_JOB_WORKER_COUNT", "1")))
    parser.add_argument("--max-jobs", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--summary-output", default=None)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args(argv)

    protocol = args.protocol
    output = resolve_path(args.output or f"results/quick/cats_id_size{args.id_size}_{protocol}_results.csv")
    summary_output = resolve_path(args.summary_output or f"results/quick/cats_id_size{args.id_size}_{protocol}_summary.csv")
    if args.summarize_only:
        _summarize(output, summary_output)
        return 0

    dcfg = load_yaml("configs/dataset.yaml")
    classes = list(dcfg.get("classes", TARGET_CLASSES))
    backbones = _parse_csv_values(args.backbones, ["dinov2_vitb14", "dinov2_vitl14"])
    seeds = [int(v) for v in _parse_csv_values(args.seeds, ["0", "1", "2"])]
    jobs = _jobs(backbones, seeds, classes, args.id_size, protocol)
    if args.worker_count > 1:
        start = len(jobs) * args.worker_index // args.worker_count
        end = len(jobs) * (args.worker_index + 1) // args.worker_count
        jobs = jobs.iloc[start:end].reset_index(drop=True)
    if args.max_jobs is not None:
        jobs = jobs.head(args.max_jobs)

    done: set[str] = set()
    if output.exists():
        existing = pd.read_csv(output, usecols=["job_id"])
        done = set(existing["job_id"].astype(str))

    completed = 0
    for _, job in jobs.iterrows():
        if str(job["job_id"]) in done:
            continue
        split = _load_split(int(job["seed"]))
        id_classes = str(job["id_set"]).split("|")
        eval_frame = make_ood_eval_frame(split, id_classes, protocol=protocol)
        rows, features = feature_frame_for_split(eval_frame, str(job["backbone"]))
        train_mask = rows["role"] == "id_train"
        val_mask = rows["role"] == "id_val"
        eval_mask = rows["role"].isin(["id_test", "ood_test"])
        enc = LabelEncoder().fit(rows.loc[train_mask, "class_name"])
        train_y = enc.transform(rows.loc[train_mask, "class_name"])
        val_y = enc.transform(rows.loc[val_mask, "class_name"]) if val_mask.any() else train_y
        train_x = features[train_mask.to_numpy()]
        val_x = features[val_mask.to_numpy()] if val_mask.any() else train_x
        eval_x = features[eval_mask.to_numpy()]
        scores = cats_from_env().fit(train_x, train_y, z_cal=val_x, y_cal=val_y).id_scores(eval_x)
        score_rows = rows.loc[eval_mask, ["ood_label"]].copy()
        is_id = (score_rows["ood_label"].to_numpy() == 0).astype(int)
        metrics = ood_metrics(is_id, scores)
        result = {**job.to_dict(), **metrics}
        append_csv_rows(pd.DataFrame([result]), output)
        completed += 1
        print(json.dumps({"completed": completed, "job_id": str(job["job_id"])}, sort_keys=True), flush=True)

    _summarize(output, summary_output)
    print(json.dumps({"completed": completed, "output": str(output), "summary": str(summary_output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
