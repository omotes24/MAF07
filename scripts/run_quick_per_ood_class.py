#!/usr/bin/env python
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from maf07 import TARGET_CLASSES
from maf07.config import load_yaml, resolve_path
from maf07.features import feature_frame_for_split
from maf07.jobs import _stable_job_id, append_csv_rows
from maf07.methods.baselines_distance import knn_score
from maf07.methods.card import card_from_env
from maf07.methods.psm import psm_from_env
from maf07.methods.rsn import rsn_from_env
from maf07.metrics import ood_metrics
from maf07.runner import _load_split, _split_logits
from maf07.splits import make_ood_eval_frame


def _parse_csv_values(raw: str | None, default: list[str]) -> list[str]:
    if not raw:
        return default
    return [item.strip().lower() for chunk in raw.split(",") for item in chunk.split() if item.strip()]


def _jobs(
    backbones: list[str],
    seeds: list[int],
    classes: list[str],
    id_size: int,
    protocol: str,
    methods: list[str],
) -> pd.DataFrame:
    rows = []
    for seed, backbone, id_set, method in itertools.product(
        seeds, backbones, itertools.combinations(classes, id_size), methods
    ):
        id_classes = list(id_set)
        row = {
            "job_kind": "quick_per_ood_class",
            "protocol": protocol,
            "seed": int(seed),
            "backbone": backbone,
            "method": method,
            "variant": "main",
            "id_size": len(id_classes),
            "id_set": "|".join(id_classes),
            "ood_set": "|".join(c for c in classes if c not in set(id_classes)),
        }
        row["job_id"] = _stable_job_id(row)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["backbone", "seed", "id_set", "method"]).reset_index(drop=True)


def _base_jobs(backbones: list[str], seeds: list[int], classes: list[str], id_size: int) -> pd.DataFrame:
    rows = []
    for seed, backbone, id_set in itertools.product(seeds, backbones, itertools.combinations(classes, id_size)):
        rows.append({"seed": int(seed), "backbone": backbone, "id_set": "|".join(id_set)})
    return pd.DataFrame(rows).sort_values(["backbone", "seed", "id_set"]).reset_index(drop=True)


def _score_method(
    method: str,
    train_x,
    train_y,
    val_x,
    val_y,
    eval_x,
    cache: dict[str, object],
):
    if method == "knn":
        if "knn" not in cache:
            cache["knn"] = knn_score(train_x, eval_x)
        return cache["knn"]
    if method == "psm":
        if "psm" not in cache:
            cache["psm"] = psm_from_env().fit(train_x).id_scores(eval_x)
        return cache["psm"]
    if method == "rsn":
        if "rsn" not in cache:
            cache["rsn"] = rsn_from_env().fit(train_x, train_y, z_cal=val_x, y_cal=val_y).id_scores(eval_x)
        return cache["rsn"]
    if method == "card":
        if "card" not in cache:
            if "split_logits" not in cache:
                cache["split_logits"] = _split_logits(train_x, train_y, val_x, eval_x)
            _, _, eval_logits = cache["split_logits"]
            cache["card"] = card_from_env().fit(train_x, train_y, z_cal=val_x, y_cal=val_y).id_scores(
                eval_x, eval_logits
            )
        return cache["card"]
    raise ValueError(f"Unsupported quick per-OOD method: {method}")


def _per_ood_metrics(base_rows: pd.DataFrame, scores) -> list[dict[str, object]]:
    frame = base_rows.copy()
    frame["score"] = scores
    frame["is_id"] = (frame["ood_label"].to_numpy() == 0).astype(int)
    id_rows = frame[frame["is_id"] == 1]
    ood_rows = frame[frame["is_id"] == 0]
    rows = []
    for ood_class, ood_class_rows in ood_rows.groupby("class_name", sort=True):
        block = pd.concat([id_rows, ood_class_rows], ignore_index=True)
        rows.append(
            {
                "ood_class": ood_class,
                "id_n": int(len(id_rows)),
                "ood_n": int(len(ood_class_rows)),
                **ood_metrics(block["is_id"].to_numpy(), block["score"].to_numpy()),
            }
        )
    return rows


def _summarize(results_path: Path, summary_path: Path) -> None:
    if not results_path.exists():
        return
    df = pd.read_csv(results_path)
    if df.empty:
        return
    df = df.drop_duplicates(["job_id", "ood_class"], keep="last")
    metrics = ["AUROC", "FPR95", "AUPR_OUT", "DetectionError", "OSCR"]
    metrics = [metric for metric in metrics if metric in df]
    summary = (
        df.groupby(["protocol", "id_size", "backbone", "method", "variant", "ood_class"], dropna=False)
        .agg(
            n=("AUROC", "size"),
            id_n_mean=("id_n", "mean"),
            ood_n_mean=("ood_n", "mean"),
            **{f"{metric}_mean": (metric, "mean") for metric in metrics},
            **{f"{metric}_std": (metric, "std") for metric in metrics},
        )
        .reset_index()
    )
    for col in summary.columns:
        if col.endswith("_mean") or col.endswith("_std"):
            summary[col] = summary[col].round(6)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--id-size", type=int, default=2)
    parser.add_argument("--protocol", default="fair", choices=["fair", "oracle"])
    parser.add_argument("--methods", default=os.environ.get("MAF07_QUICK_PER_OOD_METHODS", "rsn,knn,card"))
    parser.add_argument("--backbones", default=os.environ.get("MAF07_BACKBONES", "dinov2_vitb14,dinov2_vitl14"))
    parser.add_argument("--seeds", default=os.environ.get("MAF07_SEEDS", "0,1,2"))
    parser.add_argument("--worker-index", type=int, default=int(os.environ.get("MAF07_JOB_WORKER_INDEX", "0")))
    parser.add_argument("--worker-count", type=int, default=int(os.environ.get("MAF07_JOB_WORKER_COUNT", "1")))
    parser.add_argument("--max-base-jobs", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--summary-output", default=None)
    parser.add_argument("--split-dir", default=os.environ.get("MAF07_SPLIT_DIR", "results/splits"))
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args(argv)

    protocol = args.protocol
    output = resolve_path(args.output or f"results/analysis/per_ood_class_id_size{args.id_size}_{protocol}.csv")
    summary_output = resolve_path(
        args.summary_output or f"results/analysis/per_ood_class_id_size{args.id_size}_{protocol}_summary.csv"
    )
    if args.summarize_only:
        _summarize(output, summary_output)
        return 0

    dcfg = load_yaml("configs/dataset.yaml")
    classes = list(dcfg.get("classes", TARGET_CLASSES))
    backbones = _parse_csv_values(args.backbones, ["dinov2_vitb14", "dinov2_vitl14"])
    seeds = [int(v) for v in _parse_csv_values(args.seeds, ["0", "1", "2"])]
    methods = _parse_csv_values(args.methods, ["rsn", "knn", "card"])
    all_jobs = _jobs(backbones, seeds, classes, args.id_size, protocol, methods)
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

        split = _load_split(int(base["seed"]), args.split_dir)
        eval_frame = make_ood_eval_frame(split, id_classes, protocol=protocol)
        rows, features = feature_frame_for_split(eval_frame, str(base["backbone"]))
        train_mask = rows["role"] == "id_train"
        val_mask = rows["role"] == "id_val"
        eval_mask = rows["role"].isin(["id_test", "ood_test"])
        enc = LabelEncoder().fit(rows.loc[train_mask, "class_name"])
        train_y = enc.transform(rows.loc[train_mask, "class_name"])
        val_y = enc.transform(rows.loc[val_mask, "class_name"]) if val_mask.any() else train_y
        train_x = features[train_mask.to_numpy()]
        val_x = features[val_mask.to_numpy()] if val_mask.any() else train_x
        eval_x = features[eval_mask.to_numpy()]
        base_score_rows = rows.loc[eval_mask, ["image_id", "class_name", "split", "role", "ood_label"]].copy()
        cache: dict[str, object] = {}

        for _, job in job_rows.iterrows():
            scores = _score_method(str(job["method"]), train_x, train_y, val_x, val_y, eval_x, cache)
            per_class = _per_ood_metrics(base_score_rows, scores)
            out_rows = [{**job.to_dict(), **row} for row in per_class]
            append_csv_rows(pd.DataFrame(out_rows), output)
            completed += 1
            print(json.dumps({"completed": completed, "job_id": str(job["job_id"])}, sort_keys=True), flush=True)

    _summarize(output, summary_output)
    print(json.dumps({"completed": completed, "output": str(output), "summary": str(summary_output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
