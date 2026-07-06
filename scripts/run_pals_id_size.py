#!/usr/bin/env python
from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from maf07 import TARGET_CLASSES
from maf07.config import load_yaml, resolve_path
from maf07.jobs import _stable_job_id, append_csv_rows
from maf07.methods.pals import pals_from_env
from maf07.metrics import ood_metrics
from maf07.patch_features import patch_indices_for_split
from maf07.runner import _load_split
from maf07.splits import make_ood_eval_frame


DEFAULT_VARIANTS = "nll,comp,full"


def _parse_csv_values(raw: str | None, default: list[str]) -> list[str]:
    if not raw:
        return default
    return [item.strip() for chunk in raw.split(",") for item in chunk.split() if item.strip()]


def _jobs(
    backbones: list[str],
    seeds: list[int],
    classes: list[str],
    id_size: int,
    protocol: str,
    variants: list[str],
) -> pd.DataFrame:
    rows = []
    for seed, backbone, id_set, variant in itertools.product(
        seeds,
        backbones,
        itertools.combinations(classes, id_size),
        variants,
    ):
        id_classes = list(id_set)
        row = {
            "job_kind": "quick_pals",
            "protocol": protocol,
            "seed": int(seed),
            "backbone": backbone,
            "method": "pals",
            "variant": variant,
            "id_size": len(id_classes),
            "id_set": "|".join(id_classes),
            "ood_set": "|".join(c for c in classes if c not in set(id_classes)),
        }
        row["job_id"] = _stable_job_id(row)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["backbone", "seed", "id_set", "variant"]).reset_index(drop=True)


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
    for scope, sub in df.groupby("protocol", dropna=False):
        grouped = sub.groupby(["backbone", "method", "variant"], dropna=False).agg(
            n=("AUROC", "size"),
            AUROC_mean=("AUROC", "mean"),
            FPR95_mean=("FPR95", "mean"),
            AUPR_OUT_mean=("AUPR_OUT", "mean"),
        ).reset_index()
        allg = sub.groupby(["method", "variant"], dropna=False).agg(
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


def _class_rows(rows: pd.DataFrame, indices: np.ndarray, class_name: str, role: str) -> np.ndarray:
    mask = (rows["class_name"] == class_name) & (rows["role"] == role)
    return indices[mask.to_numpy()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--id-size", type=int, default=2)
    parser.add_argument("--protocol", default="fair", choices=["fair", "oracle"])
    parser.add_argument("--variants", default=os.environ.get("MAF07_PALS_VARIANTS", DEFAULT_VARIANTS))
    parser.add_argument("--backbones", default=os.environ.get("MAF07_BACKBONES", "dinov2_vitb14,dinov2_vitl14"))
    parser.add_argument("--seeds", default=os.environ.get("MAF07_SEEDS", "0,1,2"))
    parser.add_argument("--foreground-ratio", type=float, default=float(os.environ.get("MAF07_PALS_FOREGROUND_RATIO", "0.35")))
    parser.add_argument("--max-patches", type=int, default=int(os.environ.get("MAF07_PALS_MAX_FOREGROUND_PATCHES", "128")))
    parser.add_argument("--worker-index", type=int, default=int(os.environ.get("MAF07_JOB_WORKER_INDEX", "0")))
    parser.add_argument("--worker-count", type=int, default=int(os.environ.get("MAF07_JOB_WORKER_COUNT", "1")))
    parser.add_argument("--max-base-jobs", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--summary-output", default=None)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args(argv)

    variants = _parse_csv_values(args.variants, DEFAULT_VARIANTS.split(","))
    protocol = args.protocol
    output = resolve_path(args.output or f"results/quick/pals_id_size{args.id_size}_{protocol}_results.csv")
    summary_output = resolve_path(args.summary_output or f"results/quick/pals_id_size{args.id_size}_{protocol}_summary.csv")
    if args.summarize_only:
        _summarize(output, summary_output)
        return 0

    dcfg = load_yaml("configs/dataset.yaml")
    classes = list(dcfg.get("classes", TARGET_CLASSES))
    backbones = _parse_csv_values(args.backbones, ["dinov2_vitb14", "dinov2_vitl14"])
    seeds = [int(v) for v in _parse_csv_values(args.seeds, ["0", "1", "2"])]
    all_variant_jobs = _jobs(backbones, seeds, classes, args.id_size, protocol, variants)
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

    detector = pals_from_env()
    class_model_cache: dict[tuple[str, int, str], dict[str, object]] = {}
    completed = 0
    for _, base in base_jobs.iterrows():
        seed = int(base["seed"])
        backbone = str(base["backbone"])
        id_classes = str(base["id_set"]).split("|")
        variant_jobs = all_variant_jobs[
            (all_variant_jobs["seed"] == seed)
            & (all_variant_jobs["backbone"] == backbone)
            & (all_variant_jobs["id_set"] == str(base["id_set"]))
        ]
        variant_jobs = variant_jobs[~variant_jobs["job_id"].astype(str).isin(done)]
        if variant_jobs.empty:
            continue

        split = _load_split(seed)
        eval_frame = make_ood_eval_frame(split, id_classes, protocol=protocol)
        rows, patch_indices, patches, _ = patch_indices_for_split(
            eval_frame,
            backbone,
            foreground_ratio=float(args.foreground_ratio),
            max_patches=int(args.max_patches),
        )
        models = []
        for class_name in id_classes:
            key = (backbone, seed, class_name)
            if key not in class_model_cache:
                train_idx = _class_rows(rows, patch_indices, class_name, "id_train")
                cal_idx = _class_rows(rows, patch_indices, class_name, "id_val")
                if len(cal_idx) == 0:
                    cal_idx = train_idx
                class_model_cache[key] = detector.fit_class_model(
                    class_name,
                    patches,
                    train_idx,
                    cal_idx,
                    seed=int(_stable_job_id({"backbone": backbone, "seed": seed, "class": class_name})[:8], 16),
                )
                print(
                    json.dumps(
                        {
                            "class_model": class_name,
                            "backbone": backbone,
                            "seed": seed,
                            "train": int(len(train_idx)),
                            "cal": int(len(cal_idx)),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            models.append(class_model_cache[key])

        eval_mask = rows["role"].isin(["id_test", "ood_test"])
        eval_indices = patch_indices[eval_mask.to_numpy()]
        score_rows = rows.loc[eval_mask, ["ood_label"]].copy()
        is_id = (score_rows["ood_label"].to_numpy() == 0).astype(int)
        for _, job in variant_jobs.iterrows():
            variant = str(job["variant"]).lower()
            scores = detector.id_scores(patches, eval_indices, models, variant=variant)
            metrics = ood_metrics(is_id, scores)
            result = {**job.to_dict(), **metrics}
            append_csv_rows(pd.DataFrame([result]), output)
            completed += 1
            print(
                json.dumps(
                    {"completed": completed, "job_id": str(job["job_id"]), "variant": variant},
                    sort_keys=True,
                ),
                flush=True,
            )

    _summarize(output, summary_output)
    print(json.dumps({"completed": completed, "output": str(output), "summary": str(summary_output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
