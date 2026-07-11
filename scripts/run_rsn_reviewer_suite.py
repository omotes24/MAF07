#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from maf07 import TARGET_CLASSES
from maf07.config import load_yaml, resolve_path
from maf07.features import feature_frame_for_split
from maf07.jobs import _stable_job_id, append_csv_rows
from maf07.methods.diagcard import DiagCARDDetector
from maf07.methods.rsn_factorial import (
    FactorialNeighborSuite,
    nnguide_score,
    reviewer_factor_specs,
)
from maf07.methods.verified_baselines import VerifiedBaselineSuite, score_fingerprint
from maf07.metrics import ood_metrics
from maf07.runner import _load_split
from maf07.splits import make_ood_eval_frame


IMPLEMENTATION_VERSION = "reviewer_factorial_v1"
SPECIAL_METHODS = {
    "rsn_class_std_huber_calibrated",
    "nnguide_k10",
    "nnguide_k50",
}
SUMMARY_METRICS = [
    "AUROC",
    "FPR95",
    "AUPR_OUT",
    "AUPR_OUT_BALANCED",
    "MACRO_AUROC",
    "MACRO_FPR95",
    "MACRO_AUPR_OUT",
]


def _parse_values(raw: str) -> list[str]:
    return [item.strip() for chunk in raw.split(",") for item in chunk.split() if item.strip()]


def _classes() -> list[str]:
    config = load_yaml("configs/dataset.yaml")
    return list(config.get("classes", TARGET_CLASSES))


def _method_names() -> list[str]:
    return [spec.name for spec in reviewer_factor_specs()] + sorted(SPECIAL_METHODS)


def _jobs(
    backbones: list[str],
    seeds: list[int],
    classes: list[str],
    id_size: int,
    methods: list[str],
) -> pd.DataFrame:
    id_sets = list(itertools.combinations(classes, id_size))
    id_set_ids = {"|".join(values): index for index, values in enumerate(id_sets)}
    rows = []
    for seed, backbone, id_set, method in itertools.product(
        seeds, backbones, id_sets, methods
    ):
        id_text = "|".join(id_set)
        id_classes = set(id_set)
        row = {
            "job_kind": "rsn_reviewer_factorial",
            "scope": "fair",
            "protocol": "fair",
            "seed": int(seed),
            "backbone": backbone,
            "id_size": int(id_size),
            "id_set_id": int(id_set_ids[id_text]),
            "id_set": id_text,
            "ood_set": "|".join(cls for cls in classes if cls not in id_classes),
            "method": method,
            "implementation_version": IMPLEMENTATION_VERSION,
        }
        row["job_id"] = _stable_job_id(row)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["backbone", "seed", "id_set_id", "method"]
    ).reset_index(drop=True)


def _base_jobs(
    backbones: list[str], seeds: list[int], classes: list[str], id_size: int
) -> pd.DataFrame:
    rows = []
    for seed, backbone, id_set in itertools.product(
        seeds, backbones, itertools.combinations(classes, id_size)
    ):
        rows.append(
            {
                "seed": int(seed),
                "backbone": backbone,
                "id_set": "|".join(id_set),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["backbone", "seed", "id_set"]
    ).reset_index(drop=True)


def balanced_aupr_out(y_is_id: np.ndarray, id_scores: np.ndarray) -> float:
    y_ood = 1 - np.asarray(y_is_id, dtype=int)
    scores = -np.asarray(id_scores, dtype=np.float64)
    n_ood = int(y_ood.sum())
    n_id = int(len(y_ood) - n_ood)
    weights = np.where(y_ood == 1, 0.5 / n_ood, 0.5 / n_id)
    return float(average_precision_score(y_ood, scores, sample_weight=weights))


def _per_ood_metrics(
    score_rows: pd.DataFrame,
    scores: np.ndarray,
    job: dict[str, object],
) -> tuple[list[dict[str, object]], dict[str, float]]:
    frame = score_rows.copy()
    frame["score"] = np.asarray(scores, dtype=np.float64)
    frame["is_id"] = (frame["ood_label"].to_numpy() == 0).astype(int)
    id_rows = frame[frame["is_id"] == 1]
    per_ood = []
    for class_name, ood_rows in frame[frame["is_id"] == 0].groupby(
        "class_name", sort=True
    ):
        block = pd.concat([id_rows, ood_rows], ignore_index=True)
        values = ood_metrics(block["is_id"].to_numpy(), block["score"].to_numpy())
        per_ood.append(
            {
                **job,
                "ood_class": str(class_name),
                "id_n": int(len(id_rows)),
                "ood_n": int(len(ood_rows)),
                "AUPR_OUT_BALANCED": balanced_aupr_out(
                    block["is_id"].to_numpy(), block["score"].to_numpy()
                ),
                **values,
            }
        )
    per_frame = pd.DataFrame(per_ood)
    macro = {
        "MACRO_AUROC": float(per_frame["AUROC"].mean()),
        "MACRO_FPR95": float(per_frame["FPR95"].mean()),
        "MACRO_AUPR_OUT": float(per_frame["AUPR_OUT"].mean()),
        "MACRO_AUPR_OUT_BALANCED": float(per_frame["AUPR_OUT_BALANCED"].mean()),
    }
    return per_ood, macro


def _summarize(results_path: Path, summary_path: Path) -> None:
    if not results_path.exists() or results_path.stat().st_size == 0:
        return
    frame = pd.read_csv(results_path).drop_duplicates("job_id", keep="last")
    metrics = [metric for metric in SUMMARY_METRICS if metric in frame]
    grouped = (
        frame.groupby(["backbone", "id_size", "method"], dropna=False)[metrics]
        .agg(["size", "mean", "std"])
    )
    grouped.columns = [f"{metric}_{stat}" for metric, stat in grouped.columns]
    grouped = grouped.reset_index()
    pooled = frame.groupby(["id_size", "method"], dropna=False)[metrics].agg(
        ["size", "mean", "std"]
    )
    pooled.columns = [f"{metric}_{stat}" for metric, stat in pooled.columns]
    pooled = pooled.reset_index()
    pooled.insert(0, "backbone", "ALL")
    summary = pd.concat([grouped, pooled], ignore_index=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.sort_values(["id_size", "backbone", "AUROC_mean"], ascending=[True, True, False]).to_csv(
        summary_path, index=False
    )


def _selected_sample_fold(args, base: pd.Series) -> bool:
    return bool(
        args.score_sample_output
        and int(base["seed"]) == args.score_sample_seed
        and str(base["backbone"]) == args.score_sample_backbone
        and str(base["id_set"]) == args.score_sample_id_set
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--id-size", type=int, required=True)
    parser.add_argument("--methods", default="all")
    parser.add_argument(
        "--backbones", default=os.environ.get("MAF07_BACKBONES", "dinov2_vitb14,dinov2_vitl14")
    )
    parser.add_argument("--seeds", default=os.environ.get("MAF07_SEEDS", "0,1,2"))
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--worker-count", type=int, default=1)
    parser.add_argument("--max-base-jobs", type=int, default=None)
    parser.add_argument(
        "--split-dir",
        default=os.environ.get(
            "MAF07_SPLIT_DIR", "results/splits_content_cleaned_conservative"
        ),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--per-ood-output", required=True)
    parser.add_argument("--score-sample-output", default="")
    parser.add_argument(
        "--score-sample-methods",
        default=(
            "rsn_class_std_huber_k150,rsn_class_std_huber_calibrated,"
            "knn_l2_pooled_k50_kth,nnguide_k10"
        ),
    )
    parser.add_argument("--score-sample-seed", type=int, default=0)
    parser.add_argument("--score-sample-backbone", default="dinov2_vitb14")
    parser.add_argument("--score-sample-id-set", default="cheetah|jaguar")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument(
        "--no-final-summary",
        action="store_true",
        help="Skip the shared summary write in parallel workers.",
    )
    args = parser.parse_args(argv)

    output = resolve_path(args.output)
    summary_output = resolve_path(args.summary_output)
    per_ood_output = resolve_path(args.per_ood_output)
    score_sample_output = (
        resolve_path(args.score_sample_output) if args.score_sample_output else None
    )
    if args.summarize_only:
        _summarize(output, summary_output)
        return 0

    classes = _classes()
    backbones = _parse_values(args.backbones)
    seeds = [int(value) for value in _parse_values(args.seeds)]
    available = _method_names()
    methods = available if args.methods == "all" else _parse_values(args.methods)
    invalid = sorted(set(methods) - set(available))
    if invalid:
        raise SystemExit(f"Unknown reviewer methods: {invalid}")
    sample_methods = set(_parse_values(args.score_sample_methods))
    spec_by_name = {spec.name: spec for spec in reviewer_factor_specs()}

    all_jobs = _jobs(backbones, seeds, classes, args.id_size, methods)
    done: set[str] = set()
    if output.exists() and output.stat().st_size > 0:
        done = set(pd.read_csv(output, usecols=["job_id"])["job_id"].astype(str))

    base_jobs = _base_jobs(backbones, seeds, classes, args.id_size)
    if args.worker_count > 1:
        start = len(base_jobs) * args.worker_index // args.worker_count
        end = len(base_jobs) * (args.worker_index + 1) // args.worker_count
        base_jobs = base_jobs.iloc[start:end].reset_index(drop=True)
    if args.max_base_jobs is not None:
        base_jobs = base_jobs.head(args.max_base_jobs)

    completed = 0
    for _, base in base_jobs.iterrows():
        jobs = all_jobs[
            (all_jobs["seed"] == int(base["seed"]))
            & (all_jobs["backbone"] == str(base["backbone"]))
            & (all_jobs["id_set"] == str(base["id_set"]))
        ]
        jobs = jobs[~jobs["job_id"].astype(str).isin(done)]
        if jobs.empty:
            continue

        id_classes = str(base["id_set"]).split("|")
        split = _load_split(int(base["seed"]), args.split_dir)
        eval_frame = make_ood_eval_frame(split, id_classes, protocol="fair")
        rows, features = feature_frame_for_split(eval_frame, str(base["backbone"]))
        train_mask = rows["role"] == "id_train"
        val_mask = rows["role"] == "id_val"
        eval_mask = rows["role"].isin(["id_test", "ood_test"])
        encoder = LabelEncoder().fit(rows.loc[train_mask, "class_name"])
        train_y = encoder.transform(rows.loc[train_mask, "class_name"])
        val_y = encoder.transform(rows.loc[val_mask, "class_name"])
        train_x = features[train_mask.to_numpy()]
        val_x = features[val_mask.to_numpy()]
        eval_x = features[eval_mask.to_numpy()]
        score_rows = rows.loc[
            eval_mask, ["image_id", "class_name", "role", "ood_label"]
        ].reset_index(drop=True)
        is_id = (score_rows["ood_label"].to_numpy() == 0).astype(int)
        class_bank_counts = np.bincount(train_y)

        requested_names = set(jobs["method"].astype(str))
        factor_specs = [
            spec_by_name[name]
            for name in requested_names
            if name in spec_by_name
        ]
        scores_by_method: dict[str, np.ndarray] = {}
        runtime_by_method: dict[str, float] = {}
        if factor_specs:
            started = time.perf_counter()
            factor_suite = FactorialNeighborSuite(
                train_x,
                train_y,
                device=os.environ.get("MAF07_REVIEWER_DEVICE"),
                score_batch=int(os.environ.get("MAF07_REVIEWER_SCORE_BATCH", "64")),
            )
            factor_scores = factor_suite.score_many(eval_x, factor_specs)
            shared_runtime = time.perf_counter() - started
            scores_by_method.update(factor_scores)
            for spec in factor_specs:
                runtime_by_method[spec.name] = shared_runtime

        needs_probe = bool(requested_names.intersection({"nnguide_k10", "nnguide_k50"}))
        baseline_suite = VerifiedBaselineSuite(train_x, train_y, val_x) if needs_probe else None
        if baseline_suite is not None:
            eval_logits = baseline_suite.probe.logits(eval_x)
            for k in (10, 50):
                name = f"nnguide_k{k}"
                if name not in requested_names:
                    continue
                started = time.perf_counter()
                scores_by_method[name] = nnguide_score(
                    train_x,
                    baseline_suite.train_logits,
                    eval_x,
                    eval_logits,
                    k=k,
                    batch_size=int(os.environ.get("MAF07_NNGUIDE_BATCH", "128")),
                    device=os.environ.get("MAF07_REVIEWER_DEVICE"),
                )
                runtime_by_method[name] = time.perf_counter() - started

        if "rsn_class_std_huber_calibrated" in requested_names:
            started = time.perf_counter()
            detector = DiagCARDDetector(
                k=150,
                delta=1.345,
                normalize=False,
                min_std=1e-3,
                device=os.environ.get("MAF07_REVIEWER_DEVICE"),
                score_batch=int(os.environ.get("MAF07_REVIEWER_SCORE_BATCH", "64")),
            ).fit(train_x, train_y, z_cal=val_x, y_cal=val_y)
            scores_by_method["rsn_class_std_huber_calibrated"] = detector.id_scores(
                eval_x, logits=None, delta=1.345, calibrated=True
            )
            runtime_by_method["rsn_class_std_huber_calibrated"] = (
                time.perf_counter() - started
            )

        sample_fold = _selected_sample_fold(args, base)
        for _, job in jobs.iterrows():
            method = str(job["method"])
            scores = np.asarray(scores_by_method[method], dtype=np.float64)
            if scores.shape != (len(eval_x),) or not np.isfinite(scores).all():
                raise RuntimeError(f"Invalid score vector for {method}: {scores.shape}")
            job_dict = job.to_dict()
            per_ood, macro = _per_ood_metrics(score_rows, scores, job_dict)
            metrics = ood_metrics(is_id, scores)
            result = {
                **job_dict,
                "n": int(len(scores)),
                "id_n": int(is_id.sum()),
                "ood_n": int((1 - is_id).sum()),
                "ood_prevalence": float((1 - is_id).mean()),
                "id_train_n": int(len(train_x)),
                "id_class_count": int(len(class_bank_counts)),
                "ood_class_count": int(len(classes) - len(class_bank_counts)),
                "min_class_bank_n": int(class_bank_counts.min()),
                "max_class_bank_n": int(class_bank_counts.max()),
                "mean_class_bank_n": float(class_bank_counts.mean()),
                "shared_or_method_runtime_sec": float(runtime_by_method[method]),
                "score_sha256": score_fingerprint(scores),
                "AUPR_OUT_BALANCED": balanced_aupr_out(is_id, scores),
                **macro,
                **metrics,
            }
            append_csv_rows(pd.DataFrame([result]), output)
            append_csv_rows(pd.DataFrame(per_ood), per_ood_output)
            if sample_fold and score_sample_output is not None and method in sample_methods:
                sample = score_rows.copy()
                sample.insert(0, "job_id", str(job["job_id"]))
                sample.insert(1, "method", method)
                sample["score"] = scores
                append_csv_rows(sample, score_sample_output)
            completed += 1
            print(
                json.dumps(
                    {
                        "completed": completed,
                        "id_size": args.id_size,
                        "id_set": str(base["id_set"]),
                        "method": method,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    if not args.no_final_summary:
        _summarize(output, summary_output)
    print(
        json.dumps(
            {"completed": completed, "output": str(output), "summary": str(summary_output)},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
