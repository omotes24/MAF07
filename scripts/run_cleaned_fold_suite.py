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
from maf07.methods.diagcard import diagcard_from_env
from maf07.metrics import ood_metrics
from maf07.runner import _load_split, _score_ood_group_method
from maf07.splits import make_ood_eval_frame


DIAG_METHODS = {"diag_raw", "diag_calib", "diag_huber_raw", "diag_huber_calib"}


def _parse_values(raw: str | None, default: list[str]) -> list[str]:
    if not raw:
        return default
    return [item.strip() for chunk in raw.split(",") for item in chunk.split() if item.strip()]


def _variant_config(name: str) -> tuple[float | None, bool]:
    if name == "diag_raw":
        return None, False
    if name == "diag_calib":
        return None, True
    if name == "diag_huber_raw":
        return float(os.environ.get("MAF07_DIAGCARD_DELTA", "1.345")), False
    if name == "diag_huber_calib":
        return float(os.environ.get("MAF07_DIAGCARD_DELTA", "1.345")), True
    raise ValueError(f"Unknown DiagCARD method: {name}")


def _class_order() -> list[str]:
    dcfg = load_yaml("configs/dataset.yaml")
    return list(dcfg.get("classes", TARGET_CLASSES))


def _id_set_ids(classes: list[str], id_size: int) -> dict[str, int]:
    return {"|".join(id_set): idx for idx, id_set in enumerate(itertools.combinations(classes, id_size))}


def _jobs(
    backbones: list[str],
    seeds: list[int],
    classes: list[str],
    id_size: int,
    protocol: str,
    methods: list[str],
) -> pd.DataFrame:
    id_ids = _id_set_ids(classes, id_size)
    rows: list[dict[str, object]] = []
    for seed, backbone, id_set, method in itertools.product(
        seeds,
        backbones,
        itertools.combinations(classes, id_size),
        methods,
    ):
        id_text = "|".join(id_set)
        id_classes = set(id_set)
        row = {
            "job_kind": "cleaned_fold_suite",
            "scope": protocol,
            "protocol": protocol,
            "seed": int(seed),
            "backbone": backbone,
            "id_size": int(id_size),
            "id_set_id": int(id_ids[id_text]),
            "id_set": id_text,
            "ood_set": "|".join(c for c in classes if c not in id_classes),
            "method": method,
            "variant": "main",
        }
        row["job_id"] = _stable_job_id(row)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["backbone", "seed", "id_set_id", "method"]).reset_index(drop=True)


def _base_jobs(backbones: list[str], seeds: list[int], classes: list[str], id_size: int) -> pd.DataFrame:
    id_ids = _id_set_ids(classes, id_size)
    rows = []
    for seed, backbone, id_set in itertools.product(seeds, backbones, itertools.combinations(classes, id_size)):
        id_text = "|".join(id_set)
        rows.append(
            {
                "seed": int(seed),
                "backbone": backbone,
                "id_size": int(id_size),
                "id_set_id": int(id_ids[id_text]),
                "id_set": id_text,
            }
        )
    return pd.DataFrame(rows).sort_values(["backbone", "seed", "id_set_id"]).reset_index(drop=True)


def _per_ood_rows(base_rows: pd.DataFrame, scores, job: dict[str, object]) -> list[dict[str, object]]:
    frame = base_rows.copy()
    frame["score"] = scores
    frame["is_id"] = (frame["ood_label"].to_numpy() == 0).astype(int)
    id_rows = frame[frame["is_id"] == 1]
    out: list[dict[str, object]] = []
    for ood_class, ood_rows in frame[frame["is_id"] == 0].groupby("class_name", sort=True):
        block = pd.concat([id_rows, ood_rows], ignore_index=True)
        out.append(
            {
                **job,
                "ood_class": str(ood_class),
                "id_n": int(len(id_rows)),
                "ood_n": int(len(ood_rows)),
                **ood_metrics(block["is_id"].to_numpy(), block["score"].to_numpy()),
            }
        )
    return out


def _summarize(results_path: Path, summary_path: Path) -> None:
    if not results_path.exists():
        return
    df = pd.read_csv(results_path)
    if df.empty:
        return
    df = df.drop_duplicates("job_id", keep="last")
    rows = []
    for id_size, sub0 in df.groupby("id_size", dropna=False):
        by_backbone = (
            sub0.groupby(["backbone", "id_size", "method"], dropna=False)
            .agg(
                n=("AUROC", "size"),
                AUROC=("AUROC", "mean"),
                FPR95=("FPR95", "mean"),
                AUPR_OUT=("AUPR_OUT", "mean"),
            )
            .reset_index()
        )
        allg = (
            sub0.groupby(["id_size", "method"], dropna=False)
            .agg(
                n=("AUROC", "size"),
                AUROC=("AUROC", "mean"),
                FPR95=("FPR95", "mean"),
                AUPR_OUT=("AUPR_OUT", "mean"),
            )
            .reset_index()
        )
        allg.insert(0, "backbone", "ALL")
        rows.extend(pd.concat([by_backbone, allg], ignore_index=True).to_dict("records"))
    summary = pd.DataFrame(rows)
    for col in ["AUROC", "FPR95", "AUPR_OUT"]:
        summary[col] = summary[col].round(6)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.sort_values(["id_size", "backbone", "method"]).to_csv(summary_path, index=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--id-size", type=int, required=True)
    parser.add_argument("--protocol", default="fair", choices=["fair", "oracle"])
    parser.add_argument("--methods", required=True)
    parser.add_argument("--per-ood-methods", default="")
    parser.add_argument("--backbones", default=os.environ.get("MAF07_BACKBONES", "dinov2_vitb14,dinov2_vitl14"))
    parser.add_argument("--seeds", default=os.environ.get("MAF07_SEEDS", "0,1,2"))
    parser.add_argument("--worker-index", type=int, default=int(os.environ.get("MAF07_JOB_WORKER_INDEX", "0")))
    parser.add_argument("--worker-count", type=int, default=int(os.environ.get("MAF07_JOB_WORKER_COUNT", "1")))
    parser.add_argument("--max-base-jobs", type=int, default=None)
    parser.add_argument("--split-dir", default=os.environ.get("MAF07_SPLIT_DIR", "results/splits"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--per-ood-output", default="")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args(argv)

    output = resolve_path(args.output)
    summary_output = resolve_path(args.summary_output)
    per_ood_output = resolve_path(args.per_ood_output) if args.per_ood_output else None
    if args.summarize_only:
        _summarize(output, summary_output)
        return 0

    classes = _class_order()
    backbones = _parse_values(args.backbones, ["dinov2_vitb14", "dinov2_vitl14"])
    seeds = [int(v) for v in _parse_values(args.seeds, ["0", "1", "2"])]
    methods = [m.lower() for m in _parse_values(args.methods, [])]
    per_ood_methods = {m.lower() for m in _parse_values(args.per_ood_methods, [])}
    forbidden = {"card", "cqs", "lar"}
    forbidden_seen = forbidden.intersection(methods)
    if forbidden_seen:
        raise SystemExit(f"Forbidden discarded methods requested: {sorted(forbidden_seen)}")

    all_jobs = _jobs(backbones, seeds, classes, args.id_size, args.protocol, methods)
    done: set[str] = set()
    if output.exists() and output.stat().st_size > 0:
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
        ].copy()
        job_rows = job_rows[~job_rows["job_id"].astype(str).isin(done)]
        if job_rows.empty:
            continue

        split = _load_split(int(base["seed"]), args.split_dir)
        eval_frame = make_ood_eval_frame(split, id_classes, protocol=args.protocol)
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
        score_base = rows.loc[eval_mask, ["image_id", "class_name", "split", "role", "ood_label"]].copy()
        is_id = (score_base["ood_label"].to_numpy() == 0).astype(int)
        cache: dict[str, object] = {}
        diag_detector = None

        for _, job in job_rows.iterrows():
            method = str(job["method"]).lower()
            job_dict = job.to_dict()
            if method in DIAG_METHODS:
                if diag_detector is None:
                    diag_detector = diagcard_from_env().fit(train_x, train_y, z_cal=val_x, y_cal=val_y)
                delta, calibrated = _variant_config(method)
                scores = diag_detector.id_scores(eval_x, delta=delta, calibrated=calibrated)
            else:
                scores = _score_ood_group_method(method, train_x, train_y, val_x, val_y, eval_x, cache)
            metrics = ood_metrics(is_id, scores)
            result = {
                **job_dict,
                "n": int(len(is_id)),
                "id_n": int(is_id.sum()),
                "ood_n": int((1 - is_id).sum()),
                **metrics,
            }
            append_csv_rows(pd.DataFrame([result]), output)
            if per_ood_output is not None and method in per_ood_methods:
                append_csv_rows(pd.DataFrame(_per_ood_rows(score_base, scores, job_dict)), per_ood_output)
            completed += 1
            print(
                json.dumps(
                    {"completed": completed, "job_id": str(job["job_id"]), "method": method},
                    sort_keys=True,
                ),
                flush=True,
            )

    _summarize(output, summary_output)
    print(json.dumps({"completed": completed, "output": str(output), "summary": str(summary_output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
