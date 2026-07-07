#!/usr/bin/env python
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from maf07 import TARGET_CLASSES
from maf07.config import load_yaml, resolve_path
from maf07.features import feature_frame_for_split
from maf07.jobs import _stable_job_id, append_csv_rows
from maf07.methods.rsn import rsn_from_env
from maf07.runner import _load_split
from maf07.splits import make_ood_eval_frame


def _parse_csv_values(raw: str | None, default: list[str]) -> list[str]:
    if not raw:
        return default
    return [item.strip() for chunk in raw.split(",") for item in chunk.split() if item.strip()]


def _base_jobs(
    backbones: list[str],
    seeds: list[int],
    classes: list[str],
    id_sizes: list[int],
    protocol: str,
) -> pd.DataFrame:
    rows = []
    for seed, backbone, id_size in itertools.product(seeds, backbones, id_sizes):
        for id_set in itertools.combinations(classes, id_size):
            row = {
                "job_kind": "rsn_ks",
                "protocol": protocol,
                "seed": int(seed),
                "backbone": backbone,
                "method": "rsn",
                "variant": "main",
                "id_size": int(id_size),
                "id_set": "|".join(id_set),
                "ood_set": "|".join(c for c in classes if c not in set(id_set)),
            }
            row["job_id"] = _stable_job_id(row)
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["backbone", "seed", "id_size", "id_set"]).reset_index(drop=True)


def _class_validation_distances(detector, class_names: np.ndarray) -> dict[str, np.ndarray]:
    if detector.cal_x_ is None or detector.cal_y_ is None:
        raise RuntimeError("RSN detector has no calibration features")
    out: dict[str, np.ndarray] = {}
    for class_id, class_name in enumerate(class_names):
        rows_np = np.where(detector.cal_y_ == int(class_id))[0]
        if len(rows_np) == 0:
            continue
        torch = detector._torch()
        rows = torch.as_tensor(rows_np, dtype=torch.long, device=detector.cal_x_.device)
        distances = detector._distance_class(detector.cal_x_[rows], int(class_id), delta=float(detector.delta))
        out[str(class_name)] = distances.detach().cpu().numpy().astype(float)
    return out


def _ks_rows(distances: dict[str, np.ndarray]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for class_a, class_b in itertools.combinations(sorted(distances), 2):
        a = distances[class_a]
        b = distances[class_b]
        stat = ks_2samp(a, b, alternative="two-sided", mode="auto")
        rows.append(
            {
                "class_a": class_a,
                "class_b": class_b,
                "n_a": int(len(a)),
                "n_b": int(len(b)),
                "mean_a": float(np.mean(a)),
                "mean_b": float(np.mean(b)),
                "std_a": float(np.std(a)),
                "std_b": float(np.std(b)),
                "ks_stat": float(stat.statistic),
                "ks_pvalue": float(stat.pvalue),
            }
        )
    return rows


def _summary_from_pairs(pairs: list[dict[str, object]]) -> dict[str, object]:
    if not pairs:
        return {
            "pair_count": 0,
            "ks_mean": np.nan,
            "ks_median": np.nan,
            "ks_max": np.nan,
            "mean_distance_cv": np.nan,
        }
    stats = np.asarray([row["ks_stat"] for row in pairs], dtype=float)
    means = []
    for row in pairs:
        means.extend([float(row["mean_a"]), float(row["mean_b"])])
    means_arr = np.asarray(means, dtype=float)
    return {
        "pair_count": int(len(pairs)),
        "ks_mean": float(np.mean(stats)),
        "ks_median": float(np.median(stats)),
        "ks_max": float(np.max(stats)),
        "mean_distance_cv": float(np.std(means_arr) / max(abs(float(np.mean(means_arr))), 1e-12)),
    }


def _cdf_rows(distances: dict[str, np.ndarray]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for class_name, values in sorted(distances.items()):
        vals = np.sort(np.asarray(values, dtype=float))
        if len(vals) == 0:
            continue
        cdf = np.arange(1, len(vals) + 1, dtype=float) / float(len(vals))
        for distance, prob in zip(vals, cdf, strict=True):
            rows.append({"class_name": class_name, "distance": float(distance), "cdf": float(prob)})
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default="fair", choices=["fair", "oracle"])
    parser.add_argument("--id-sizes", default=os.environ.get("MAF07_RSN_KS_ID_SIZES", "2,3,4,5,6,7"))
    parser.add_argument("--backbones", default=os.environ.get("MAF07_BACKBONES", "dinov2_vitb14,dinov2_vitl14"))
    parser.add_argument("--seeds", default=os.environ.get("MAF07_SEEDS", "0,1,2"))
    parser.add_argument("--worker-index", type=int, default=int(os.environ.get("MAF07_JOB_WORKER_INDEX", "0")))
    parser.add_argument("--worker-count", type=int, default=int(os.environ.get("MAF07_JOB_WORKER_COUNT", "1")))
    parser.add_argument("--max-base-jobs", type=int, default=None)
    parser.add_argument("--output", default="results/analysis/rsn_validation_ks_summary.csv")
    parser.add_argument("--pairs-output", default="results/analysis/rsn_validation_ks_pairs.csv")
    parser.add_argument("--split-dir", default=os.environ.get("MAF07_SPLIT_DIR", "results/splits"))
    parser.add_argument("--cdf-output", default="")
    parser.add_argument("--cdf-id-size", type=int, default=7)
    parser.add_argument("--cdf-seed", type=int, default=0)
    parser.add_argument("--cdf-backbone", default="dinov2_vitl14")
    args = parser.parse_args(argv)

    dcfg = load_yaml("configs/dataset.yaml")
    classes = list(dcfg.get("classes", TARGET_CLASSES))
    backbones = _parse_csv_values(args.backbones, ["dinov2_vitb14", "dinov2_vitl14"])
    seeds = [int(v) for v in _parse_csv_values(args.seeds, ["0", "1", "2"])]
    id_sizes = [int(v) for v in _parse_csv_values(args.id_sizes, ["2", "3", "4", "5", "6", "7"])]
    jobs = _base_jobs(backbones, seeds, classes, id_sizes, args.protocol)
    if args.worker_count > 1:
        start = len(jobs) * args.worker_index // args.worker_count
        end = len(jobs) * (args.worker_index + 1) // args.worker_count
        jobs = jobs.iloc[start:end].reset_index(drop=True)
    if args.max_base_jobs is not None:
        jobs = jobs.head(args.max_base_jobs)

    output = resolve_path(args.output)
    pairs_output = resolve_path(args.pairs_output)
    done: set[str] = set()
    if output.exists():
        existing = pd.read_csv(output, usecols=["job_id"])
        done = set(existing["job_id"].astype(str))

    completed = 0
    for _, job in jobs.iterrows():
        if str(job["job_id"]) in done:
            continue
        split = _load_split(int(job["seed"]), args.split_dir)
        id_classes = str(job["id_set"]).split("|")
        eval_frame = make_ood_eval_frame(split, id_classes, protocol=str(job["protocol"]))
        rows, features = feature_frame_for_split(eval_frame, str(job["backbone"]))
        train_mask = rows["role"] == "id_train"
        val_mask = rows["role"] == "id_val"
        enc = LabelEncoder().fit(rows.loc[train_mask, "class_name"])
        train_y = enc.transform(rows.loc[train_mask, "class_name"])
        val_y = enc.transform(rows.loc[val_mask, "class_name"])
        train_x = features[train_mask.to_numpy()]
        val_x = features[val_mask.to_numpy()]
        detector = rsn_from_env().fit(train_x, train_y, z_cal=val_x, y_cal=val_y)
        distances = _class_validation_distances(detector, enc.classes_)
        pair_rows = [{**job.to_dict(), **row} for row in _ks_rows(distances)]
        if pair_rows:
            append_csv_rows(pd.DataFrame(pair_rows), pairs_output)
        summary = {**job.to_dict(), **_summary_from_pairs(pair_rows)}
        append_csv_rows(pd.DataFrame([summary]), output)
        if args.cdf_output and int(job["seed"]) == int(args.cdf_seed) and int(job["id_size"]) == int(
            args.cdf_id_size
        ) and str(job["backbone"]) == str(args.cdf_backbone):
            cdf_output = resolve_path(args.cdf_output)
            if not cdf_output.exists() or cdf_output.stat().st_size == 0:
                cdf_output.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame([{**job.to_dict(), **row} for row in _cdf_rows(distances)]).to_csv(
                    cdf_output,
                    index=False,
                )
        completed += 1
        print(json.dumps({"completed": completed, "job_id": str(job["job_id"])}, sort_keys=True), flush=True)

    print(json.dumps({"completed": completed, "output": str(output), "pairs_output": str(pairs_output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
