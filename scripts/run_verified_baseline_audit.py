#!/usr/bin/env python
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from maf07 import TARGET_CLASSES
from maf07.config import load_yaml, resolve_path
from maf07.features import feature_frame_for_split
from maf07.jobs import _stable_job_id, append_csv_rows
from maf07.methods.diagcard import DiagCARDDetector
from maf07.methods.verified_baselines import (
    UNSUPPORTED_DINOV2_METHODS,
    VERIFIED_METHODS,
    VerifiedBaselineSuite,
    score_fingerprint,
)
from maf07.metrics import ood_metrics
from maf07.runner import _load_split
from maf07.splits import make_ood_eval_frame

EXPECTED_CLEAN_COUNTS = {
    "cheetah": 11349,
    "jaguar": 10452,
    "leopard": 10984,
    "lion": 15845,
    "ocelot": 12342,
    "puma": 18763,
    "serval": 9149,
    "tiger": 16670,
}

DEFAULT_METHODS = (
    "rsn_reported,rsn_paper,knn,mahalanobis,mahalanobispp,rmd,"
    "msp,entropy,energy,maxlogit,gen,gradnorm,kl_matching,vim,react,"
    "ashp,ashb,ashs,dice,scale,nci,openmax"
)


def _parse_values(raw: str) -> list[str]:
    return [item.strip() for chunk in raw.split(",") for item in chunk.split() if item.strip()]


def _verify_manifest(path: str | Path) -> None:
    manifest = pd.read_csv(resolve_path(path))
    if "class_name" not in manifest:
        raise SystemExit(f"Manifest has no class_name column: {path}")
    actual = manifest.groupby("class_name").size().astype(int).to_dict()
    if actual != EXPECTED_CLEAN_COUNTS:
        raise SystemExit(
            "Cleaned dataset count mismatch: "
            + json.dumps({"expected": EXPECTED_CLEAN_COUNTS, "actual": actual}, sort_keys=True)
        )


def _class_order() -> list[str]:
    config = load_yaml("configs/dataset.yaml")
    return list(config.get("classes", TARGET_CLASSES))


def _jobs(
    backbones: list[str],
    seeds: list[int],
    classes: list[str],
    id_size: int,
    methods: list[str],
) -> pd.DataFrame:
    id_sets = list(itertools.combinations(classes, id_size))
    id_set_ids = {"|".join(values): idx for idx, values in enumerate(id_sets)}
    rows: list[dict[str, object]] = []
    for seed, backbone, id_set, method in itertools.product(seeds, backbones, id_sets, methods):
        id_text = "|".join(id_set)
        id_classes = set(id_set)
        row = {
            "job_kind": "verified_baseline_audit",
            "scope": "fair",
            "protocol": "fair",
            "seed": int(seed),
            "backbone": backbone,
            "id_size": int(id_size),
            "id_set_id": int(id_set_ids[id_text]),
            "id_set": id_text,
            "ood_set": "|".join(cls for cls in classes if cls not in id_classes),
            "method": method,
            "implementation_version": "verified_v1",
        }
        row["job_id"] = _stable_job_id(row)
        rows.append(row)
    return (
        pd.DataFrame(rows)
        .sort_values(["backbone", "seed", "id_set_id", "method"])
        .reset_index(drop=True)
    )


def _base_jobs(
    backbones: list[str],
    seeds: list[int],
    classes: list[str],
    id_size: int,
) -> pd.DataFrame:
    rows = []
    for seed, backbone, id_set in itertools.product(
        seeds,
        backbones,
        itertools.combinations(classes, id_size),
    ):
        rows.append(
            {
                "seed": int(seed),
                "backbone": backbone,
                "id_set": "|".join(id_set),
            }
        )
    return pd.DataFrame(rows).sort_values(["backbone", "seed", "id_set"]).reset_index(drop=True)


def _summarize(results_path: Path, summary_path: Path, equivalence_path: Path) -> None:
    if not results_path.exists() or results_path.stat().st_size == 0:
        return
    frame = pd.read_csv(results_path).drop_duplicates("job_id", keep="last")
    group_cols = ["backbone", "id_size", "method", "implementation_version"]
    summary = (
        frame.groupby(group_cols, dropna=False)
        .agg(
            n=("AUROC", "size"),
            AUROC=("AUROC", "mean"),
            FPR95=("FPR95", "mean"),
            AUPR_OUT=("AUPR_OUT", "mean"),
        )
        .reset_index()
    )
    pooled = (
        frame.groupby(["id_size", "method", "implementation_version"], dropna=False)
        .agg(
            n=("AUROC", "size"),
            AUROC=("AUROC", "mean"),
            FPR95=("FPR95", "mean"),
            AUPR_OUT=("AUPR_OUT", "mean"),
        )
        .reset_index()
    )
    pooled.insert(0, "backbone", "ALL")
    summary = pd.concat([summary, pooled], ignore_index=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.sort_values(["id_size", "backbone", "AUROC"], ascending=[True, True, False]).to_csv(
        summary_path,
        index=False,
    )

    keys = ["backbone", "seed", "id_set_id"]
    methods = sorted(frame["method"].unique())
    equivalence_rows = []
    for method_a, method_b in itertools.combinations(methods, 2):
        value_cols = ["score_sha256", "AUROC", "FPR95", "AUPR_OUT"]
        left = frame[frame["method"] == method_a][keys + value_cols]
        right = frame[frame["method"] == method_b][keys + value_cols]
        merged = left.merge(right, on=keys, suffixes=("_a", "_b"))
        if merged.empty:
            continue
        equivalence_rows.append(
            {
                "method_a": method_a,
                "method_b": method_b,
                "n_common": int(len(merged)),
                "all_score_fingerprints_equal": bool(
                    (merged["score_sha256_a"] == merged["score_sha256_b"]).all()
                ),
                "all_metrics_equal": bool(
                    np.array_equal(
                        merged[["AUROC_a", "FPR95_a", "AUPR_OUT_a"]].to_numpy(),
                        merged[["AUROC_b", "FPR95_b", "AUPR_OUT_b"]].to_numpy(),
                    )
                ),
            }
        )
    pd.DataFrame(equivalence_rows).to_csv(equivalence_path, index=False)


def _rsn_score(
    method: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    eval_x: np.ndarray,
    eval_logits: np.ndarray,
) -> np.ndarray:
    if method == "rsn_reported":
        detector = DiagCARDDetector(
            k=150,
            delta=1.345,
            topq=3,
            normalize=False,
            min_std=1e-3,
            device=None,
            score_batch=96,
        ).fit(train_x, train_y, z_cal=val_x, y_cal=val_y)
        return detector.id_scores(eval_x, logits=None, delta=1.345, calibrated=False)
    if method == "rsn_paper":
        detector = DiagCARDDetector(
            k=10,
            delta=1.345,
            topq=3,
            normalize=True,
            min_std=1e-3,
            device=None,
            score_batch=96,
        ).fit(train_x, train_y, z_cal=val_x, y_cal=val_y)
        return detector.id_scores(eval_x, logits=eval_logits, delta=1.345, calibrated=False)
    raise ValueError(f"Unknown RSN audit profile: {method}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--id-size", type=int, default=2)
    parser.add_argument("--methods", default=DEFAULT_METHODS)
    parser.add_argument("--backbones", default="dinov2_vitb14,dinov2_vitl14")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--worker-count", type=int, default=1)
    parser.add_argument("--max-base-jobs", type=int, default=None)
    parser.add_argument("--split-dir", default="results/splits_content_cleaned_conservative")
    parser.add_argument("--manifest", default="results/manifest.content_cleaned_conservative.csv")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--equivalence-output", required=True)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args(argv)

    output = resolve_path(args.output)
    summary_output = resolve_path(args.summary_output)
    equivalence_output = resolve_path(args.equivalence_output)
    if args.summarize_only:
        _summarize(output, summary_output, equivalence_output)
        return 0

    _verify_manifest(args.manifest)
    classes = _class_order()
    backbones = _parse_values(args.backbones)
    seeds = [int(value) for value in _parse_values(args.seeds)]
    methods = [value.lower() for value in _parse_values(args.methods)]
    allowed = set(VERIFIED_METHODS) | {"rsn_reported", "rsn_paper"}
    invalid = sorted(set(methods) - allowed)
    if invalid:
        unsupported = sorted(
            set(invalid).intersection(UNSUPPORTED_DINOV2_METHODS | {"mah_mindist"})
        )
        raise SystemExit(
            f"Invalid verified methods: {invalid}. "
            f"Unsupported/mislabelled DINOv2 methods: {unsupported}"
        )

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
        id_classes = str(base["id_set"]).split("|")
        jobs = all_jobs[
            (all_jobs["seed"] == int(base["seed"]))
            & (all_jobs["backbone"] == str(base["backbone"]))
            & (all_jobs["id_set"] == str(base["id_set"]))
        ]
        jobs = jobs[~jobs["job_id"].astype(str).isin(done)]
        if jobs.empty:
            continue

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
        is_id = (rows.loc[eval_mask, "ood_label"].to_numpy() == 0).astype(int)

        suite = VerifiedBaselineSuite(train_x, train_y, val_x)
        eval_logits = suite.probe.logits(eval_x)
        for _, job in jobs.iterrows():
            method = str(job["method"])
            if method.startswith("rsn_"):
                scores = _rsn_score(method, train_x, train_y, val_x, val_y, eval_x, eval_logits)
            else:
                scores = suite.score(method, eval_x)
            scores = np.asarray(scores, dtype=np.float64)
            if scores.shape != (len(eval_x),) or not np.isfinite(scores).all():
                raise RuntimeError(f"Invalid score vector for {method}: shape={scores.shape}")
            result = {
                **job.to_dict(),
                "n": int(len(scores)),
                "id_n": int(is_id.sum()),
                "ood_n": int((1 - is_id).sum()),
                "score_sha256": score_fingerprint(scores),
                **ood_metrics(is_id, scores),
            }
            append_csv_rows(pd.DataFrame([result]), output)
            completed += 1
            print(
                json.dumps(
                    {"completed": completed, "job_id": str(job["job_id"]), "method": method},
                    sort_keys=True,
                ),
                flush=True,
            )

    _summarize(output, summary_output, equivalence_output)
    print(json.dumps({"completed": completed, "output": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
