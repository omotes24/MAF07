#!/usr/bin/env python3
"""Evaluate frozen PULSE and RC-MSPS on the locked untouched final suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from audit_symmetric_support_proxy import metric_row
from final_lock import artifact_path, sha256_file, verify_locked_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--locked-config", type=Path, required=True)
    return parser.parse_args()


def load_scores(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {
            "pulse": data["pulse_ood_score"].astype(np.float64),
            "rc": data["rc_msps_confidence"].astype(np.float64),
        }


def load_dataset_scores(
    lock: dict[str, Any], dataset: str
) -> dict[str, np.ndarray]:
    root = Path(lock["outputs"]["score_shard_dir"])
    shards = []
    for rank in range(int(lock["execution"]["num_shards"])):
        path = root / f"{dataset}.rank{rank}.npz"
        manifest_path = path.with_suffix(".manifest.json")
        if not path.is_file() or not manifest_path.is_file():
            raise FileNotFoundError(f"missing final score shard or manifest: {path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["output_sha256"] != sha256_file(path):
            raise RuntimeError(f"final score shard changed after extraction: {path}")
        shards.append(load_scores(path))
    return {
        name: np.concatenate([shard[name] for shard in shards])
        for name in ("pulse", "rc")
    }


def load_dataset_paths(lock: dict[str, Any], dataset: str) -> np.ndarray:
    root = Path(lock["outputs"]["score_shard_dir"])
    paths = []
    for rank in range(int(lock["execution"]["num_shards"])):
        path = root / f"{dataset}.rank{rank}.jsonl"
        paths.extend(
            json.loads(line)["relative_path"]
            for line in path.read_text(encoding="utf-8").splitlines()
        )
    return np.asarray(paths, dtype=str)


def load_id_paths(lock: dict[str, Any]) -> np.ndarray:
    paths = []
    for artifact_name in lock["id_reference"]["path_artifacts"]:
        path = artifact_path(lock, artifact_name)
        paths.extend(
            json.loads(line)["path"]
            for line in path.read_text(encoding="utf-8").splitlines()
        )
    return np.asarray(paths, dtype=str)


def append_cases(
    rows: list[dict[str, Any]],
    *,
    method: str,
    split: str,
    case_type: str,
    paths: np.ndarray,
    scores: np.ndarray,
    indices: np.ndarray,
    threshold: float,
    ascending: bool,
    limit: int = 50,
) -> None:
    order = indices[np.argsort(scores[indices])]
    if not ascending:
        order = order[::-1]
    for rank, index in enumerate(order[:limit], 1):
        rows.append(
            {
                "method": method,
                "split": split,
                "case_type": case_type,
                "rank": rank,
                "path": paths[index],
                "score": scores[index],
                "decision_threshold": threshold,
            }
        )


def main() -> None:
    args = parse_args()
    lock, config_sha = verify_locked_config(args.locked_config)
    id_scores = load_scores(artifact_path(lock, "id_scores"))
    id_paths = load_id_paths(lock)
    if len(id_paths) != len(id_scores["pulse"]):
        raise RuntimeError("ID score and path manifests are not aligned")
    rows = []
    all_scores: dict[str, dict[str, np.ndarray]] = {}
    for spec in lock["final_suite"]["datasets"]:
        key = str(spec["key"])
        scores = load_dataset_scores(lock, key)
        all_scores[str(spec["name"])] = scores
        if len(scores["pulse"]) != int(spec["expected_images"]):
            raise RuntimeError(
                f"{key}: scored {len(scores['pulse'])}, expected {spec['expected_images']}"
            )
        rows.extend(
            [
                {
                    "method": "PULSE",
                    "dataset": str(spec["name"]),
                    "n_id": len(id_scores["pulse"]),
                    "n_ood": len(scores["pulse"]),
                    **metric_row(-id_scores["pulse"], -scores["pulse"]),
                },
                {
                    "method": "RC-MSPS",
                    "dataset": str(spec["name"]),
                    "n_id": len(id_scores["rc"]),
                    "n_ood": len(scores["rc"]),
                    **metric_row(id_scores["rc"], scores["rc"]),
                },
            ]
        )

    per_dataset = pd.DataFrame(rows)
    summary = per_dataset.groupby("method", as_index=False).agg(
        Macro_AUROC=("AUROC", "mean"),
        Macro_FPR95=("FPR95", "mean"),
        Macro_AUPR_OUT=("AUPR_OUT", "mean"),
    )
    pulse = summary.set_index("method").loc["PULSE"]
    rc = summary.set_index("method").loc["RC-MSPS"]
    comparison = pd.DataFrame(
        [
            {
                "comparison": "PULSE_minus_RC-MSPS",
                "Macro_AUROC_diff": pulse["Macro_AUROC"] - rc["Macro_AUROC"],
                "Macro_FPR95_diff": pulse["Macro_FPR95"] - rc["Macro_FPR95"],
                "Macro_AUPR_OUT_diff": pulse["Macro_AUPR_OUT"] - rc["Macro_AUPR_OUT"],
            }
        ]
    )
    worst = (
        per_dataset.groupby("method", as_index=False)
        .agg(Worst_Dataset_AUROC=("AUROC", "min"))
        .merge(
            per_dataset.loc[
                per_dataset.groupby("method")["AUROC"].idxmin(),
                ["method", "dataset"],
            ].rename(columns={"dataset": "Worst_Dataset"}),
            on="method",
            how="left",
        )
    )

    pulse_threshold = float(np.quantile(id_scores["pulse"], 0.95))
    rc_threshold = float(np.quantile(id_scores["rc"], 0.05))
    error_rows, case_rows = [], []
    pulse_id_reject = np.flatnonzero(id_scores["pulse"] > pulse_threshold)
    rc_id_reject = np.flatnonzero(id_scores["rc"] < rc_threshold)
    append_cases(
        case_rows,
        method="PULSE",
        split="ImageNet-1K-ID",
        case_type="false_reject",
        paths=id_paths,
        scores=id_scores["pulse"],
        indices=pulse_id_reject,
        threshold=pulse_threshold,
        ascending=False,
    )
    append_cases(
        case_rows,
        method="RC-MSPS",
        split="ImageNet-1K-ID",
        case_type="false_reject",
        paths=id_paths,
        scores=id_scores["rc"],
        indices=rc_id_reject,
        threshold=rc_threshold,
        ascending=True,
    )
    for spec in lock["final_suite"]["datasets"]:
        name, key = str(spec["name"]), str(spec["key"])
        scores = all_scores[name]
        paths = load_dataset_paths(lock, key)
        if len(paths) != len(scores["pulse"]):
            raise RuntimeError(f"{name}: final score and path manifests are not aligned")
        pulse_accept = np.flatnonzero(scores["pulse"] <= pulse_threshold)
        rc_accept = np.flatnonzero(scores["rc"] >= rc_threshold)
        for method, accepted, rejected, threshold in (
            ("PULSE", pulse_accept, pulse_id_reject, pulse_threshold),
            ("RC-MSPS", rc_accept, rc_id_reject, rc_threshold),
        ):
            error_rows.append(
                {
                    "dataset": name,
                    "method": method,
                    "id_acceptance_target": 0.95,
                    "decision_threshold": threshold,
                    "ood_false_accepts": len(accepted),
                    "ood_false_accept_rate": len(accepted) / len(paths),
                    "id_false_rejects": len(rejected),
                    "id_false_reject_rate": len(rejected) / len(id_paths),
                }
            )
        error_rows.append(
            {
                "dataset": name,
                "method": "PULSE_and_RC-MSPS_overlap",
                "id_acceptance_target": 0.95,
                "decision_threshold": np.nan,
                "ood_false_accepts": len(np.intersect1d(pulse_accept, rc_accept)),
                "ood_false_accept_rate": len(np.intersect1d(pulse_accept, rc_accept))
                / len(paths),
                "id_false_rejects": len(
                    np.intersect1d(pulse_id_reject, rc_id_reject)
                ),
                "id_false_reject_rate": len(
                    np.intersect1d(pulse_id_reject, rc_id_reject)
                )
                / len(id_paths),
            }
        )
        append_cases(
            case_rows,
            method="PULSE",
            split=name,
            case_type="false_accept",
            paths=paths,
            scores=scores["pulse"],
            indices=pulse_accept,
            threshold=pulse_threshold,
            ascending=True,
        )
        append_cases(
            case_rows,
            method="RC-MSPS",
            split=name,
            case_type="false_accept",
            paths=paths,
            scores=scores["rc"],
            indices=rc_accept,
            threshold=rc_threshold,
            ascending=False,
        )

    runtime_rows = []
    for spec in lock["final_suite"]["datasets"]:
        key = str(spec["key"])
        manifests = [
            json.loads(
                (
                    Path(lock["outputs"]["score_shard_dir"])
                    / f"{key}.rank{rank}.manifest.json"
                ).read_text(encoding="utf-8")
            )
            for rank in range(int(lock["execution"]["num_shards"]))
        ]
        runtime_rows.append(
            {
                "dataset": str(spec["name"]),
                "images": int(spec["expected_images"]),
                "parallel_wall_seconds": max(row["runtime_seconds"] for row in manifests),
                "aggregate_gpu_process_seconds": sum(
                    row["runtime_seconds"] for row in manifests
                ),
                "images_per_wall_second": int(spec["expected_images"])
                / max(row["runtime_seconds"] for row in manifests),
            }
        )

    resource = pd.DataFrame(
        [
            {
                "method": "PULSE",
                "compact_index_bytes": int(
                    lock["resource_accounting"]["compact_index_bytes"]
                ),
                "final_prototypes_bytes": int(
                    lock["resource_accounting"]["final_prototypes_bytes"]
                ),
                "additional_persistent_bytes": int(
                    lock["resource_accounting"]["additional_persistent_bytes"]
                ),
                "raw_train_feature_bank_required": False,
                "classifier_prediction_changed": False,
                "id_top1_accuracy_delta": 0.0,
            }
        ]
    )
    output = Path(lock["outputs"]["result_dir"])
    output.mkdir(parents=True, exist_ok=True)
    per_dataset.to_csv(output / "final_per_dataset.csv", index=False)
    summary.to_csv(output / "final_summary.csv", index=False)
    comparison.to_csv(output / "final_comparison.csv", index=False)
    worst.to_csv(output / "worst_dataset.csv", index=False)
    pd.DataFrame(error_rows).to_csv(output / "error_analysis.csv", index=False)
    pd.DataFrame(case_rows).to_csv(output / "failure_cases.csv", index=False)
    pd.DataFrame(runtime_rows).to_csv(output / "inference_runtime.csv", index=False)
    resource.to_csv(output / "resource_accounting.csv", index=False)
    (output / "evaluation_provenance.json").write_text(
        json.dumps(
            {
                "config_sha256": config_sha,
                "final_suite_accessed_once_after_lock": True,
                "same_locked_method_all_datasets": True,
                "target_ood_used_for_fit_or_calibration": False,
                "test_image_sharing": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(per_dataset.to_string(index=False))
    print(summary.to_string(index=False))
    print(comparison.to_string(index=False))
    print(worst.to_string(index=False))


if __name__ == "__main__":
    main()
