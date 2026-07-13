#!/usr/bin/env python3
"""Audit the frozen symmetric-support candidate on ID-only pseudo-OOD families."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from build_atomic_cache import FINAL_DIM, STAGE_DIMS, split_rows
from methods.id_proxies import generate_id_only_proxies
from methods.symmetric_support import activation_dispersion, score_from_raw
from reproduce_nnguide import assert_no_final_access, expand
from validate_proxies import load_validation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-shards", nargs="+", required=True)
    parser.add_argument("--atomic-dir", type=Path, required=True)
    parser.add_argument("--knn-dir", type=Path, required=True)
    parser.add_argument("--knn-validation", type=Path)
    parser.add_argument("--knn-calibration-indices", type=Path)
    parser.add_argument("--proxy-lock", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def metric_row(id_confidence: np.ndarray, proxy_confidence: np.ndarray) -> dict[str, float]:
    labels = np.concatenate([np.zeros(len(id_confidence)), np.ones(len(proxy_confidence))])
    ood = -np.concatenate([id_confidence, proxy_confidence])
    fpr, tpr, _ = roc_curve(1 - labels, -ood)
    index = int(np.flatnonzero(tpr >= 0.95)[0])
    return {
        "AUROC": float(roc_auc_score(labels, ood)),
        "FPR95": float(fpr[index]),
        "AUPR_OUT": float(average_precision_score(labels, ood)),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assemble_raw(
    atomic: dict[str, np.ndarray],
    knn: dict[str, np.ndarray],
    radial: np.ndarray,
) -> dict[str, np.ndarray]:
    return {
        "rc_msps": atomic["rc_msps"],
        "deep_support": atomic["deep_support"],
        "msps": atomic["msps"],
        "proto_relative": atomic["proto_relative"],
        "knn": knn["confidence_k200"],
        "radial": radial,
    }


def main() -> None:
    args = parse_args()
    validation_paths = expand(args.validation_shards)
    audit_paths = [
        args.atomic_dir,
        args.knn_dir,
        args.proxy_lock,
        args.preregistration,
        args.output,
    ]
    if args.knn_validation is not None:
        audit_paths.append(args.knn_validation)
    if args.knn_calibration_indices is not None:
        audit_paths.append(args.knn_calibration_indices)
    assert_no_final_access(validation_paths + audit_paths)
    prereg = json.loads(args.preregistration.read_text(encoding="utf-8"))
    if prereg["status"] != "frozen_before_id_only_proxy_audit":
        raise ValueError("candidate preregistration is not frozen")
    if prereg["payload_sha256"] != preregistration_payload_hash(prereg):
        raise ValueError("candidate preregistration hash mismatch")

    features, labels, paths = load_validation(validation_paths)
    fit, fit_labels, source, source_labels, calibration_features, _ = split_rows(
        features, labels, paths
    )
    class_means = np.stack([fit[fit_labels == class_id].mean(0) for class_id in range(1000)])
    generated = generate_id_only_proxies(
        source,
        source_labels,
        fit,
        class_means,
        stage_dims=STAGE_DIMS,
    )
    proxy_lock = json.loads(args.proxy_lock.read_text(encoding="utf-8"))
    selected = proxy_lock["selected_proxies"]

    atomic_calibration = load_npz(args.atomic_dir / "calibration.npz")
    knn_validation_path = args.knn_validation or (
        args.knn_dir.parent / "knn_k_diagnostic" / "validation.npz"
    )
    knn_indices_path = args.knn_calibration_indices or (
        args.knn_dir / "calibration_indices.npy"
    )
    validation_knn = load_npz(knn_validation_path)
    knn_indices = np.load(knn_indices_path, allow_pickle=False)
    calibration = assemble_raw(
        atomic_calibration,
        {"confidence_k200": validation_knn["confidence_k200"][knn_indices]},
        activation_dispersion(calibration_features[:, -FINAL_DIM:]),
    )

    id_atomic = load_npz(args.atomic_dir / "id_proxy_reference.npz")
    id_knn = load_npz(args.knn_dir / "id_proxy_reference.npz")
    id_raw = assemble_raw(
        id_atomic,
        id_knn,
        activation_dispersion(source[:, -FINAL_DIM:]),
    )
    id_candidate = score_from_raw(calibration, id_raw)

    rows: list[dict[str, object]] = []
    for proxy_name in selected:
        proxy_atomic = load_npz(args.atomic_dir / f"proxy_{proxy_name}.npz")
        proxy_knn = load_npz(args.knn_dir / f"proxy_{proxy_name}.npz")
        proxy_raw = assemble_raw(
            proxy_atomic,
            proxy_knn,
            activation_dispersion(generated[proxy_name][:, -FINAL_DIM:]),
        )
        methods = {
            "SymmetricSupport": (
                id_candidate,
                score_from_raw(calibration, proxy_raw),
            ),
            "RC-MSPS": (id_atomic["rc_msps"], proxy_atomic["rc_msps"]),
            "MSPS": (id_atomic["msps"], proxy_atomic["msps"]),
            "DeepSupport": (id_atomic["deep_support"], proxy_atomic["deep_support"]),
            "KNN-k200": (id_knn["confidence_k200"], proxy_knn["confidence_k200"]),
        }
        for method, (id_score, proxy_score) in methods.items():
            rows.append({"proxy": proxy_name, "method": method, **metric_row(id_score, proxy_score)})

    table = pd.DataFrame(rows)
    table["AUROC_rank"] = table.groupby("proxy")["AUROC"].rank(
        method="average", ascending=False
    )
    summary = (
        table.groupby("method", as_index=False)
        .agg(
            mean_AUROC=("AUROC", "mean"),
            worst_AUROC=("AUROC", "min"),
            mean_FPR95=("FPR95", "mean"),
            worst_FPR95=("FPR95", "max"),
            mean_AUROC_rank=("AUROC_rank", "mean"),
        )
        .sort_values(["mean_AUROC_rank", "worst_AUROC"], ascending=[True, False])
    )
    candidate = summary.set_index("method").loc["SymmetricSupport"]
    msps = summary.set_index("method").loc["MSPS"]
    passes = bool(
        candidate["mean_AUROC_rank"] <= msps["mean_AUROC_rank"]
        and candidate["worst_AUROC"] >= msps["worst_AUROC"] - 0.02
    )
    args.output.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output / "per_proxy.csv", index=False)
    summary.to_csv(args.output / "summary.csv", index=False)
    result = {
        "candidate": "SymmetricSupport",
        "passes_id_only_proxy_gate": passes,
        "gate": {
            "candidate_mean_AUROC_rank": float(candidate["mean_AUROC_rank"]),
            "msps_mean_AUROC_rank": float(msps["mean_AUROC_rank"]),
            "candidate_worst_AUROC": float(candidate["worst_AUROC"]),
            "minimum_allowed_worst_AUROC": float(msps["worst_AUROC"] - 0.02),
        },
        "selected_proxies": selected,
        "preregistration_sha256": sha256(args.preregistration),
        "legacy_ood_read_by_audit": False,
        "final_benchmark_accessed": False,
    }
    (args.output / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(json.dumps(result, indent=2, sort_keys=True))
    if not passes:
        raise SystemExit(2)


def preregistration_payload_hash(preregistration: dict[str, object]) -> str:
    payload = dict(preregistration)
    payload.pop("payload_sha256", None)
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(serialized).hexdigest()


if __name__ == "__main__":
    main()
