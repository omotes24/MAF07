#!/usr/bin/env python3
"""Reproduce official-form NNGuide on the frozen strict ImageNet-O protocol."""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from methods.nnguide import NNGuideState


FINAL_DIM = 2048
FINAL_BENCHMARK_TOKENS = ("ninco", "ssb_hard", "ssb-hard", "autonomous_sota_final")
OFFICIAL_NNGUIDE_COMMIT = "c123cac961b17a6c4f11adefd9ad861298be1469"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-shards", nargs="+", required=True)
    parser.add_argument("--id-shards", nargs="+", required=True)
    parser.add_argument("--ood", action="append", default=[], help="NAME=glob")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--per-class", type=int, default=10)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args()


def expand(patterns: list[str]) -> list[Path]:
    paths = sorted({Path(item).resolve() for pattern in patterns for item in glob.glob(pattern)})
    if not paths:
        raise FileNotFoundError(f"no cache shards matched {patterns}")
    return paths


def assert_no_final_access(paths: list[Path]) -> None:
    rejected = [str(path) for path in paths if any(token in str(path).lower() for token in FINAL_BENCHMARK_TOKENS)]
    if rejected:
        raise PermissionError(f"untouched final benchmark access is forbidden before lock: {rejected}")


def load_paths(cache_path: Path) -> np.ndarray:
    jsonl_path = cache_path.with_suffix(".jsonl")
    rows = [json.loads(line)["path"] for line in jsonl_path.read_text(encoding="utf-8").splitlines()]
    return np.asarray(rows, dtype=str)


def deterministic_balanced_bank(shards: list[Path], per_class: int) -> tuple[np.ndarray, np.ndarray]:
    """Select path-sorted ID-train examples, independently of shard ordering."""
    candidates: list[list[tuple[str, np.ndarray]]] = [[] for _ in range(1000)]
    for shard_path in sorted(shards):
        paths = load_paths(shard_path)
        with np.load(shard_path, allow_pickle=False) as shard:
            labels = shard["labels"].astype(np.int64)
            features = shard["base_pooled"][:, -FINAL_DIM:]
            if len(paths) != len(labels):
                raise ValueError(f"path/cache length mismatch in {shard_path}")
            for class_id in np.unique(labels):
                indices = np.flatnonzero(labels == class_id)
                for index in indices:
                    candidates[int(class_id)].append((paths[index], features[index].copy()))
    bank: list[np.ndarray] = []
    labels_out: list[int] = []
    for class_id, rows in enumerate(candidates):
        rows.sort(key=lambda item: item[0])
        if len(rows) < per_class:
            raise ValueError(f"class {class_id} has only {len(rows)} train examples")
        bank.extend(feature for _, feature in rows[:per_class])
        labels_out.extend([class_id] * per_class)
    return np.stack(bank).astype(np.float32), np.asarray(labels_out, dtype=np.int64)


def load_final_features(paths: list[Path]) -> np.ndarray:
    parts: list[np.ndarray] = []
    for path in paths:
        with np.load(path, allow_pickle=False) as shard:
            parts.append(shard["base_pooled"][:, -FINAL_DIM:])
    return np.concatenate(parts)


def classifier() -> tuple[np.ndarray, np.ndarray]:
    import timm

    model = timm.create_model("resnet50d", pretrained=True).eval()
    return (
        model.fc.weight.detach().cpu().numpy().astype(np.float32),
        model.fc.bias.detach().cpu().numpy().astype(np.float32),
    )


def metrics(id_confidence: np.ndarray, ood_confidence: np.ndarray) -> dict[str, float]:
    labels = np.concatenate([np.zeros(len(id_confidence)), np.ones(len(ood_confidence))])
    ood_score = -np.concatenate([id_confidence, ood_confidence])
    fpr, tpr, _ = roc_curve(1 - labels, -ood_score)
    threshold_index = int(np.flatnonzero(tpr >= 0.95)[0])
    return {
        "AUROC": float(roc_auc_score(labels, ood_score)),
        "FPR95": float(fpr[threshold_index]),
        "AUPR_OUT": float(average_precision_score(labels, ood_score)),
    }


def sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).view(np.uint8)).hexdigest()


def main() -> None:
    args = parse_args()
    started = time.time()
    train_paths = expand(args.train_shards)
    id_paths = expand(args.id_shards)
    ood_specs = dict(spec.split("=", 1) for spec in args.ood)
    ood_paths = {name: expand([pattern]) for name, pattern in ood_specs.items()}
    all_paths = train_paths + id_paths + [path for paths in ood_paths.values() for path in paths]
    assert_no_final_access(all_paths)
    args.output.mkdir(parents=True, exist_ok=True)

    # Fit must complete before any OOD feature cache is opened.
    bank_features, bank_labels = deterministic_balanced_bank(train_paths, args.per_class)
    weight, bias = classifier()
    state = NNGuideState.fit(bank_features, weight, bias, k=args.k, device=args.device)
    provenance = {
        "method": "NNGuide",
        "official_commit": OFFICIAL_NNGUIDE_COMMIT,
        "formula": "energy(query) * mean_topk_ip(unit_query, energy(train) * unit_train)",
        "score_orientation": "higher_is_id_confidence",
        "backbone": "timm/resnet50d pretrained",
        "fit_data": "ImageNet-1K train only",
        "ood_used_for_fit_or_selection": False,
        "per_class": args.per_class,
        "bank_size": len(bank_features),
        "k": args.k,
        "bank_label_histogram": np.bincount(bank_labels, minlength=1000).tolist(),
        "bank_sha256": sha256_array(bank_features),
        "classifier_weight_sha256": sha256_array(weight),
        "fit_completed_before_ood_loading": True,
    }
    (args.output / "fit_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    id_features = load_final_features(id_paths)
    id_confidence = state.confidence(id_features, device=args.device, batch_size=args.batch_size)
    score_dir = args.output / "scores"
    score_dir.mkdir(exist_ok=True)
    rows = []
    for dataset, paths in ood_paths.items():
        ood_features = load_final_features(paths)
        ood_confidence = state.confidence(
            ood_features, device=args.device, batch_size=args.batch_size
        )
        np.savez(
            score_dir / f"{dataset}.npz",
            id_confidence=id_confidence,
            ood_confidence=ood_confidence,
        )
        rows.append(
            {
                "method": "NNGuide",
                "dataset": dataset,
                "n_id": len(id_confidence),
                "n_ood": len(ood_confidence),
                **metrics(id_confidence, ood_confidence),
            }
        )
    per_dataset = pd.DataFrame(rows)
    per_dataset.to_csv(args.output / "per_dataset.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    summary = {
        "method": "NNGuide",
        "Macro_AUROC": float(per_dataset["AUROC"].mean()),
        "Macro_FPR95": float(per_dataset["FPR95"].mean()),
        "Macro_AUPR_OUT": float(per_dataset["AUPR_OUT"].mean()),
        "datasets": int(len(per_dataset)),
        "runtime_seconds": time.time() - started,
    }
    pd.DataFrame([summary]).to_csv(args.output / "summary.csv", index=False)
    print(per_dataset.to_string(index=False))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
