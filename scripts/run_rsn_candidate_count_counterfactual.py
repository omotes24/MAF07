#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from maf07.config import load_yaml, resolve_path
from maf07.features import feature_frame_for_split
from maf07.jobs import append_csv_rows
from maf07.metrics import ood_metrics
from maf07.runner import _load_split


def _balanced_aupr(y_is_id: np.ndarray, scores: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score

    y_ood = 1 - np.asarray(y_is_id, dtype=int)
    n_ood = int(y_ood.sum())
    n_id = int(len(y_ood) - n_ood)
    weights = np.where(y_ood == 1, 0.5 / n_ood, 0.5 / n_id)
    return float(average_precision_score(y_ood, -scores, sample_weight=weights))


def _class_distances(
    train_x: np.ndarray,
    train_labels: np.ndarray,
    query_x: np.ndarray,
    *,
    class_count: int,
    device: str,
    batch_size: int,
    rsn_k: int,
    knn_k: int,
    delta: float,
    min_std: float,
) -> tuple[np.ndarray, np.ndarray]:
    import torch

    torch_device = torch.device(device)
    train = torch.as_tensor(train_x, dtype=torch.float32, device=torch_device)
    query = torch.as_tensor(query_x, dtype=torch.float32, device=torch_device)
    query_l2 = query / query.norm(dim=1, keepdim=True).clamp_min(1e-12)
    rsn = np.empty((len(query_x), class_count), dtype=np.float32)
    knn = np.empty((len(query_x), class_count, knn_k), dtype=np.float32)

    with torch.no_grad():
        for class_id in range(class_count):
            mask = torch.as_tensor(
                np.flatnonzero(train_labels == class_id),
                dtype=torch.long,
                device=torch_device,
            )
            bank = train[mask]
            if len(bank) < max(rsn_k, knn_k):
                raise RuntimeError(
                    f"class={class_id} bank={len(bank)} is smaller than requested k"
                )
            scale = torch.std(bank, dim=0, unbiased=True).clamp_min(min_std)
            rsn_bank = (bank / scale).contiguous()
            rsn_bank_norm = torch.sum(rsn_bank * rsn_bank, dim=1)
            knn_bank = bank / bank.norm(dim=1, keepdim=True).clamp_min(1e-12)
            knn_bank_norm = torch.sum(knn_bank * knn_bank, dim=1)

            for start in range(0, len(query), batch_size):
                end = min(start + batch_size, len(query))
                scaled = (query[start:end] / scale).contiguous()
                d2 = (
                    torch.sum(scaled * scaled, dim=1, keepdim=True)
                    + rsn_bank_norm[None, :]
                    - 2.0 * scaled @ rsn_bank.T
                ).clamp_min(0.0)
                nearest = torch.topk(d2, k=rsn_k, dim=1, largest=False).indices
                diff = scaled[:, None, :] - rsn_bank[nearest]
                absolute = torch.abs(diff)
                penalty = torch.where(
                    absolute <= delta,
                    diff * diff,
                    2.0 * delta * absolute - delta * delta,
                )
                rsn[start:end, class_id] = (
                    penalty.sum(dim=2).mean(dim=1).cpu().numpy()
                )

                normalized = query_l2[start:end]
                knn_d2 = (
                    torch.sum(normalized * normalized, dim=1, keepdim=True)
                    + knn_bank_norm[None, :]
                    - 2.0 * normalized @ knn_bank.T
                ).clamp_min(0.0)
                knn[start:end, class_id] = (
                    torch.sqrt(
                        torch.topk(knn_d2, k=knn_k, dim=1, largest=False).values
                    )
                    .cpu()
                    .numpy()
                )
    return rsn, knn


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--split-dir", default="results/splits_content_cleaned_conservative"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--rsn-k", type=int, default=150)
    parser.add_argument("--knn-k", type=int, default=50)
    parser.add_argument("--delta", type=float, default=1.345)
    parser.add_argument("--min-std", type=float, default=1e-3)
    parser.add_argument("--max-configs", type=int, default=None)
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--no-final-summary", action="store_true")
    args = parser.parse_args()

    output = resolve_path(args.output)
    summary_path = resolve_path(args.summary_output)
    if args.summarize_only:
        full = pd.read_csv(output).drop_duplicates(
            ["configuration_id", "method"], keep="last"
        )
        summary = (
            full.groupby(["candidate_count", "method"])[
                ["AUROC", "FPR95", "AUPR_OUT", "AUPR_OUT_BALANCED"]
            ]
            .agg(["size", "mean", "std"])
        )
        summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary.reset_index().to_csv(summary_path, index=False)
        print(json.dumps({"result_rows": int(len(full)), "output": str(output)}))
        return 0

    classes = list(load_yaml("configs/dataset.yaml").get("classes", []))
    class_to_id = {name: index for index, name in enumerate(classes)}
    split = _load_split(args.seed, args.split_dir)
    rows, features = feature_frame_for_split(split, args.backbone)
    train_mask = rows["split"].eq("train").to_numpy()
    test_mask = rows["split"].eq("test").to_numpy()
    train_labels = rows.loc[train_mask, "class_name"].map(class_to_id).to_numpy(int)
    test_rows = rows.loc[test_mask].reset_index(drop=True)
    test_x = features[test_mask]
    rsn_distance, knn_top = _class_distances(
        features[train_mask],
        train_labels,
        test_x,
        class_count=len(classes),
        device=args.device,
        batch_size=args.batch_size,
        rsn_k=args.rsn_k,
        knn_k=args.knn_k,
        delta=args.delta,
        min_std=args.min_std,
    )

    if output.exists() and output.stat().st_size:
        done = set(
            pd.read_csv(output, usecols=["configuration_id"])["configuration_id"]
            .astype(str)
            .unique()
        )
    else:
        done = set()
    generated = 0
    for id_class, ood_class in itertools.permutations(classes, 2):
        distractors = [
            name for name in classes if name not in {id_class, ood_class}
        ]
        pair_mask = test_rows["class_name"].isin([id_class, ood_class]).to_numpy()
        pair_rows = test_rows.loc[pair_mask]
        y_is_id = pair_rows["class_name"].eq(id_class).to_numpy(int)
        for candidate_count in range(1, len(classes)):
            for extra in itertools.combinations(distractors, candidate_count - 1):
                candidates = (id_class, *extra)
                configuration_id = (
                    f"{args.backbone}:s{args.seed}:{id_class}>{ood_class}:"
                    f"{'|'.join(candidates)}"
                )
                if configuration_id in done:
                    continue
                candidate_ids = np.asarray(
                    [class_to_id[name] for name in candidates], dtype=int
                )
                rsn_score = -np.min(
                    rsn_distance[pair_mask][:, candidate_ids], axis=1
                )
                merged = knn_top[pair_mask][:, candidate_ids, :].reshape(
                    int(pair_mask.sum()), -1
                )
                knn_score = -np.partition(
                    merged, args.knn_k - 1, axis=1
                )[:, args.knn_k - 1]
                output_rows = []
                for method, score in (
                    ("rsn_class_std_huber_k150", rsn_score),
                    ("knn_l2_pooled_k50_kth", knn_score),
                ):
                    output_rows.append(
                        {
                            "configuration_id": configuration_id,
                            "backbone": args.backbone,
                            "seed": args.seed,
                            "anchor_id_class": id_class,
                            "target_ood_class": ood_class,
                            "candidate_count": candidate_count,
                            "candidate_classes": "|".join(candidates),
                            "method": method,
                            "id_n": int(y_is_id.sum()),
                            "ood_n": int((1 - y_is_id).sum()),
                            "ood_prevalence": float((1 - y_is_id).mean()),
                            "AUPR_OUT_BALANCED": _balanced_aupr(y_is_id, score),
                            **ood_metrics(y_is_id, score),
                        }
                    )
                append_csv_rows(pd.DataFrame(output_rows), output)
                generated += 1
                if args.max_configs is not None and generated >= args.max_configs:
                    break
            if args.max_configs is not None and generated >= args.max_configs:
                break
        if args.max_configs is not None and generated >= args.max_configs:
            break

    full = pd.read_csv(output).drop_duplicates(
        ["configuration_id", "method"], keep="last"
    )
    if not args.no_final_summary:
        summary = (
            full.groupby(["candidate_count", "method"])[
                ["AUROC", "FPR95", "AUPR_OUT", "AUPR_OUT_BALANCED"]
            ]
            .agg(["size", "mean", "std"])
        )
        summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary.reset_index().to_csv(summary_path, index=False)
    print(
        json.dumps(
            {
                "generated_configurations": generated,
                "result_rows": int(len(full)),
                "output": str(output),
                "causal_scope": (
                    "Anchor ID species, target OOD species, test samples, and prevalence "
                    "are held fixed while only distractor candidate banks are added."
                ),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
