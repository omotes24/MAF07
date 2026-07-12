#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from maf07.config import resolve_path
from maf07.features import feature_frame_for_split
from maf07.methods.rsn_factorial import (
    FactorSpec,
    FactorialNeighborSuite,
    nnguide_score,
)
from maf07.methods.verified_baselines import VerifiedBaselineSuite, fit_ridge_linear_probe
from maf07.runner import _load_split
from maf07.splits import make_ood_eval_frame


RSN_SPEC = FactorSpec(
    "rsn_class_std_huber_k150", scale="class_std", aggregation="huber", k=150
)
KNN_SPEC = FactorSpec(
    "knn_l2_pooled_k50_kth",
    normalize=True,
    scale="none",
    aggregation="euclidean",
    bank_mode="pooled",
    neighbor_reduce="kth",
    k=50,
)


def _open_square(path: str, size: int = 220) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.fit(
            image.convert("RGB"), (size, size), method=Image.Resampling.LANCZOS
        )


def _nearest_for_query(
    train_x: np.ndarray,
    train_y: np.ndarray,
    query: np.ndarray,
    *,
    k_score: int = 150,
    k_show: int = 5,
) -> tuple[np.ndarray, np.ndarray, int]:
    train_norm = train_x / (np.linalg.norm(train_x, axis=1, keepdims=True) + 1e-12)
    query_norm = query / (np.linalg.norm(query) + 1e-12)
    pooled_distance = np.linalg.norm(train_norm - query_norm[None, :], axis=1)
    knn_indices = np.argsort(pooled_distance)[:k_show]

    best_score = np.inf
    best_class = -1
    best_indices = None
    for class_id in sorted(np.unique(train_y)):
        global_indices = np.flatnonzero(train_y == int(class_id))
        class_x = train_x[global_indices]
        scale = np.std(class_x, axis=0, ddof=1).clip(1e-3)
        difference = (class_x - query[None, :]) / scale[None, :]
        squared = np.sum(difference * difference, axis=1)
        k = min(int(k_score), len(class_x))
        local = np.argpartition(squared, k - 1)[:k]
        abs_diff = np.abs(difference[local])
        penalty = np.where(
            abs_diff <= 1.345,
            difference[local] ** 2,
            2.0 * 1.345 * abs_diff - 1.345**2,
        ).sum(axis=1)
        score = float(penalty.mean())
        if score < best_score:
            best_score = score
            best_class = int(class_id)
            ordered = local[np.argsort(squared[local])]
            best_indices = global_indices[ordered[:k_show]]
    return knn_indices, np.asarray(best_indices, dtype=int), best_class


def _case_indices(rows: pd.DataFrame, scores: np.ndarray, per_group: int) -> list[tuple[str, int]]:
    is_id = rows["role"].eq("id_test").to_numpy()
    groups = [
        ("ood_false_accept", np.flatnonzero(~is_id), True),
        ("ood_correct_reject", np.flatnonzero(~is_id), False),
        ("id_false_reject", np.flatnonzero(is_id), False),
        ("id_correct_accept", np.flatnonzero(is_id), True),
    ]
    selected = []
    for name, indices, descending in groups:
        order = indices[np.argsort(scores[indices])]
        if descending:
            order = order[::-1]
        selected.extend((name, int(index)) for index in order[:per_group])
    return selected


def _qualitative_panels(
    output_dir: Path,
    score_rows: pd.DataFrame,
    train_rows: pd.DataFrame,
    train_x: np.ndarray,
    train_y: np.ndarray,
    eval_x: np.ndarray,
    scores: np.ndarray,
    encoder: LabelEncoder,
    per_group: int,
) -> pd.DataFrame:
    records = []
    for case_index, (case_type, query_index) in enumerate(
        _case_indices(score_rows, scores, per_group)
    ):
        query_row = score_rows.iloc[query_index]
        knn_indices, rsn_indices, best_class = _nearest_for_query(
            train_x, train_y, eval_x[query_index]
        )
        fig, axes = plt.subplots(2, 6, figsize=(14, 5.2))
        for row_index, (label, indices) in enumerate(
            (("KNN", knn_indices), ("RSN", rsn_indices))
        ):
            axes[row_index, 0].imshow(_open_square(str(query_row["path"])))
            axes[row_index, 0].set_title(
                f"Query\n{query_row['class_name']}\n{case_type}", fontsize=8
            )
            axes[row_index, 0].axis("off")
            for column, train_index in enumerate(indices, start=1):
                neighbor = train_rows.iloc[int(train_index)]
                axes[row_index, column].imshow(_open_square(str(neighbor["path"])))
                axes[row_index, column].set_title(
                    f"{label} #{column}\n{neighbor['class_name']}", fontsize=8
                )
                axes[row_index, column].axis("off")
                records.append(
                    {
                        "case_type": case_type,
                        "query_image_id": str(query_row["image_id"]),
                        "query_class": str(query_row["class_name"]),
                        "query_score": float(scores[query_index]),
                        "neighbor_method": label,
                        "neighbor_rank": int(column),
                        "neighbor_image_id": str(neighbor["image_id"]),
                        "neighbor_class": str(neighbor["class_name"]),
                        "rsn_selected_class": str(encoder.inverse_transform([best_class])[0]),
                    }
                )
        fig.suptitle("Nearest-neighbor evidence for an RSN decision", weight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(
            output_dir / f"qualitative_{case_index:02d}_{case_type}.png",
            dpi=180,
            bbox_inches="tight",
        )
        plt.close(fig)
    return pd.DataFrame(records)


def _dimension_analysis(
    train_x: np.ndarray,
    train_y: np.ndarray,
    eval_x: np.ndarray,
    is_id: np.ndarray,
    *,
    sample_per_group: int,
    device: str,
) -> pd.DataFrame:
    import torch

    rng = np.random.default_rng(20260712)
    selected = []
    for value in (0, 1):
        candidates = np.flatnonzero(is_id == value)
        selected.extend(
            rng.choice(
                candidates, size=min(sample_per_group, len(candidates)), replace=False
            ).tolist()
        )
    torch_device = torch.device(device if torch.cuda.is_available() else "cpu")
    train_tensor = torch.as_tensor(train_x, dtype=torch.float32, device=torch_device)
    query_tensor = torch.as_tensor(
        eval_x[np.asarray(selected, dtype=int)], dtype=torch.float32, device=torch_device
    )
    class_scales = {}
    best_total = torch.full((len(selected),), torch.inf, device=torch_device)
    best_squared = torch.zeros(
        (len(selected), train_x.shape[1]), device=torch_device
    )
    best_huber = torch.zeros_like(best_squared)
    with torch.no_grad():
        for cls in sorted(np.unique(train_y)):
            class_indices = torch.as_tensor(
                np.flatnonzero(train_y == int(cls)),
                dtype=torch.long,
                device=torch_device,
            )
            class_values = train_tensor[class_indices]
            scale = torch.std(class_values, dim=0, unbiased=True).clamp_min(1e-3)
            class_scales[int(cls)] = scale.detach().cpu().numpy()
            bank = class_values / scale
            bank_norm = torch.sum(bank * bank, dim=1)
            k = min(150, len(bank))
            for start in range(0, len(query_tensor), 16):
                end = min(start + 16, len(query_tensor))
                query = query_tensor[start:end] / scale
                d2 = (
                    torch.sum(query * query, dim=1, keepdim=True)
                    + bank_norm[None, :]
                    - 2.0 * (query @ bank.T)
                ).clamp_min(0.0)
                indices = torch.topk(d2, k=k, dim=1, largest=False).indices
                difference = query[:, None, :] - bank[indices]
                squared = (difference * difference).mean(dim=1)
                absolute = torch.abs(difference)
                huber = torch.where(
                    absolute <= 1.345,
                    difference * difference,
                    2.0 * 1.345 * absolute - 1.345**2,
                ).mean(dim=1)
                total = huber.sum(dim=1)
                replace = total < best_total[start:end]
                if torch.any(replace):
                    rows = torch.nonzero(replace, as_tuple=False).flatten() + start
                    best_total[rows] = total[replace]
                    best_squared[rows] = squared[replace]
                    best_huber[rows] = huber[replace]

    selected_is_id = is_id[np.asarray(selected, dtype=int)]
    id_mask = torch.as_tensor(selected_is_id == 1, device=torch_device)
    ood_mask = ~id_mask
    accumulators = {
        "id_squared": best_squared[id_mask].mean(dim=0).cpu().numpy(),
        "id_huber": best_huber[id_mask].mean(dim=0).cpu().numpy(),
        "ood_squared": best_squared[ood_mask].mean(dim=0).cpu().numpy(),
        "ood_huber": best_huber[ood_mask].mean(dim=0).cpu().numpy(),
    }

    mean_scale = np.mean(np.stack(list(class_scales.values())), axis=0)
    frame = pd.DataFrame(
        {
            "dimension": np.arange(train_x.shape[1]),
            "mean_class_std": mean_scale,
            **accumulators,
        }
    )
    frame["ood_huber_suppression"] = frame["ood_squared"] - frame["ood_huber"]
    frame["id_huber_suppression"] = frame["id_squared"] - frame["id_huber"]
    frame["ood_minus_id_huber"] = frame["ood_huber"] - frame["id_huber"]
    frame["low_variance_rank"] = frame["mean_class_std"].rank(method="first")
    frame["suppression_rank"] = frame["ood_huber_suppression"].rank(
        method="first", ascending=False
    )
    return frame.sort_values("suppression_rank")


def _closed_confusion(
    split: pd.DataFrame,
    backbone: str,
    output_dir: Path,
) -> None:
    rows, features = feature_frame_for_split(split, backbone)
    train = rows["split"].eq("train").to_numpy()
    test = rows["split"].eq("test").to_numpy()
    encoder = LabelEncoder().fit(rows.loc[train, "class_name"])
    train_y = encoder.transform(rows.loc[train, "class_name"])
    test_y = encoder.transform(rows.loc[test, "class_name"])
    probe = fit_ridge_linear_probe(features[train], train_y)
    prediction = probe.logits(features[test]).argmax(axis=1)
    matrix = confusion_matrix(test_y, prediction, normalize="true")
    pd.DataFrame(matrix, index=encoder.classes_, columns=encoder.classes_).to_csv(
        output_dir / "species_confusion_matrix.csv"
    )
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    image = ax.imshow(matrix, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(encoder.classes_)), encoder.classes_, rotation=45, ha="right")
    ax.set_yticks(range(len(encoder.classes_)), encoder.classes_)
    ax.set_xlabel("Predicted species")
    ax.set_ylabel("True species")
    ax.set_title("Closed-set species confusion (row-normalized)", weight="bold")
    fig.colorbar(image, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(output_dir / "species_confusion_matrix.png", dpi=220)
    plt.close(fig)


def _dimension_extrema_panel(
    rows: pd.DataFrame,
    features: np.ndarray,
    dimensions: pd.DataFrame,
    output_dir: Path,
    *,
    top_dimensions: int = 6,
    examples_per_side: int = 3,
) -> None:
    selected = dimensions.head(top_dimensions)["dimension"].astype(int).tolist()
    records = []
    fig, axes = plt.subplots(
        len(selected),
        2 * examples_per_side,
        figsize=(2.2 * 2 * examples_per_side, 2.25 * len(selected)),
        squeeze=False,
    )
    for row_index, dimension in enumerate(selected):
        order = np.argsort(features[:, dimension])
        indices = np.concatenate(
            [order[:examples_per_side], order[-examples_per_side:][::-1]]
        )
        for column, image_index in enumerate(indices):
            image_row = rows.iloc[int(image_index)]
            axes[row_index, column].imshow(_open_square(str(image_row["path"]), 180))
            side = "low" if column < examples_per_side else "high"
            axes[row_index, column].set_title(
                f"d={dimension} {side}\n{image_row['class_name']}\n"
                f"{features[int(image_index), dimension]:.3g}",
                fontsize=7,
            )
            axes[row_index, column].axis("off")
            records.append(
                {
                    "dimension": dimension,
                    "activation_side": side,
                    "rank_within_side": (
                        column + 1
                        if side == "low"
                        else column - examples_per_side + 1
                    ),
                    "image_id": str(image_row["image_id"]),
                    "class_name": str(image_row["class_name"]),
                    "activation": float(features[int(image_index), dimension]),
                }
            )
    fig.suptitle(
        "Image extrema for dimensions most affected by Huber aggregation",
        weight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(output_dir / "huber_dimension_extrema.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(records).to_csv(
        output_dir / "huber_dimension_extrema.csv", index=False
    )


def _score_distribution(
    rows: pd.DataFrame,
    scores: dict[str, np.ndarray],
    output_dir: Path,
) -> None:
    is_id = rows["role"].eq("id_test").to_numpy()
    records = []
    fig, axes = plt.subplots(1, len(scores), figsize=(6.2 * len(scores), 4.0))
    axes = np.atleast_1d(axes)
    for ax, (method, values) in zip(axes, scores.items(), strict=True):
        ax.hist(values[is_id], bins=60, density=True, alpha=0.5, label="ID")
        ax.hist(values[~is_id], bins=60, density=True, alpha=0.5, label="OOD")
        ax.set_title(method, weight="bold")
        ax.set_xlabel("ID score")
        ax.legend(frameon=False)
        for label, mask in (("ID", is_id), ("OOD", ~is_id)):
            records.append(
                {
                    "method": method,
                    "group": label,
                    "n": int(mask.sum()),
                    "mean": float(values[mask].mean()),
                    "std": float(values[mask].std()),
                    "q05": float(np.quantile(values[mask], 0.05)),
                    "q50": float(np.quantile(values[mask], 0.50)),
                    "q95": float(np.quantile(values[mask], 0.95)),
                }
            )
    fig.tight_layout()
    fig.savefig(output_dir / "score_distributions.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(records).to_csv(output_dir / "score_distribution_summary.csv", index=False)


def _benchmark(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    eval_x: np.ndarray,
    *,
    device: str,
    repeats: int,
) -> pd.DataFrame:
    import torch

    baseline = VerifiedBaselineSuite(train_x, train_y, val_x)
    eval_logits = baseline.probe.logits(eval_x)
    methods = [(RSN_SPEC.name, RSN_SPEC), (KNN_SPEC.name, KNN_SPEC)]
    rows = []
    for name, spec in methods:
        durations = []
        peaks = []
        for _ in range(repeats):
            if torch.cuda.is_available() and device.startswith("cuda"):
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            started = time.perf_counter()
            FactorialNeighborSuite(train_x, train_y, device=device).score_many(
                eval_x, [spec]
            )
            if torch.cuda.is_available() and device.startswith("cuda"):
                torch.cuda.synchronize()
                peak = int(torch.cuda.max_memory_allocated())
            else:
                peak = 0
            durations.append(time.perf_counter() - started)
            peaks.append(peak)
        rows.append(
            {
                "method": name,
                "query_n": int(len(eval_x)),
                "bank_n": int(len(train_x)),
                "bank_bytes": int(train_x.nbytes),
                "runtime_sec_median": float(np.median(durations)),
                "runtime_sec_min": float(np.min(durations)),
                "queries_per_sec": float(len(eval_x) / np.median(durations)),
                "peak_gpu_bytes": int(max(peaks)),
            }
        )
    durations = []
    peaks = []
    for _ in range(repeats):
        if torch.cuda.is_available() and device.startswith("cuda"):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        started = time.perf_counter()
        nnguide_score(
            train_x,
            baseline.train_logits,
            eval_x,
            eval_logits,
            k=10,
            device=device,
        )
        if torch.cuda.is_available() and device.startswith("cuda"):
            torch.cuda.synchronize()
            peak = int(torch.cuda.max_memory_allocated())
        else:
            peak = 0
        durations.append(time.perf_counter() - started)
        peaks.append(peak)
    rows.append(
        {
            "method": "nnguide_k10",
            "query_n": int(len(eval_x)),
            "bank_n": int(len(train_x)),
            "bank_bytes": int(train_x.nbytes),
            "runtime_sec_median": float(np.median(durations)),
            "runtime_sec_min": float(np.min(durations)),
            "queries_per_sec": float(len(eval_x) / np.median(durations)),
            "peak_gpu_bytes": int(max(peaks)),
        }
    )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--backbone", default="dinov2_vitb14")
    parser.add_argument("--id-set", default="cheetah|jaguar")
    parser.add_argument(
        "--split-dir", default="results/splits_content_cleaned_conservative"
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cases-per-group", type=int, default=2)
    parser.add_argument("--dimension-samples-per-group", type=int, default=256)
    parser.add_argument("--benchmark-repeats", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split = _load_split(args.seed, args.split_dir)
    eval_frame = make_ood_eval_frame(split, args.id_set.split("|"), protocol="fair")
    rows, features = feature_frame_for_split(eval_frame, args.backbone)
    train_mask = rows["role"].eq("id_train").to_numpy()
    val_mask = rows["role"].eq("id_val").to_numpy()
    eval_mask = rows["role"].isin(["id_test", "ood_test"]).to_numpy()
    train_rows = rows.loc[train_mask].reset_index(drop=True)
    score_rows = rows.loc[eval_mask].reset_index(drop=True)
    encoder = LabelEncoder().fit(train_rows["class_name"])
    train_y = encoder.transform(train_rows["class_name"])
    train_x = features[train_mask]
    val_x = features[val_mask]
    eval_x = features[eval_mask]
    is_id = score_rows["role"].eq("id_test").to_numpy().astype(int)

    suite = FactorialNeighborSuite(train_x, train_y, device=args.device)
    scores = suite.score_many(eval_x, [RSN_SPEC, KNN_SPEC])
    qualitative = _qualitative_panels(
        output_dir,
        score_rows,
        train_rows,
        train_x,
        train_y,
        eval_x,
        scores[RSN_SPEC.name],
        encoder,
        args.cases_per_group,
    )
    qualitative.to_csv(output_dir / "qualitative_cases.csv", index=False)
    dimensions = _dimension_analysis(
        train_x,
        train_y,
        eval_x,
        is_id,
        sample_per_group=args.dimension_samples_per_group,
        device=args.device,
    )
    dimensions.to_csv(output_dir / "dimension_huber_analysis.csv", index=False)
    _dimension_extrema_panel(score_rows, eval_x, dimensions, output_dir)
    _closed_confusion(split, args.backbone, output_dir)
    _score_distribution(score_rows, scores, output_dir)
    benchmark = _benchmark(
        train_x,
        train_y,
        val_x,
        eval_x,
        device=args.device,
        repeats=args.benchmark_repeats,
    )
    benchmark.to_csv(output_dir / "runtime_memory.csv", index=False)
    report = {
        "seed": int(args.seed),
        "backbone": args.backbone,
        "id_set": args.id_set,
        "train_n": int(len(train_x)),
        "eval_n": int(len(eval_x)),
        "qualitative_cases": int(qualitative["query_image_id"].nunique()),
        "dimension_samples_per_group": int(args.dimension_samples_per_group),
    }
    (output_dir / "representative_fold_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
