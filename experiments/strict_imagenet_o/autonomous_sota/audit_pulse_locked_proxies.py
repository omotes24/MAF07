#!/usr/bin/env python3
"""Audit PULSE against MSPS on the preregistered ID-only proxy families."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "msps_sota"))

from calibration import calibrate_candidates  # noqa: E402
from prototype_stats import ReferenceStats  # noqa: E402
from scores import (  # noqa: E402
    candidate_confidence,
    gather_candidate_support,
    original_msps_confidence,
    select_candidates,
    stage_similarities,
)

from audit_symmetric_support_proxy import metric_row  # noqa: E402
from methods.compact_knn import mean_neighbor_confidence  # noqa: E402
from methods.id_proxies import class_partner_indices  # noqa: E402
from methods.pulse import fit_pulse_state  # noqa: E402
from reproduce_nnguide import assert_no_final_access, expand  # noqa: E402


STAGE_DIMS = (256, 512, 1024, 2048)
FINAL_DIM = 2048
PROXIES = ("radial_extrapolation", "channel_mask", "class_direction_extrapolation")
K = 200
NPROBE = 64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--activation-shards", nargs="+", required=True)
    parser.add_argument("--localized-shards", nargs="+", required=True)
    parser.add_argument("--multiview-shards", nargs="+", required=True)
    parser.add_argument("--rc-state", type=Path, required=True)
    parser.add_argument("--rc-config", type=Path, required=True)
    parser.add_argument("--compact-index", type=Path, required=True)
    parser.add_argument("--proxy-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--threads", type=int, default=24)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_aligned(
    activation_paths: list[Path],
    localized_paths: list[Path],
    multiview_paths: list[Path],
) -> dict[str, np.ndarray]:
    if not (len(activation_paths) == len(localized_paths) == len(multiview_paths)):
        raise ValueError("shard counts do not match")
    parts = {name: [] for name in ("labels", "shift", "canonical", "localized", "meanview")}
    calibration_parts = {name: [] for name in ("labels", "canonical", "localized", "meanview")}
    for activation_path, localized_path, multiview_path in zip(
        activation_paths, localized_paths, multiview_paths
    ):
        with np.load(activation_path, allow_pickle=False) as data:
            activation_labels = data["labels"].astype(np.int64)
            shift = data["shift_ratio"].astype(np.float32)
        with np.load(localized_path, allow_pickle=False) as data:
            localized_labels = data["labels"].astype(np.int64)
            localized = data["base_pooled"].astype(np.float32)
        with np.load(multiview_path, allow_pickle=False) as data:
            multiview_labels = data["labels"].astype(np.int64)
            canonical = data["base_pooled"].astype(np.float32)
            meanview = data["features"].astype(np.float32).mean(axis=1)
        if not np.array_equal(localized_labels, multiview_labels):
            raise ValueError(f"full validation alignment failed for {localized_path}")
        sampled = np.arange(0, len(multiview_labels), 10)
        if not np.array_equal(activation_labels, multiview_labels[sampled]):
            raise ValueError(f"activation alignment failed for {activation_path}")
        # The remaining 45 images per class estimate proxy directions independently.
        calibration = np.ones(len(multiview_labels), dtype=bool)
        calibration[sampled] = False
        parts["labels"].append(activation_labels)
        parts["shift"].append(shift)
        parts["canonical"].append(canonical[sampled])
        parts["localized"].append(localized[sampled])
        parts["meanview"].append(meanview[sampled])
        calibration_parts["labels"].append(multiview_labels[calibration])
        calibration_parts["canonical"].append(canonical[calibration])
        calibration_parts["localized"].append(localized[calibration])
        calibration_parts["meanview"].append(meanview[calibration])
    output = {name: np.concatenate(value) for name, value in parts.items()}
    output.update(
        {f"calibration_{name}": np.concatenate(value) for name, value in calibration_parts.items()}
    )
    return output


def split_fit_source(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    fit, source = [], []
    for class_id in range(1000):
        index = np.flatnonzero(labels == class_id)
        if len(index) < 4 or len(index) % 2:
            raise ValueError(f"class {class_id} cannot be split evenly: {len(index)} rows")
        middle = len(index) // 2
        fit.extend(index[:middle])
        source.extend(index[middle:])
    return np.asarray(fit), np.asarray(source)


def class_means(values: np.ndarray, labels: np.ndarray) -> np.ndarray:
    return np.stack([values[labels == class_id].mean(axis=0) for class_id in range(1000)])


def proxy_transform(
    values: np.ndarray,
    labels: np.ndarray,
    means: np.ndarray,
    proxy: str,
    *,
    stage_dims: tuple[int, ...] | None = None,
) -> np.ndarray:
    """Apply a locked proxy to vectors or view-by-vector tensors."""
    values = np.asarray(values, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    own = means[labels]
    if proxy == "radial_extrapolation":
        return own + 1.8 * (values - own)
    if proxy == "class_direction_extrapolation":
        partner = class_partner_indices(labels, shift=1)
        return values + 0.75 * (own - means[labels[partner]])
    if proxy != "channel_mask":
        raise ValueError(proxy)
    result = values.copy()
    if result.ndim == 3:
        result[..., ::10] = 0.0
        result /= 0.9
        return result
    if result.ndim != 2:
        raise ValueError("proxy features must have shape N x D or N x V x D")
    boundaries = (0, result.shape[1]) if stage_dims is None else tuple(np.cumsum((0, *stage_dims)))
    for begin, stop in zip(boundaries[:-1], boundaries[1:]):
        result[:, np.arange(begin, stop, 10)] = 0.0
        result[:, begin:stop] /= 0.9
    return result


@torch.inference_mode()
def geometry(
    canonical: np.ndarray,
    localized: np.ndarray,
    prototypes: np.ndarray,
    index,
    meanview: np.ndarray,
    *,
    device: str,
    batch_size: int,
) -> dict[str, np.ndarray]:
    prototype = F.normalize(torch.as_tensor(prototypes, dtype=torch.float32, device=device), dim=1)
    uniform, deep, local = [], [], []
    for begin in range(0, len(canonical), batch_size):
        whole = F.normalize(
            torch.as_tensor(canonical[begin : begin + batch_size, -FINAL_DIM:], device=device),
            dim=1,
        )
        crop = F.normalize(
            torch.as_tensor(localized[begin : begin + batch_size], device=device), dim=2
        )
        similarity = whole @ prototype.T
        rms = similarity.square().mean(dim=1).sqrt().clamp_min(1e-12)
        uniform.append((similarity.mean(dim=1) / rms).cpu().numpy())
        deep.append(similarity.amax(dim=1).cpu().numpy())
        local.append(torch.einsum("bvd,cd->bvc", crop, prototype).amax(dim=(1, 2)).cpu().numpy())
    return {
        "uniform": np.concatenate(uniform),
        "deep": np.concatenate(deep),
        "localized": np.concatenate(local),
        "knn": mean_neighbor_confidence(
            index, meanview, k=K, nprobe=NPROBE, batch_size=max(batch_size, 512)
        ),
    }


def confidence_scores(
    canonical: np.ndarray,
    geometry_values: dict[str, np.ndarray],
    shift: np.ndarray,
    pulse_state,
    stats: ReferenceStats,
    extra: dict[str, np.ndarray],
    rc_config: dict[str, object],
    *,
    device: str,
) -> dict[str, np.ndarray]:
    similarity = stage_similarities(canonical, stats.prototypes, device=device, batch_size=256)
    known = np.ones(1000, dtype=bool)
    msps = original_msps_confidence(similarity, known)
    candidates = select_candidates(similarity, known, int(rc_config["k"]))
    support = gather_candidate_support(similarity, candidates)
    calibrated = calibrate_candidates(
        support,
        candidates,
        extra["class_sorted"],
        extra["global_sorted"],
        float(rc_config["class_shrinkage"]),
    )
    rc = candidate_confidence(
        calibrated,
        candidates,
        similarity.argmax(axis=2),
        weights=np.asarray(rc_config["weights"], dtype=np.float32),
        fusion=str(rc_config["fusion"]),
        variance_penalty=float(rc_config["lambda"]),
        consensus_reward=float(rc_config["eta"]),
    )
    component = pulse_state.components(
        shift,
        geometry_values["uniform"],
        geometry_values["localized"],
        geometry_values["knn"],
    )
    return {
        "MSPS": msps,
        "RC-MSPS": rc,
        "PULSE": -sum(component.values()),
        "PULSE-no-shift": -(component["uniform"] + component["localized"] + component["knn"]),
        "DeepPrototype": geometry_values["deep"],
        "LocalizedPrototype": geometry_values["localized"],
        "MeanViewKNN": geometry_values["knn"],
        **{f"_ood_{name}": value for name, value in component.items()},
        **{
            f"_stage_support_{layer}": similarity[:, layer, :].max(axis=1)
            for layer in range(similarity.shape[1])
        },
    }


def anchor_formulas(
    scores: dict[str, np.ndarray],
    *,
    msps_center: float,
    msps_scale: float,
) -> dict[str, np.ndarray]:
    """Return fixed MSPS-anchored consensus corrections as ID confidence."""
    anchor = (-scores["MSPS"] - msps_center) / msps_scale
    evidence = {
        name: np.maximum(scores[f"_ood_{name}"], 0.0)
        for name in ("shift", "uniform", "localized", "knn")
    }
    output = {}
    for names in itertools.combinations(evidence, 3):
        values = np.column_stack([evidence[name] for name in names])
        product = np.prod(values, axis=1)
        positive_count = (values > 0.0).sum(axis=1)
        aggregations = {
            "min": values.min(axis=1),
            "geomean": np.cbrt(product),
            "median": np.median(values, axis=1),
            "harmonic": np.where(
                positive_count == 3,
                3.0 / np.maximum((1.0 / np.maximum(values, 1e-12)).sum(axis=1), 1e-12),
                0.0,
            ),
        }
        suffix = "".join(name[0] for name in names).upper()
        for aggregation, correction in aggregations.items():
            # Confidence is the negative of the higher-is-OOD anchored score.
            output[f"Anchor{aggregation.title()}-{suffix}"] = -(anchor + correction)
    return output


def empirical_quantile(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    ordered = np.sort(np.asarray(reference, dtype=np.float64))
    return (1.0 + np.searchsorted(ordered, values, side="right")) / (1.0 + len(ordered))


def cauchy_stage_confidence(
    scores: dict[str, np.ndarray],
    stage_references: tuple[np.ndarray, ...],
) -> np.ndarray:
    q = np.column_stack(
        [
            empirical_quantile(reference, scores[f"_stage_support_{layer}"])
            for layer, reference in enumerate(stage_references)
        ]
    )
    q = np.clip(q, 1e-4, 1.0 - 1e-4)
    statistic = np.tan((0.5 - q) * np.pi).mean(axis=1)
    return 0.5 - np.arctan(statistic) / np.pi


def cauchy_anchor_formulas(
    scores: dict[str, np.ndarray],
    *,
    anchor_center: float,
    anchor_scale: float,
) -> dict[str, np.ndarray]:
    """Fuse one multi-stage anchor with three coefficient-free OOD cues."""
    anchor = (-scores["CauchyStageTail"] - anchor_center) / anchor_scale
    evidence = {
        name: np.maximum(scores[f"_ood_{name}"], 0.0)
        for name in ("shift", "uniform", "localized", "knn")
    }
    output = {}
    for names in itertools.combinations(evidence, 3):
        values = np.column_stack([evidence[name] for name in names])
        product = np.prod(values, axis=1)
        aggregations = {
            "min": values.min(axis=1),
            "geomean": np.cbrt(product),
            "median": np.median(values, axis=1),
            "sum": values.sum(axis=1),
        }
        suffix = "".join(name[0] for name in names).upper()
        for aggregation, correction in aggregations.items():
            output[f"Cauchy{aggregation.title()}-{suffix}"] = -(anchor + correction)
    return output


def cauchy_envelope_formulas(
    scores: dict[str, np.ndarray],
    *,
    anchor_center: float,
    anchor_scale: float,
) -> dict[str, np.ndarray]:
    """Use a scale-balanced OR envelope between stage-tail and three cues."""
    anchor = (-scores["CauchyStageTail"] - anchor_center) / anchor_scale
    evidence = {
        name: scores[f"_ood_{name}"] for name in ("shift", "uniform", "localized", "knn")
    }
    output = {}
    for names in itertools.combinations(evidence, 3):
        values = np.column_stack([evidence[name] for name in names])
        positive = np.maximum(values, 0.0)
        branches = {
            "sum": values.sum(axis=1) / np.sqrt(3.0),
            "median": np.median(values, axis=1),
            "geomean": np.cbrt(np.prod(positive, axis=1)),
            "max": values.max(axis=1),
        }
        suffix = "".join(name[0] for name in names).upper()
        for branch_name, branch in branches.items():
            output[f"Envelope{branch_name.title()}-{suffix}"] = -np.maximum(anchor, branch)
    return output


def quantile_envelope_references(
    scores: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    references = {"cauchy": -scores["CauchyStageTail"]}
    names = ("shift", "uniform", "localized", "knn")
    for selected in itertools.combinations(names, 3):
        suffix = "".join(name[0] for name in selected).upper()
        references[suffix] = sum(scores[f"_ood_{name}"] for name in selected)
    return references


def quantile_envelope_formulas(
    scores: dict[str, np.ndarray],
    references: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """OR-combine two branches after independent ID-only ECDF alignment."""
    cauchy_q = empirical_quantile(references["cauchy"], -scores["CauchyStageTail"])
    output = {}
    names = ("shift", "uniform", "localized", "knn")
    for selected in itertools.combinations(names, 3):
        suffix = "".join(name[0] for name in selected).upper()
        branch = sum(scores[f"_ood_{name}"] for name in selected)
        branch_q = empirical_quantile(references[suffix], branch)
        output[f"QuantileEnvelope-{suffix}"] = -np.maximum(cauchy_q, branch_q)
    return output


def main() -> None:
    args = parse_args()
    activation_paths = expand(args.activation_shards)
    localized_paths = expand(args.localized_shards)
    multiview_paths = expand(args.multiview_shards)
    input_paths = [
        *activation_paths,
        *localized_paths,
        *multiview_paths,
        args.rc_state,
        args.rc_config,
        args.compact_index,
        args.proxy_lock,
        args.output,
    ]
    assert_no_final_access(input_paths)
    locked = json.loads(args.proxy_lock.read_text(encoding="utf-8"))
    if tuple(locked["selected_proxies"]) != PROXIES:
        raise ValueError("proxy lock does not match the audited proxy families")

    import faiss

    faiss.omp_set_num_threads(args.threads)
    index = faiss.read_index(str(args.compact_index))
    data = load_aligned(activation_paths, localized_paths, multiview_paths)
    fit_index, source_index = split_fit_source(data["labels"])
    calibration_labels = data["calibration_labels"]
    means = {
        name: class_means(data[f"calibration_{name}"], calibration_labels)
        for name in ("canonical", "localized", "meanview")
    }
    stats, extra = ReferenceStats.load(args.rc_state)
    rc_config = json.loads(args.rc_config.read_text(encoding="utf-8"))["hyperparameters"]
    fit_geometry = geometry(
        data["canonical"][fit_index],
        data["localized"][fit_index],
        stats.prototypes[-1],
        index,
        data["meanview"][fit_index],
        device=args.device,
        batch_size=args.batch_size,
    )
    from methods.activation_shift import robust_location_scale

    shift_center, shift_scale = robust_location_scale(np.log1p(data["shift"][fit_index]))
    pulse_state = fit_pulse_state(
        shift_center=shift_center,
        shift_scale=shift_scale,
        uniform_cosine=fit_geometry["uniform"],
        localized_prototype_confidence=fit_geometry["localized"],
        compact_knn_confidence=fit_geometry["knn"],
    )
    fit_scores = confidence_scores(
        data["canonical"][fit_index], fit_geometry, data["shift"][fit_index], pulse_state,
        stats, extra, rc_config, device=args.device,
    )
    msps_center, msps_scale = robust_location_scale(-fit_scores["MSPS"])
    stage_references = tuple(
        fit_scores[f"_stage_support_{layer}"] for layer in range(len(STAGE_DIMS))
    )
    fit_scores["CauchyStageTail"] = cauchy_stage_confidence(fit_scores, stage_references)
    cauchy_center, cauchy_scale = robust_location_scale(-fit_scores["CauchyStageTail"])
    quantile_references = quantile_envelope_references(fit_scores)
    args.output.mkdir(parents=True, exist_ok=True)
    pulse_state.save(args.output / "pulse_state.npz")
    np.savez(
        args.output / "cauchy_state.npz",
        **{
            f"stage_reference_{layer}": reference
            for layer, reference in enumerate(stage_references)
        },
        cauchy_center=np.asarray(cauchy_center, dtype=np.float64),
        cauchy_scale=np.asarray(cauchy_scale, dtype=np.float64),
        **{
            f"quantile_reference_{name}": reference
            for name, reference in quantile_references.items()
        },
    )
    source = {name: data[name][source_index] for name in ("canonical", "localized", "meanview")}
    source_labels = data["labels"][source_index]
    source_shift = data["shift"][source_index]
    source_geometry = geometry(
        source["canonical"], source["localized"], stats.prototypes[-1], index,
        source["meanview"], device=args.device, batch_size=args.batch_size,
    )
    id_scores = confidence_scores(
        source["canonical"], source_geometry, source_shift, pulse_state, stats, extra,
        rc_config, device=args.device,
    )
    id_scores.update(
        anchor_formulas(id_scores, msps_center=msps_center, msps_scale=msps_scale)
    )
    id_scores["CauchyStageTail"] = cauchy_stage_confidence(id_scores, stage_references)
    id_scores.update(
        cauchy_anchor_formulas(
            id_scores, anchor_center=cauchy_center, anchor_scale=cauchy_scale
        )
    )
    id_scores.update(
        cauchy_envelope_formulas(
            id_scores, anchor_center=cauchy_center, anchor_scale=cauchy_scale
        )
    )
    id_scores.update(quantile_envelope_formulas(id_scores, quantile_references))

    rows = []
    score_dir = args.output / "scores"
    score_dir.mkdir(parents=True, exist_ok=True)
    for proxy in PROXIES:
        transformed = {
            "canonical": proxy_transform(
                source["canonical"], source_labels, means["canonical"], proxy,
                stage_dims=STAGE_DIMS,
            ),
            "localized": proxy_transform(
                source["localized"], source_labels, means["localized"], proxy
            ),
            "meanview": proxy_transform(
                source["meanview"], source_labels, means["meanview"], proxy
            ),
        }
        proxy_geometry = geometry(
            transformed["canonical"], transformed["localized"], stats.prototypes[-1], index,
            transformed["meanview"], device=args.device, batch_size=args.batch_size,
        )
        # Activation shift is an image-map statistic; feature-space proxies leave it unchanged.
        proxy_scores = confidence_scores(
            transformed["canonical"], proxy_geometry, source_shift, pulse_state, stats, extra,
            rc_config, device=args.device,
        )
        proxy_scores.update(
            anchor_formulas(proxy_scores, msps_center=msps_center, msps_scale=msps_scale)
        )
        proxy_scores["CauchyStageTail"] = cauchy_stage_confidence(
            proxy_scores, stage_references
        )
        proxy_scores.update(
            cauchy_anchor_formulas(
                proxy_scores, anchor_center=cauchy_center, anchor_scale=cauchy_scale
            )
        )
        proxy_scores.update(
            cauchy_envelope_formulas(
                proxy_scores, anchor_center=cauchy_center, anchor_scale=cauchy_scale
            )
        )
        proxy_scores.update(
            quantile_envelope_formulas(proxy_scores, quantile_references)
        )
        payload = {}
        for method in id_scores:
            if method.startswith("_"):
                payload[f"id{method}"] = id_scores[method].astype(np.float32)
                payload[f"proxy{method}"] = proxy_scores[method].astype(np.float32)
                continue
            rows.append(
                {
                    "proxy": proxy,
                    "method": method,
                    **metric_row(id_scores[method], proxy_scores[method]),
                }
            )
            payload[f"id_{method}"] = id_scores[method].astype(np.float32)
            payload[f"proxy_{method}"] = proxy_scores[method].astype(np.float32)
        np.savez_compressed(score_dir / f"{proxy}.npz", **payload)

    table = pd.DataFrame(rows)
    table["rank"] = table.groupby("proxy")["AUROC"].rank(method="average", ascending=False)
    summary = (
        table.groupby("method", as_index=False)
        .agg(
            mean_rank=("rank", "mean"),
            mean_AUROC=("AUROC", "mean"),
            worst_AUROC=("AUROC", "min"),
            mean_FPR95=("FPR95", "mean"),
        )
        .sort_values(["mean_rank", "worst_AUROC"], ascending=[True, False])
    )
    msps = summary.set_index("method").loc["MSPS"]
    summary["passes_proxy_gate"] = (
        (summary["mean_rank"] <= msps["mean_rank"])
        & (summary["worst_AUROC"] >= msps["worst_AUROC"] - 0.02)
    )
    args.output.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output / "proxy_metrics.csv", index=False)
    summary.to_csv(args.output / "summary.csv", index=False)
    provenance = {
        "selected_proxies": list(PROXIES),
        "proxy_lock_sha256": sha256(args.proxy_lock),
        "compact_index_sha256": sha256(args.compact_index),
        "pulse_state_sha256": sha256(args.output / "pulse_state.npz"),
        "cauchy_state_sha256": sha256(args.output / "cauchy_state.npz"),
        "id_fit_rows": int(len(fit_index)),
        "id_source_rows": int(len(source_index)),
        "proxy_direction_rows_per_class_min": int(
            np.bincount(calibration_labels, minlength=1000).min()
        ),
        "proxy_direction_rows_per_class_max": int(
            np.bincount(calibration_labels, minlength=1000).max()
        ),
        "activation_shift_proxy_policy": "preserve own-image value under feature-space proxy",
        "target_ood_accessed": False,
        "final_benchmark_accessed": False,
        "selection_rule": "mean rank no worse than MSPS and worst AUROC within 0.02",
    }
    (args.output / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(table.to_string(index=False))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
