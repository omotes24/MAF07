#!/usr/bin/env python3
"""Materialize reusable atomic scores from ID-only calibration and locked proxies."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "msps_sota"))

from calibration import calibrate_candidates  # noqa: E402
from prototype_stats import ReferenceStats  # noqa: E402
from scores import (  # noqa: E402
    candidate_confidence,
    diagonal_residuals,
    gather_candidate_support,
    original_msps_confidence,
    select_candidates,
    stage_similarities,
)

from methods.atomic_components import (  # noqa: E402
    AtomicFit,
    class_empirical_cdf,
    fit_atomic_references,
    geometric_barycentric_confidence,
    logits_numpy,
)
from methods.id_proxies import generate_id_only_proxies  # noqa: E402
from reproduce_nnguide import assert_no_final_access, classifier, expand  # noqa: E402
from validate_proxies import load_validation  # noqa: E402


STAGE_DIMS = (256, 512, 1024, 2048)
FINAL_DIM = 2048


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-shards", nargs="+", required=True)
    parser.add_argument("--rc-state", type=Path, required=True)
    parser.add_argument("--rc-config", type=Path, required=True)
    parser.add_argument("--proxy-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def split_rows(
    features: np.ndarray, labels: np.ndarray, paths: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    fit_indices, source_indices, calibration_indices = [], [], []
    for class_id in range(1000):
        indices = np.flatnonzero(labels == class_id)
        ordered = indices[np.argsort(paths[indices])]
        if len(ordered) != 50:
            raise ValueError(f"class {class_id} has {len(ordered)} rows, expected 50")
        fit_indices.extend(ordered[:25])
        source_indices.extend(ordered[25:30])
        calibration_indices.extend(ordered[30:40])
    fit_indices = np.asarray(fit_indices)
    source_indices = np.asarray(source_indices)
    calibration_indices = np.asarray(calibration_indices)
    return (
        features[fit_indices].astype(np.float32),
        labels[fit_indices],
        features[source_indices].astype(np.float32),
        labels[source_indices],
        features[calibration_indices].astype(np.float32),
        labels[calibration_indices],
    )


class AtomicExtractor:
    def __init__(
        self,
        stats: ReferenceStats,
        extra: dict[str, np.ndarray],
        fit: AtomicFit,
        weight: np.ndarray,
        bias: np.ndarray,
        rc_config: dict[str, object],
        device: str,
    ) -> None:
        self.stats = stats
        self.extra = extra
        self.fit = fit
        self.weight = weight
        self.bias = bias
        self.config = rc_config
        self.device = device
        self.stage_weights = np.asarray(rc_config["weights"], dtype=np.float32)

    def extract(self, features: np.ndarray) -> dict[str, np.ndarray]:
        similarity = stage_similarities(
            features, self.stats.prototypes, device=self.device, batch_size=256
        )
        similarity32 = similarity.astype(np.float32)
        weighted = np.einsum("nlc,l->nc", similarity32, self.stage_weights)
        top5 = np.argpartition(weighted, -5, axis=1)[:, -5:]
        top5_values = np.take_along_axis(weighted, top5, axis=1)
        order = np.argsort(top5_values, axis=1)[:, ::-1]
        top5 = np.take_along_axis(top5, order, axis=1)
        top5_values = np.take_along_axis(top5_values, order, axis=1)
        stage_winners = similarity32.argmax(axis=2)
        deep_winner = stage_winners[:, -1]

        known = np.ones(len(self.stats.counts), dtype=bool)
        rc_candidates = select_candidates(similarity, known, int(self.config["k"]))
        rc_support = gather_candidate_support(similarity, rc_candidates)
        rc_q = calibrate_candidates(
            rc_support,
            rc_candidates,
            self.extra["class_sorted"],
            self.extra["global_sorted"],
            float(self.config["class_shrinkage"]),
        )
        rc_msps = candidate_confidence(
            rc_q,
            rc_candidates,
            stage_winners,
            weights=self.stage_weights,
            fusion=str(self.config["fusion"]),
            variance_penalty=float(self.config["lambda"]),
            consensus_reward=float(self.config["eta"]),
        )
        msps = original_msps_confidence(similarity, known)

        final = features[:, -FINAL_DIM:].astype(np.float32)
        logits = logits_numpy(final, self.weight, self.bias, device=self.device)
        logit_top5 = np.argpartition(logits, -5, axis=1)[:, -5:]
        logit_top5_values = np.take_along_axis(logits, logit_top5, axis=1)
        logit_order = np.argsort(logit_top5_values, axis=1)[:, ::-1]
        logit_top5 = np.take_along_axis(logit_top5, logit_order, axis=1)
        candidates = np.concatenate([top5, logit_top5], axis=1)
        candidate_proto = np.take_along_axis(weighted, candidates, axis=1)
        candidate_logits = np.take_along_axis(logits, candidates, axis=1)
        proto_q = class_empirical_cdf(candidate_proto, candidates, self.fit.own_proto_sorted)
        logit_q = class_empirical_cdf(candidate_logits, candidates, self.fit.own_logit_sorted)

        diagonal = diagonal_residuals(
            features,
            top5,
            self.stats,
            rho=0.1,
            device=self.device,
            batch_size=128,
        )
        class_diagonal = np.einsum("nlk,l->nk", diagonal, self.stage_weights).min(axis=1)
        boundaries = np.cumsum((0, *STAGE_DIMS))
        global_diagonal = np.zeros(len(features), dtype=np.float32)
        for layer, (begin, stop) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            stage = features[:, begin:stop].astype(np.float32)
            stage /= np.maximum(np.linalg.norm(stage, axis=1, keepdims=True), 1e-12)
            variance = np.maximum(self.stats.global_variances[layer], 1e-8)
            residual = ((stage - self.fit.global_centers[layer]) ** 2 / variance).mean(axis=1)
            global_diagonal += self.stage_weights[layer] * residual

        top1 = top5[:, 0]
        margin = top5_values[:, 0] - top5_values[:, 1]
        normalized_margin = class_empirical_cdf(
            margin[:, None], top1[:, None], self.fit.own_margin_sorted
        )[:, 0]
        final_norm = np.linalg.norm(final, axis=1)
        norm_shell = -np.abs(
            (final_norm - self.fit.norm_median[top1]) / self.fit.norm_mad[top1]
        )
        barycentric, unit_final, _ = geometric_barycentric_confidence(
            final,
            top5,
            top5_values,
            self.stats.prototypes[-1],
            device=self.device,
        )
        top_proto = self.stats.prototypes[-1][top1]
        global_var = np.maximum(self.stats.global_variances[-1], 1e-8)
        fisher_proto = -((unit_final - top_proto) ** 2 / global_var).mean(axis=1)
        unit_weight = self.weight[top1]
        unit_weight /= np.maximum(np.linalg.norm(unit_weight, axis=1, keepdims=True), 1e-12)
        classifier_alignment = (unit_final * unit_weight).sum(axis=1)

        deep_support = similarity32[np.arange(len(features)), -1, deep_winner]
        reciprocal_ranks = np.zeros(len(features), dtype=np.float32)
        for layer in range(len(STAGE_DIMS)):
            target = similarity32[np.arange(len(features)), layer, deep_winner]
            rank = (similarity32[:, layer, :] > target[:, None]).sum(axis=1)
            reciprocal_ranks += 1.0 / (1.0 + rank)
        reciprocal_ranks /= len(STAGE_DIMS)

        logits_tensor = torch.from_numpy(logits)
        probabilities = torch.softmax(logits_tensor, dim=1).numpy()
        energy = torch.logsumexp(logits_tensor, dim=1).numpy()
        logit_sorted = np.partition(logits, -2, axis=1)[:, -2:]
        logit_margin = logit_sorted.max(axis=1) - logit_sorted.min(axis=1)
        result = {
            "rc_msps": rc_msps,
            "msps": msps,
            "energy": energy,
            "msp": probabilities.max(axis=1),
            "maxlogit": logits.max(axis=1),
            "logit_margin": logit_margin,
            "proto_relative": top5_values[:, 0] - weighted.mean(axis=1),
            "proto_margin": margin,
            "boundary_ratio": -(1.0 - top5_values[:, 0]) / np.maximum(1.0 - top5_values[:, 1], 1e-6),
            "cross_stage_agreement": (stage_winners == deep_winner[:, None]).mean(axis=1),
            "reciprocal_rank_stability": reciprocal_ranks,
            "class_proto_quantile": proto_q.max(axis=1),
            "class_joint_geom": np.sqrt(proto_q * logit_q).max(axis=1),
            "class_joint_min": np.minimum(proto_q, logit_q).max(axis=1),
            "class_joint_harmonic": (2.0 * proto_q * logit_q / np.maximum(proto_q + logit_q, 1e-8)).max(axis=1),
            "diagonal_confidence": -class_diagonal,
            "relative_diagonal": global_diagonal - class_diagonal,
            "diagonal_ratio": -class_diagonal / np.maximum(global_diagonal, 1e-8),
            "normalized_margin": normalized_margin,
            "norm_shell": norm_shell,
            "classifier_alignment": classifier_alignment,
            "barycentric": barycentric,
            "fisher_proto": fisher_proto,
            "proto_logit_same_class": (top1 == logits.argmax(axis=1)).astype(np.float32),
            "deep_support": deep_support,
        }
        for layer in range(len(STAGE_DIMS)):
            result[f"stage_support_{layer}"] = similarity32[:, layer, :].max(axis=1)
        for name, values in result.items():
            values = np.asarray(values, dtype=np.float32)
            if values.shape != (len(features),) or not np.isfinite(values).all():
                raise ValueError(f"invalid atomic component {name}: {values.shape}")
            result[name] = values
        return result


def save_atomic(path: Path, values: dict[str, np.ndarray], labels: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, labels=np.asarray(labels, dtype=np.int64), **values)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    validation_paths = expand(args.validation_shards)
    assert_no_final_access(validation_paths + [args.rc_state, args.rc_config, args.proxy_lock])
    args.output.mkdir(parents=True, exist_ok=True)
    features, labels, paths = load_validation(validation_paths)
    fit_features, fit_labels, source, source_labels, calibration, calibration_labels = split_rows(
        features, labels, paths
    )
    class_means = np.stack([fit_features[fit_labels == c].mean(0) for c in range(1000)])
    all_proxies = generate_id_only_proxies(
        source,
        source_labels,
        fit_features,
        class_means,
        stage_dims=STAGE_DIMS,
    )
    proxy_lock = json.loads(args.proxy_lock.read_text(encoding="utf-8"))
    selected_proxies = proxy_lock["selected_proxies"]

    stats, extra = ReferenceStats.load(args.rc_state)
    rc_locked = json.loads(args.rc_config.read_text(encoding="utf-8"))
    rc_config = rc_locked["hyperparameters"]
    weight, bias = classifier()
    fit_similarity = stage_similarities(
        fit_features, stats.prototypes, device=args.device, batch_size=256
    )
    fit_logits = logits_numpy(fit_features[:, -FINAL_DIM:], weight, bias, device=args.device)
    atomic_fit = fit_atomic_references(
        fit_features,
        fit_labels,
        fit_similarity,
        fit_logits,
        np.asarray(rc_config["weights"], dtype=np.float32),
        STAGE_DIMS,
    )
    np.savez_compressed(args.output / "atomic_fit_state.npz", **atomic_fit.payload())
    extractor = AtomicExtractor(stats, extra, atomic_fit, weight, bias, rc_config, args.device)
    save_atomic(
        args.output / "calibration.npz", extractor.extract(calibration), calibration_labels
    )
    save_atomic(args.output / "id_proxy_reference.npz", extractor.extract(source), source_labels)
    for proxy_name in sorted(all_proxies):
        save_atomic(
            args.output / f"proxy_{proxy_name}.npz",
            extractor.extract(all_proxies[proxy_name]),
            source_labels,
        )
    manifest = {
        "id_only": True,
        "final_benchmark_accessed": False,
        "selected_proxies": selected_proxies,
        "cached_proxy_families": sorted(all_proxies),
        "n_fit": len(fit_features),
        "n_calibration": len(calibration),
        "n_id_proxy_reference": len(source),
        "component_files": {},
    }
    for path in sorted(args.output.glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            manifest["component_files"][path.name] = {
                "sha256": file_sha256(path),
                "keys": sorted(data.files),
            }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
