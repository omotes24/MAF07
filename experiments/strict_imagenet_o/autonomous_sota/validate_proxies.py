#!/usr/bin/env python3
"""Audit ID-only pseudo-OOD families against known legacy method ordering."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "msps_sota"))
sys.path.insert(0, str(ROOT.parent / "simple_fusion"))

from calibration import calibrate_candidates  # noqa: E402
from prototype_stats import ReferenceStats  # noqa: E402
from scores import (  # noqa: E402
    candidate_confidence,
    gather_candidate_support,
    original_msps_confidence,
    select_candidates,
    stage_similarities,
)
from vim_score import VimState  # noqa: E402

from methods.id_proxies import generate_id_only_proxies  # noqa: E402
from methods.nnguide import NNGuideState  # noqa: E402
from reproduce_nnguide import (  # noqa: E402
    assert_no_final_access,
    classifier,
    deterministic_balanced_bank,
    expand,
)


STAGE_DIMS = (256, 512, 1024, 2048)
FINAL_DIM = 2048
LEGACY_MACRO_AUROC = {
    "RC-MSPS": 0.8876363198137707,
    "MSPS": 0.8814656948374291,
    "NNGuide": 0.8535489330198424,
    "ViM": 0.836215,
    "MSP": 0.779758,
    "MaxLogit": 0.753500,
    "Energy": 0.710313,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-shards", nargs="+", required=True)
    parser.add_argument("--train-shards", nargs="+", required=True)
    parser.add_argument("--rc-state", type=Path, required=True)
    parser.add_argument("--rc-config", type=Path, required=True)
    parser.add_argument("--vim-state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--samples-per-class", type=int, default=5)
    return parser.parse_args()


def load_validation(paths: list[Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features, labels, image_paths = [], [], []
    for path in sorted(paths):
        with np.load(path, allow_pickle=False) as shard:
            features.append(shard["base_pooled"])
            labels.append(shard["labels"].astype(np.int64))
        rows = [json.loads(line)["path"] for line in path.with_suffix(".jsonl").read_text().splitlines()]
        image_paths.extend(rows)
    return np.concatenate(features), np.concatenate(labels), np.asarray(image_paths, dtype=str)


def split_validation(
    features: np.ndarray,
    labels: np.ndarray,
    paths: np.ndarray,
    samples_per_class: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    calibration_indices, source_indices = [], []
    for class_id in range(1000):
        indices = np.flatnonzero(labels == class_id)
        ordered = indices[np.argsort(paths[indices])]
        if len(ordered) != 50:
            raise ValueError(f"class {class_id} has {len(ordered)} validation samples, expected 50")
        calibration_indices.extend(ordered[:25])
        source_indices.extend(ordered[25 : 25 + samples_per_class])
    calibration_indices = np.asarray(calibration_indices)
    source_indices = np.asarray(source_indices)
    calibration = features[calibration_indices].astype(np.float32)
    calibration_labels = labels[calibration_indices]
    class_means = np.stack([calibration[calibration_labels == c].mean(0) for c in range(1000)])
    return (
        calibration,
        calibration_labels,
        features[source_indices].astype(np.float32),
        labels[source_indices],
        class_means,
    )


def metric_row(id_confidence: np.ndarray, proxy_confidence: np.ndarray) -> dict[str, float]:
    labels = np.concatenate([np.zeros(len(id_confidence)), np.ones(len(proxy_confidence))])
    ood_score = -np.concatenate([id_confidence, proxy_confidence])
    fpr, tpr, _ = roc_curve(1 - labels, -ood_score)
    index = int(np.flatnonzero(tpr >= 0.95)[0])
    return {
        "AUROC": float(roc_auc_score(labels, ood_score)),
        "FPR95": float(fpr[index]),
        "AUPR_OUT": float(average_precision_score(labels, ood_score)),
    }


class KnownMethodScorer:
    def __init__(self, args: argparse.Namespace, train_paths: list[Path]) -> None:
        self.device = args.device
        self.stats, self.extra = ReferenceStats.load(args.rc_state)
        locked = json.loads(args.rc_config.read_text(encoding="utf-8"))
        self.rc_config = locked["hyperparameters"]
        with np.load(args.vim_state, allow_pickle=False) as data:
            self.vim = VimState.from_mapping(data)
        bank, _ = deterministic_balanced_bank(train_paths, per_class=10)
        self.weight, self.bias = classifier()
        self.nnguide = NNGuideState.fit(
            bank, self.weight, self.bias, k=10, device=self.device
        )

    def score(self, features: np.ndarray) -> dict[str, np.ndarray]:
        similarity = stage_similarities(
            features, self.stats.prototypes, device=self.device, batch_size=256
        )
        known = np.ones(len(self.stats.counts), dtype=bool)
        msps = original_msps_confidence(similarity, known)
        candidates = select_candidates(similarity, known, int(self.rc_config["k"]))
        support = gather_candidate_support(similarity, candidates)
        calibrated = calibrate_candidates(
            support,
            candidates,
            self.extra["class_sorted"],
            self.extra["global_sorted"],
            float(self.rc_config["class_shrinkage"]),
        )
        rc_msps = candidate_confidence(
            calibrated,
            candidates,
            similarity.argmax(axis=2),
            weights=np.asarray(self.rc_config["weights"], dtype=np.float32),
            fusion=str(self.rc_config["fusion"]),
            variance_penalty=float(self.rc_config["lambda"]),
            consensus_reward=float(self.rc_config["eta"]),
        )
        final = features[:, -FINAL_DIM:]
        weight = torch.as_tensor(self.weight, device=self.device)
        bias = torch.as_tensor(self.bias, device=self.device)
        cheap = {"MSP": [], "MaxLogit": [], "Energy": []}
        with torch.inference_mode():
            for begin in range(0, len(final), 512):
                x = torch.as_tensor(final[begin : begin + 512], device=self.device)
                logits = x @ weight.T + bias
                cheap["MSP"].append(logits.softmax(1).amax(1).cpu().numpy())
                cheap["MaxLogit"].append(logits.amax(1).cpu().numpy())
                cheap["Energy"].append(torch.logsumexp(logits, 1).cpu().numpy())
        return {
            "RC-MSPS": rc_msps,
            "MSPS": msps,
            "NNGuide": self.nnguide.confidence(final, device=self.device),
            "ViM": -self.vim.score_ood(final, device=self.device),
            **{name: np.concatenate(parts) for name, parts in cheap.items()},
        }


def stable_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    args = parse_args()
    validation_paths = expand(args.validation_shards)
    train_paths = expand(args.train_shards)
    assert_no_final_access(validation_paths + train_paths + [args.rc_state, args.vim_state])
    args.output.mkdir(parents=True, exist_ok=True)

    features, labels, paths = load_validation(validation_paths)
    calibration, _, source, source_labels, class_means = split_validation(
        features, labels, paths, args.samples_per_class
    )
    proxies = generate_id_only_proxies(
        source,
        source_labels,
        calibration,
        class_means,
        stage_dims=STAGE_DIMS,
    )
    scorer = KnownMethodScorer(args, train_paths)
    id_scores = scorer.score(source)
    rows = []
    score_dir = args.output / "scores"
    score_dir.mkdir(exist_ok=True)
    for proxy_name, proxy_features in proxies.items():
        proxy_scores = scorer.score(proxy_features)
        payload = {}
        for method in LEGACY_MACRO_AUROC:
            payload[f"id_{method}"] = id_scores[method]
            payload[f"proxy_{method}"] = proxy_scores[method]
            rows.append(
                {
                    "proxy": proxy_name,
                    "method": method,
                    "legacy_macro_AUROC": LEGACY_MACRO_AUROC[method],
                    **metric_row(id_scores[method], proxy_scores[method]),
                }
            )
        np.savez_compressed(score_dir / f"{proxy_name}.npz", **payload)

    table = pd.DataFrame(rows)
    table.to_csv(args.output / "proxy_method_metrics.csv", index=False)
    fidelity_rows = []
    actual = pd.Series(LEGACY_MACRO_AUROC)
    for proxy_name, group in table.groupby("proxy", sort=False):
        proxy = group.set_index("method")["AUROC"].reindex(actual.index)
        spearman = float(spearmanr(actual, proxy).statistic)
        kendall = float(kendalltau(actual, proxy).statistic)
        fidelity_rows.append(
            {
                "proxy": proxy_name,
                "spearman": spearman,
                "kendall": kendall,
                "worst_method_proxy_AUROC": float(proxy.min()),
                "mean_proxy_AUROC": float(proxy.mean()),
            }
        )
    fidelity = pd.DataFrame(fidelity_rows).sort_values(
        ["spearman", "kendall"], ascending=False
    )
    positive = fidelity[(fidelity["spearman"] > 0) & (fidelity["kendall"] > 0)]
    selected = positive.head(3)["proxy"].tolist()
    if len(selected) < 2:
        raise RuntimeError("fewer than two ID-only proxy families have positive rank fidelity")
    fidelity["selected"] = fidelity["proxy"].isin(selected)
    fidelity.to_csv(args.output / "proxy_fidelity.csv", index=False)
    lock = {
        "selected_proxies": selected,
        "selection_rule": "top three proxies with positive Spearman and Kendall against seven frozen legacy method rankings",
        "legacy_metric_used_for_infrastructure_only": "macro_AUROC",
        "candidate_selection_may_read_raw_legacy_ood": False,
        "samples_per_class": args.samples_per_class,
        "generator_constants": {
            "interclass_midpoint_lambda": 0.5,
            "interclass_near_lambda": 0.25,
            "radial_multiplier": 1.8,
            "class_direction_multiplier": 0.75,
            "low_variance_fraction": 0.1,
            "low_variance_multiplier": 2.0,
            "channel_mask_stride": 10,
        },
    }
    lock["payload_sha256"] = stable_hash(lock)
    (args.output / "proxy_lock.json").write_text(
        json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(fidelity.to_string(index=False))
    print(json.dumps(lock, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
