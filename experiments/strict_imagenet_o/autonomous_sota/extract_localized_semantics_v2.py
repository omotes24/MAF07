#!/usr/bin/env python3
"""Extract class-semantic evidence from deterministic localized features."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT.parent / "msps_sota"))

from prototype_stats import ReferenceStats  # noqa: E402

from methods.localized_views_v2 import VIEW_NAMES  # noqa: E402
from reproduce_nnguide import assert_no_final_access, classifier, expand  # noqa: E402


FINAL_DIM = 2048


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--localized-shards", nargs="+", required=True)
    parser.add_argument("--canonical-shards", nargs="+", required=True)
    parser.add_argument("--rc-state", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


def load_localized(paths: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    features, labels = [], []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            names = tuple(data["view_names"].astype(str))
            if names != VIEW_NAMES:
                raise ValueError(f"unexpected localized views in {path}: {names}")
            features.append(data["base_pooled"][:, :, -FINAL_DIM:])
            labels.append(data["labels"].astype(np.int16))
    return np.concatenate(features).astype(np.float32), np.concatenate(labels)


def load_canonical(paths: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    features, labels = [], []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            features.append(data["base_pooled"][:, -FINAL_DIM:])
            labels.append(data["labels"].astype(np.int16))
    return np.concatenate(features).astype(np.float32), np.concatenate(labels)


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    localized_paths = expand(args.localized_shards)
    canonical_paths = expand(args.canonical_shards)
    assert_no_final_access(localized_paths + canonical_paths + [args.rc_state, args.output])
    localized, labels = load_localized(localized_paths)
    canonical, canonical_labels = load_canonical(canonical_paths)
    if not np.array_equal(labels, canonical_labels) or len(localized) != len(canonical):
        raise ValueError("localized and canonical rows are not aligned")

    weight_np, bias_np = classifier()
    weight = torch.as_tensor(weight_np, dtype=torch.float32, device=args.device)
    bias = torch.as_tensor(bias_np, dtype=torch.float32, device=args.device)
    prototypes_np = ReferenceStats.load(args.rc_state)[0].prototypes[-1]
    prototypes = F.normalize(
        torch.as_tensor(prototypes_np, dtype=torch.float32, device=args.device), dim=1
    )
    output = {
        name: []
        for name in (
            "prototype_max",
            "canonical_class_support",
            "logit_max",
            "energy",
            "msp",
            "class_agreement",
            "canonical_cosine",
        )
    }
    for begin in range(0, len(localized), args.batch_size):
        crop = torch.as_tensor(
            localized[begin : begin + args.batch_size], dtype=torch.float32, device=args.device
        )
        whole = torch.as_tensor(
            canonical[begin : begin + args.batch_size], dtype=torch.float32, device=args.device
        )
        whole_logits = whole @ weight.T + bias
        whole_prediction = whole_logits.argmax(dim=1)
        crop_logits = torch.einsum("bvd,cd->bvc", crop, weight) + bias
        crop_prediction = crop_logits.argmax(dim=2)
        crop_unit = F.normalize(crop, dim=2)
        whole_unit = F.normalize(whole, dim=1)
        prototype_similarity = torch.einsum("bvd,cd->bvc", crop_unit, prototypes)
        selected_prototype = prototypes[whole_prediction]
        values = {
            "prototype_max": prototype_similarity.amax(dim=2),
            "canonical_class_support": torch.einsum(
                "bvd,bd->bv", crop_unit, selected_prototype
            ),
            "logit_max": crop_logits.amax(dim=2),
            "energy": torch.logsumexp(crop_logits, dim=2),
            "msp": crop_logits.softmax(dim=2).amax(dim=2),
            "class_agreement": (crop_prediction == whole_prediction[:, None]).float(),
            "canonical_cosine": torch.einsum("bvd,bd->bv", crop_unit, whole_unit),
        }
        for name, value in values.items():
            output[name].append(value.cpu().numpy().astype(np.float32))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        labels=labels,
        view_names=np.asarray(VIEW_NAMES),
        **{name: np.concatenate(parts) for name, parts in output.items()},
    )
    print(f"saved={args.output} rows={len(labels)} views={len(VIEW_NAMES)}", flush=True)


if __name__ == "__main__":
    main()
