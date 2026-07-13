#!/usr/bin/env python3
"""Extract compact class-center distribution components from cached features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from methods.bilateral_distribution import bilateral_components
from reproduce_nnguide import assert_no_final_access, expand


FINAL_DIM = 2048


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=512)
    return parser.parse_args()


def load_features(paths: list[Path]) -> np.ndarray:
    parts = []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            parts.append(data["base_pooled"][:, -FINAL_DIM:])
    return np.concatenate(parts)


def main() -> None:
    args = parse_args()
    paths = expand(args.input)
    assert_no_final_access(paths + [args.state, args.output])
    with np.load(args.state, allow_pickle=False) as state:
        prototypes = state["prototype_3"].astype(np.float32)
    values = bilateral_components(
        load_features(paths), prototypes, device=args.device, batch_size=args.batch_size
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **values)
    args.output.with_suffix(".json").write_text(
        json.dumps(
            {
                "method": "ideal-OOD class-center distribution components",
                "n": len(next(iter(values.values()))),
                "n_class_centers": len(prototypes),
                "target_ood_used_for_fit_or_selection": False,
                "final_benchmark_accessed": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
