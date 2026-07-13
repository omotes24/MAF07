#!/usr/bin/env python3
"""Score deterministic mean-view features with a frozen compact IVF-PQ index."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from methods.compact_knn import mean_neighbor_confidence
from reproduce_nnguide import assert_no_final_access, expand


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-shards", nargs="+", required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--k", type=int, default=200)
    parser.add_argument("--nprobe", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--threads", type=int, default=24)
    return parser.parse_args()


def load_mean_features(paths: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    features, labels = [], []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            values = data["features"]
            if values.ndim != 3:
                raise ValueError(f"expected N x V x D features in {path}")
            features.append(values.astype(np.float32).mean(axis=1))
            labels.append(data["labels"].astype(np.int16))
    return np.concatenate(features), np.concatenate(labels)


def main() -> None:
    args = parse_args()
    input_paths = expand(args.input_shards)
    assert_no_final_access(input_paths + [args.index, args.output])
    import faiss

    faiss.omp_set_num_threads(args.threads)
    features, labels = load_mean_features(input_paths)
    index = faiss.read_index(str(args.index))
    confidence = mean_neighbor_confidence(
        index,
        features,
        k=args.k,
        nprobe=args.nprobe,
        batch_size=args.batch_size,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        labels=labels,
        mean_feature=confidence,
        k=np.asarray(args.k, dtype=np.int64),
        nprobe=np.asarray(args.nprobe, dtype=np.int64),
    )
    print(f"saved={args.output} rows={len(labels)} k={args.k} nprobe={args.nprobe}")


if __name__ == "__main__":
    main()
