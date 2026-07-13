#!/usr/bin/env python3
"""Build a frozen train-only IVF-PQ index for compact angular support."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from diagnose_knn_k import load_final
from methods.compact_knn import build_ivfpq, serialized_size
from reproduce_nnguide import assert_no_final_access, expand


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-shards", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nlist", type=int, default=2048)
    parser.add_argument("--pq-m", type=int, default=64)
    parser.add_argument("--pq-bits", type=int, default=8)
    parser.add_argument("--train-samples", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=24)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_paths = expand(args.train_shards)
    assert_no_final_access(train_paths + [args.output])
    import faiss

    faiss.omp_set_num_threads(args.threads)
    features = load_final(train_paths)
    index = build_ivfpq(
        features,
        nlist=args.nlist,
        pq_m=args.pq_m,
        pq_bits=args.pq_bits,
        train_samples=args.train_samples,
        seed=args.seed,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    index_path = args.output / "angular_support.faiss"
    index_bytes = serialized_size(index, index_path)
    float_bank_bytes = int(features.size * features.dtype.itemsize)
    metadata = {
        "id_train_only": True,
        "final_benchmark_accessed": False,
        "n_train": int(len(features)),
        "dimension": int(features.shape[1]),
        "nlist": args.nlist,
        "pq_m": args.pq_m,
        "pq_bits": args.pq_bits,
        "pq_train_samples": min(args.train_samples, len(features)),
        "seed": args.seed,
        "index_bytes": index_bytes,
        "float32_bank_bytes": float_bank_bytes,
        "compression_ratio": float(float_bank_bytes / index_bytes),
        "index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
