#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from maf07.features import extract_feature_cache


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbones", required=True)
    parser.add_argument(
        "--manifest", default="results/manifest.content_cleaned_conservative.csv"
    )
    parser.add_argument("--cache-dir", default="results/features")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    backbones = [
        value.strip()
        for value in args.backbones.replace(" ", ",").split(",")
        if value.strip()
    ]
    written = []
    for backbone in backbones:
        paths = extract_feature_cache(
            backbone,
            manifest_path=args.manifest,
            cache_dir=args.cache_dir,
            batch_size=args.batch_size,
            device=args.device,
            resume=not args.force,
        )
        written.append(
            {
                "backbone": backbone,
                "feature_path": str(paths.feature_path),
                "metadata_path": str(paths.metadata_path),
            }
        )
    print(json.dumps({"feature_caches": written}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
