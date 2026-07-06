#!/usr/bin/env python
from __future__ import annotations

import argparse
import os

from maf07.patch_features import extract_foreground_patch_cache


def _parse_csv_values(raw: str | None, default: list[str]) -> list[str]:
    if not raw:
        return default
    return [item.strip() for chunk in raw.split(",") for item in chunk.split() if item.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbones", default=os.environ.get("MAF07_BACKBONES", "dinov2_vitb14,dinov2_vitl14"))
    parser.add_argument("--foreground-ratio", type=float, default=float(os.environ.get("MAF07_PALS_FOREGROUND_RATIO", "0.35")))
    parser.add_argument("--max-patches", type=int, default=int(os.environ.get("MAF07_PALS_MAX_FOREGROUND_PATCHES", "128")))
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("MAF07_PALS_EXTRACT_BATCH", "4")))
    parser.add_argument("--device", default=os.environ.get("MAF07_PALS_EXTRACT_DEVICE", os.environ.get("MAF07_TORCH_DEVICE", "cuda")))
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args(argv)

    for backbone in _parse_csv_values(args.backbones, ["dinov2_vitb14", "dinov2_vitl14"]):
        paths = extract_foreground_patch_cache(
            backbone,
            foreground_ratio=args.foreground_ratio,
            max_patches=args.max_patches,
            batch_size=args.batch_size,
            device=args.device,
            resume=not args.no_resume,
        )
        print(paths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
