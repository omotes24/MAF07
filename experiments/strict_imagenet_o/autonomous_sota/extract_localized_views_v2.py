#!/usr/bin/env python3
"""Extract stricter CAM and activation-energy crops from frozen ResNet50d."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from methods.localized_views_v2 import VIEW_NAMES, localized_views  # noqa: E402
from reproduce_nnguide import assert_no_final_access  # noqa: E402


ImageFile.LOAD_TRUNCATED_IMAGES = True


class PathDataset(Dataset):
    def __init__(self, paths: list[Path]):
        self.paths = paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        from extract_resnet50d_views import normalize_view

        with Image.open(self.paths[index]) as image:
            return normalize_view(image.convert("RGB"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-npz", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--stride", type=int, default=1)
    return parser.parse_args()


def read_paths(cache_path: Path) -> list[Path]:
    rows = [
        Path(json.loads(line)["path"])
        for line in cache_path.with_suffix(".jsonl").read_text(encoding="utf-8").splitlines()
    ]
    if not rows or any(not path.is_file() for path in rows):
        raise FileNotFoundError(f"invalid path manifest for {cache_path}")
    return rows


@torch.inference_mode()
def extract_one(model, cache_path: Path, output_path: Path, args: argparse.Namespace) -> None:
    if args.stride < 1:
        raise ValueError("stride must be positive")
    paths = read_paths(cache_path)[:: args.stride]
    with np.load(cache_path, allow_pickle=False) as data:
        labels = np.asarray(data["labels"], dtype=np.int16)[:: args.stride]
    if len(labels) != len(paths):
        raise ValueError(f"cache/path mismatch for {cache_path}")
    loader = DataLoader(
        PathDataset(paths),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    pooled = []
    for images in loader:
        images = images.to(args.device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            feature = model.forward_features(images)
            logits = model.fc(feature.float().mean(dim=(-2, -1)))
            selected_weight = model.fc.weight[logits.argmax(dim=1)]
            class_cam = torch.einsum("bchw,bc->bhw", feature.float(), selected_weight)
            energy = feature.float().square().mean(dim=1).sqrt()
            views = localized_views(images, class_cam, energy)
            flat = model.forward_features(views.flatten(0, 1))
        pooled.append(
            flat.float().mean(dim=(-2, -1)).reshape(len(images), len(VIEW_NAMES), -1).cpu().half().numpy()
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        base_pooled=np.concatenate(pooled),
        labels=labels,
        view_names=np.asarray(VIEW_NAMES),
    )
    output_path.with_suffix(".json").write_text(
        json.dumps(
            {
                "source_cache": str(cache_path),
                "rows": len(labels),
                "stride": args.stride,
                "model": "timm/resnet50d.ra2_in1k",
                "view_names": VIEW_NAMES,
                "target_ood_used_for_fit_or_selection": False,
                "test_image_sharing": False,
                "final_benchmark_access": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"saved={output_path} rows={len(labels)}", flush=True)


def main() -> None:
    args = parse_args()
    assert_no_final_access(args.cache_npz + [args.output_dir])
    torch.backends.cudnn.benchmark = True
    import timm

    model = timm.create_model("resnet50d", pretrained=True).to(args.device).eval()
    for cache_path in args.cache_npz:
        extract_one(model, cache_path, args.output_dir / cache_path.name, args)


if __name__ == "__main__":
    main()
