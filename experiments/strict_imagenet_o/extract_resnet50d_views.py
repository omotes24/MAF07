#!/usr/bin/env python3
"""Extract fixed-view ResNet50d signals for strict inductive OOD evaluation.

Every output row depends on one input image and fixed network parameters only.
No batch statistic, OOD reference bank, or other test image enters the score.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from tqdm import tqdm


ImageFile.LOAD_TRUNCATED_IMAGES = True
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class Sample:
    path: Path
    label: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--image-list",
        type=Path,
        default=None,
        help="Optional '<relative path> <integer label>' list; duplicate rows are removed.",
    )
    parser.add_argument(
        "--path-jsonl",
        type=Path,
        default=None,
        help="Optional cache JSONL containing one absolute image path per row.",
    )
    parser.add_argument(
        "--label-cache",
        type=Path,
        default=None,
        help="Optional NPZ whose labels correspond row-for-row to --path-jsonl.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="resnet50d")
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--label-mode",
        choices=("parent", "none"),
        default="none",
        help="Use alphabetically sorted parent directories as class labels.",
    )
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def discover_samples(root: Path, label_mode: str) -> list[Sample]:
    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not paths:
        raise FileNotFoundError(f"no images found below {root}")

    if label_mode == "none":
        return [Sample(path, -1) for path in paths]

    parents = sorted({path.parent.name for path in paths})
    class_to_idx = {name: index for index, name in enumerate(parents)}
    return [Sample(path, class_to_idx[path.parent.name]) for path in paths]


def samples_from_list(root: Path, image_list: Path) -> list[Sample]:
    samples: list[Sample] = []
    seen: set[Path] = set()
    for line_number, raw_line in enumerate(image_list.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.rsplit(maxsplit=1)
        try:
            relative, raw_label = parts[0], int(parts[1])
        except (IndexError, ValueError):
            relative, raw_label = line, -1
        relative_path = Path(relative)
        stripped = Path(*relative_path.parts[1:]) if len(relative_path.parts) > 1 else relative_path
        candidates = (
            root / relative_path,
            root / stripped,
            root / "images" / relative_path.name,
            root / relative_path.name,
        )
        path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if path is None:
            raise FileNotFoundError(
                f"{image_list}:{line_number}: cannot resolve {relative!r} below {root}"
            )
        path = path.resolve()
        if path in seen:
            continue
        seen.add(path)
        samples.append(Sample(path, raw_label))
    if not samples:
        raise ValueError(f"no samples loaded from {image_list}")
    return samples


def samples_from_jsonl(path: Path, label_cache: Path | None = None) -> list[Sample]:
    cache_labels = None
    if label_cache is not None:
        with np.load(label_cache, allow_pickle=False) as data:
            cache_labels = data["labels"].astype(np.int64)
    samples = []
    seen = set()
    lines = path.read_text(encoding="utf-8").splitlines()
    for line_number, raw in enumerate(lines, 1):
        row = json.loads(raw)
        image_path = Path(str(row["path"])).resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"{path}:{line_number}: {image_path}")
        if image_path not in seen:
            label = int(row.get("label", -1)) if cache_labels is None else int(cache_labels[line_number - 1])
            samples.append(Sample(image_path, label))
            seen.add(image_path)
    if cache_labels is not None and len(cache_labels) != len(lines):
        raise ValueError("--label-cache length does not match --path-jsonl")
    if not samples:
        raise ValueError(f"no samples loaded from {path}")
    return samples


def center_crop_fraction(image: Image.Image, fraction: float) -> Image.Image:
    width, height = image.size
    crop_width = max(1, round(width * fraction))
    crop_height = max(1, round(height * fraction))
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    return image.crop((left, top, left + crop_width, top + crop_height))


def normalize_view(image: Image.Image) -> torch.Tensor:
    image = TF.resize(image, [224, 224], interpolation=InterpolationMode.BILINEAR)
    tensor = TF.to_tensor(image)
    return TF.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)


def fixed_views(image: Image.Image) -> torch.Tensor:
    """Mild, deterministic views selected without any OOD observations."""
    image = image.convert("RGB")
    zoom_90 = center_crop_fraction(image, 0.90)
    zoom_80 = center_crop_fraction(image, 0.80)
    views = (
        image,
        TF.hflip(image),
        zoom_90,
        TF.hflip(zoom_90),
        zoom_80,
    )
    return torch.stack([normalize_view(view) for view in views])


class FixedViewDataset(Dataset[tuple[torch.Tensor, int, str]]):
    def __init__(self, samples: Iterable[Sample]) -> None:
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        sample = self.samples[index]
        with Image.open(sample.path) as image:
            views = fixed_views(image)
        return views, sample.label, str(sample.path)


def main() -> None:
    args = parse_args()
    import timm

    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")

    if args.image_list is not None and args.path_jsonl is not None:
        raise ValueError("--image-list and --path-jsonl are mutually exclusive")
    if args.path_jsonl is not None:
        all_samples = samples_from_jsonl(args.path_jsonl, args.label_cache)
    elif args.image_list is not None:
        all_samples = samples_from_list(args.root, args.image_list)
    else:
        all_samples = discover_samples(args.root, args.label_mode)
    samples = all_samples[args.shard_index :: args.num_shards]
    if args.limit is not None:
        samples = samples[: args.limit]
    if not samples:
        raise ValueError("selected shard is empty")

    torch.backends.cudnn.benchmark = True
    device = torch.device(args.device)
    model = timm.create_model(args.model, pretrained=True).to(device).eval()
    stages = [model.layer1, model.layer2, model.layer3, model.layer4]
    captured: list[torch.Tensor | None] = [None] * len(stages)
    handles = []

    for stage_index, stage in enumerate(stages):
        def capture(_module, _inputs, output, index=stage_index):
            captured[index] = output

        handles.append(stage.register_forward_hook(capture))

    loader = DataLoader(
        FixedViewDataset(samples),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )

    output: dict[str, list[np.ndarray]] = {
        "features": [],
        "logits": [],
        "base_pooled": [],
        "activation_stats": [],
        "labels": [],
    }
    paths: list[str] = []

    with torch.inference_mode():
        for views, labels, batch_paths in tqdm(loader, desc=f"shard {args.shard_index}"):
            batch_size, num_views = views.shape[:2]
            x = views.flatten(0, 1).to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                spatial = model.forward_features(x)
                features = model.forward_head(spatial, pre_logits=True)
                logits = model.get_classifier()(features)

            pooled_by_stage = []
            stats_by_stage = []
            for stage_output in captured:
                if stage_output is None:
                    raise RuntimeError("a ResNet stage hook did not run")
                pooled = stage_output.float().mean(dim=(-2, -1))
                spatial_std = stage_output.float().std(dim=(-2, -1), unbiased=False).mean(dim=1)
                positive_fraction = (stage_output > 0).float().mean(dim=(1, 2, 3))
                pooled_norm = pooled.norm(dim=1)
                pooled_by_stage.append(pooled)
                stats_by_stage.append(
                    torch.stack((pooled_norm, spatial_std, positive_fraction), dim=-1)
                )

            features = features.reshape(batch_size, num_views, -1)
            logits = logits.reshape(batch_size, num_views, -1)
            base_pooled = torch.cat(
                [pooled.reshape(batch_size, num_views, -1)[:, 0] for pooled in pooled_by_stage],
                dim=-1,
            )
            activation_stats = torch.stack(stats_by_stage, dim=1).reshape(
                batch_size, num_views, len(stages), 3
            )

            output["features"].append(features.cpu().half().numpy())
            output["logits"].append(logits.cpu().half().numpy())
            output["base_pooled"].append(base_pooled.cpu().half().numpy())
            output["activation_stats"].append(activation_stats.cpu().half().numpy())
            output["labels"].append(labels.numpy().astype(np.int16, copy=False))
            paths.extend(batch_paths)
            captured[:] = [None] * len(stages)

    for handle in handles:
        handle.remove()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    arrays = {name: np.concatenate(chunks, axis=0) for name, chunks in output.items()}
    arrays["view_names"] = np.asarray(
        ["full", "full_hflip", "zoom90", "zoom90_hflip", "zoom80"]
    )
    np.savez(args.output, **arrays)
    with args.output.with_suffix(".jsonl").open("w", encoding="utf-8") as stream:
        for path in paths:
            stream.write(json.dumps({"path": path}, ensure_ascii=True) + "\n")

    manifest = {
        "model": args.model,
        "root": str(args.root),
        "image_list": None if args.image_list is None else str(args.image_list),
        "path_jsonl": None if args.path_jsonl is None else str(args.path_jsonl),
        "label_cache": None if args.label_cache is None else str(args.label_cache),
        "num_discovered": len(all_samples),
        "num_rows": len(samples),
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "preprocess": "ViM-compatible direct Resize((224,224)); bilinear; ImageNet normalization",
        "strict_inductive": True,
        "array_shapes": {name: list(value.shape) for name, value in arrays.items()},
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
