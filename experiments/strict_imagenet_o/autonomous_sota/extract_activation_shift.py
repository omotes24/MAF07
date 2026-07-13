#!/usr/bin/env python3
"""Extract fixed gradient-perturbation atoms from ResNet50d images."""

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

from extract_resnet50d_views import normalize_view  # noqa: E402
from methods.activation_shift import (  # noqa: E402
    PERTURB_FRACTION,
    PERTURB_MAGNITUDE,
    activation_shift_statistics,
    low_gradient_perturb,
)
from reproduce_nnguide import assert_no_final_access  # noqa: E402


ImageFile.LOAD_TRUNCATED_IMAGES = True


class CachePathDataset(Dataset):
    def __init__(self, cache_path: Path, stride: int = 1) -> None:
        rows = [
            Path(json.loads(line)["path"])
            for line in cache_path.with_suffix(".jsonl").read_text(encoding="utf-8").splitlines()
        ]
        with np.load(cache_path, allow_pickle=False) as cache:
            labels = cache["labels"].astype(np.int64)
        if len(rows) != len(labels):
            raise ValueError(f"cache/path mismatch for {cache_path}")
        indices = np.arange(len(rows))[::stride]
        self.paths = [rows[index] for index in indices]
        self.labels = labels[indices]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        path = self.paths[index]
        with Image.open(path) as image:
            tensor = normalize_view(image.convert("RGB"))
        return tensor, int(self.labels[index])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-npz", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--stride", type=int, default=1)
    return parser.parse_args()


def score_batch(model: torch.nn.Module, images: torch.Tensor) -> dict[str, torch.Tensor]:
    x = images.detach().requires_grad_(True)
    spatial = model.forward_features(x)
    feature = model.forward_head(spatial, pre_logits=True)
    logits = model.get_classifier()(feature)
    prediction = logits.detach().argmax(dim=1)
    selected = logits[torch.arange(len(logits), device=logits.device), prediction]
    gradient = torch.autograd.grad(selected.sum(), x, only_inputs=True)[0]
    perturbed = low_gradient_perturb(x, gradient)
    with torch.no_grad():
        perturbed_spatial = model.forward_features(perturbed)
        perturbed_feature = model.forward_head(perturbed_spatial, pre_logits=True)
        statistics = activation_shift_statistics(feature.detach(), perturbed_feature)
    return {"features": feature.detach(), **statistics}


def extract_one(model, cache_path: Path, output_path: Path, args: argparse.Namespace) -> None:
    dataset = CachePathDataset(cache_path, stride=args.stride)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    chunks: dict[str, list[np.ndarray]] = {}
    labels: list[np.ndarray] = []
    for index, (images, batch_labels) in enumerate(loader):
        values = score_batch(model, images.to(args.device, non_blocking=True))
        for name, tensor in values.items():
            dtype = np.float16 if name == "features" else np.float32
            chunks.setdefault(name, []).append(tensor.cpu().numpy().astype(dtype))
        labels.append(batch_labels.numpy().astype(np.int64))
        if index % 50 == 0:
            print(f"{cache_path.name}: batches={index + 1}/{len(loader)}", flush=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        labels=np.concatenate(labels),
        **{name: np.concatenate(parts) for name, parts in chunks.items()},
    )
    output_path.with_suffix(".json").write_text(
        json.dumps(
            {
                "source_cache": str(cache_path),
                "rows": len(dataset),
                "stride": args.stride,
                "perturb_fraction": PERTURB_FRACTION,
                "perturb_magnitude": PERTURB_MAGNITUDE,
                "target_ood_used_for_fit_or_selection": False,
                "test_image_sharing": False,
                "final_benchmark_accessed": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"saved={output_path} rows={len(dataset)}", flush=True)


def main() -> None:
    args = parse_args()
    if args.stride < 1:
        raise ValueError("stride must be positive")
    assert_no_final_access(args.cache_npz + [args.output_dir])
    torch.backends.cudnn.benchmark = True
    import timm

    model = timm.create_model("resnet50d", pretrained=True).to(args.device).eval()
    model.requires_grad_(False)
    for cache_path in args.cache_npz:
        extract_one(model, cache_path, args.output_dir / cache_path.name, args)


if __name__ == "__main__":
    main()
