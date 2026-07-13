#!/usr/bin/env python3
"""Extract CAM-localized views for strict inductive frozen-ResNet OOD scoring."""

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

from reproduce_nnguide import assert_no_final_access  # noqa: E402


ImageFile.LOAD_TRUNCATED_IMAGES = True
MASS_FRACTION = 0.8
GRID_PADDING = 1
VIEW_NAMES = ("cam_crop", "cam_soft_mask", "cam_bbox_mask")


class PathDataset(Dataset):
    def __init__(self, paths: list[Path]):
        self.paths = paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        from extract_resnet50d_views import normalize_view

        path = self.paths[index]
        with Image.open(path) as image:
            tensor = normalize_view(image.convert("RGB"))
        return tensor, str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-npz", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def read_paths(cache_path: Path) -> list[Path]:
    jsonl = cache_path.with_suffix(".jsonl")
    paths = [
        Path(json.loads(line)["path"])
        for line in jsonl.read_text(encoding="utf-8").splitlines()
    ]
    if not paths or any(not path.is_file() for path in paths):
        raise FileNotFoundError(f"invalid path manifest {jsonl}")
    return paths


def mass_bbox(
    cam: torch.Tensor,
    *,
    mass_fraction: float = MASS_FRACTION,
    padding: int = GRID_PADDING,
) -> tuple[int, int, int, int]:
    """Return an inclusive-exclusive grid box covering the strongest CAM mass."""
    if cam.ndim != 2:
        raise ValueError("cam must be a matrix")
    if not 0.0 < mass_fraction <= 1.0:
        raise ValueError("mass_fraction must be in (0, 1]")
    height, width = cam.shape
    positive = cam.float().clamp_min(0)
    if positive.sum() <= 1e-12:
        return 0, height, 0, width
    ordered, indices = positive.flatten().sort(descending=True)
    cumulative = ordered.cumsum(0)
    count = int(torch.searchsorted(cumulative, mass_fraction * cumulative[-1]).item()) + 1
    selected = indices[:count]
    ys = selected // width
    xs = selected % width
    y0 = max(0, int(ys.min()) - padding)
    y1 = min(height, int(ys.max()) + 1 + padding)
    x0 = max(0, int(xs.min()) - padding)
    x1 = min(width, int(xs.max()) + 1 + padding)
    return y0, y1, x0, x1


def saliency_views(
    images: torch.Tensor,
    cams: torch.Tensor,
    *,
    mass_fraction: float = MASS_FRACTION,
    padding: int = GRID_PADDING,
) -> torch.Tensor:
    """Create crop, soft-mask, and bbox-mask views from each sample's own CAM."""
    if images.ndim != 4 or cams.ndim != 3 or len(images) != len(cams):
        raise ValueError("expected images BxCxHxW and cams Bxhxw")
    image_height, image_width = images.shape[-2:]
    positive = cams.float().clamp_min(0).unsqueeze(1)
    soft = torch.nn.functional.interpolate(
        positive, size=(image_height, image_width), mode="bilinear", align_corners=False
    )
    soft /= soft.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
    soft_masked = images * soft

    crops, hard_masked = [], []
    grid_height, grid_width = cams.shape[-2:]
    for image, cam in zip(images, cams):
        y0, y1, x0, x1 = mass_bbox(cam, mass_fraction=mass_fraction, padding=padding)
        py0 = (y0 * image_height) // grid_height
        py1 = max(py0 + 1, (y1 * image_height + grid_height - 1) // grid_height)
        px0 = (x0 * image_width) // grid_width
        px1 = max(px0 + 1, (x1 * image_width + grid_width - 1) // grid_width)
        crop = torch.nn.functional.interpolate(
            image[:, py0:py1, px0:px1].unsqueeze(0),
            size=(image_height, image_width),
            mode="bilinear",
            align_corners=False,
        )[0]
        hard = torch.zeros_like(image)
        hard[:, py0:py1, px0:px1] = image[:, py0:py1, px0:px1]
        crops.append(crop)
        hard_masked.append(hard)
    return torch.stack(
        (torch.stack(crops), soft_masked, torch.stack(hard_masked)), dim=1
    )


@torch.inference_mode()
def extract_one(model, cache_path: Path, output_path: Path, args: argparse.Namespace) -> None:
    paths = read_paths(cache_path)
    with np.load(cache_path, allow_pickle=False) as data:
        labels = np.asarray(data["labels"], dtype=np.int16)
    if len(labels) != len(paths):
        raise ValueError(f"cache/path mismatch for {cache_path}")
    if args.limit is not None:
        paths = paths[: args.limit]
        labels = labels[: args.limit]
    loader = DataLoader(
        PathDataset(paths),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    stages = (model.layer1, model.layer2, model.layer3, model.layer4)
    captured: list[torch.Tensor | None] = [None] * len(stages)
    handles = []
    for stage_index, stage in enumerate(stages):
        def capture(_module, _inputs, value, index=stage_index):
            captured[index] = value

        handles.append(stage.register_forward_hook(capture))

    pooled_rows: list[np.ndarray] = []
    try:
        for images, _ in loader:
            images = images.to(args.device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                model.forward_features(images)
            feature4 = captured[-1]
            if feature4 is None:
                raise RuntimeError("canonical layer4 feature was not captured")
            original = feature4.float()
            prediction = model.fc(original.mean(dim=(-2, -1))).argmax(dim=1)
            selected_weight = model.fc.weight[prediction]
            cams = torch.einsum("bchw,bc->bhw", original, selected_weight)
            views = saliency_views(images, cams)
            batch_size, view_count = views.shape[:2]
            captured[:] = [None] * len(stages)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                model.forward_features(views.flatten(0, 1))
            if any(value is None for value in captured):
                raise RuntimeError("a transformed stage feature was not captured")
            pooled = torch.cat(
                [value.float().mean(dim=(-2, -1)) for value in captured if value is not None],
                dim=1,
            ).reshape(batch_size, view_count, -1)
            pooled_rows.append(pooled.cpu().half().numpy())
            captured[:] = [None] * len(stages)
    finally:
        for handle in handles:
            handle.remove()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        base_pooled=np.concatenate(pooled_rows),
        labels=labels,
        view_names=np.asarray(VIEW_NAMES),
    )
    output_path.with_suffix(".json").write_text(
        json.dumps(
            {
                "source_cache": str(cache_path),
                "rows": len(paths),
                "model": "timm/resnet50d.ra2_in1k",
                "view_names": VIEW_NAMES,
                "cam_mass_fraction": MASS_FRACTION,
                "cam_grid_padding": GRID_PADDING,
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
    print(f"saved={output_path} rows={len(paths)}", flush=True)


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
