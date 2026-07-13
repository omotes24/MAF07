#!/usr/bin/env python3
"""Extract full PULSE/MSPS atoms for deterministic ID-only image proxies."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from extract_resnet50d_views import IMAGENET_MEAN, IMAGENET_STD, fixed_views  # noqa: E402
from methods.activation_shift import activation_shift_statistics, low_gradient_perturb  # noqa: E402
from methods.localized_views_v2 import VIEW_NAMES, localized_views  # noqa: E402
from reproduce_nnguide import assert_no_final_access, expand  # noqa: E402


ImageFile.LOAD_TRUNCATED_IMAGES = True
FAMILIES = ("canonical", "patch_shuffle4", "center_cutmix", "phase_mix")
STAGE_DIMS = (256, 512, 1024, 2048)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-shards", nargs="+", required=True)
    parser.add_argument("--family", choices=FAMILIES, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--samples-per-class", type=int, default=1)
    parser.add_argument("--sample-offset", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def selected_rows(
    cache_paths: list[Path], samples_per_class: int, sample_offset: int
) -> tuple[list[Path], list[Path], np.ndarray]:
    rows: dict[int, list[Path]] = {class_id: [] for class_id in range(1000)}
    for cache_path in cache_paths:
        with np.load(cache_path, allow_pickle=False) as data:
            labels = data["labels"].astype(np.int64)
        paths = [
            Path(json.loads(line)["path"])
            for line in cache_path.with_suffix(".jsonl").read_text(encoding="utf-8").splitlines()
        ]
        if len(paths) != len(labels):
            raise ValueError(f"cache/path mismatch for {cache_path}")
        for label, path in zip(labels, paths):
            rows[int(label)].append(path)
    chosen, labels = [], []
    by_class = {}
    for class_id in range(1000):
        paths = sorted(rows[class_id])
        stop = sample_offset + samples_per_class
        if len(paths) < stop:
            raise ValueError(f"class {class_id} has {len(paths)} images, need {stop}")
        by_class[class_id] = paths[sample_offset:stop]
        chosen.extend(by_class[class_id])
        labels.extend([class_id] * samples_per_class)
    partners = []
    for class_id in range(1000):
        partner = by_class[(class_id + 1) % 1000]
        partners.extend(partner)
    return chosen, partners, np.asarray(labels, dtype=np.int16)


def resize_rgb(image: Image.Image) -> Image.Image:
    return TF.resize(image.convert("RGB"), [224, 224], interpolation=InterpolationMode.BILINEAR)


def patch_shuffle4(image: Image.Image) -> Image.Image:
    image = resize_rgb(image)
    permutation = tuple((5 * index + 3) % 16 for index in range(16))
    output = Image.new("RGB", image.size)
    patch = 56
    for destination, source in enumerate(permutation):
        source_x, source_y = (source % 4) * patch, (source // 4) * patch
        destination_x, destination_y = (destination % 4) * patch, (destination // 4) * patch
        tile = image.crop((source_x, source_y, source_x + patch, source_y + patch))
        output.paste(tile, (destination_x, destination_y))
    return output


def center_cutmix(image: Image.Image, partner: Image.Image) -> Image.Image:
    output = resize_rgb(image).copy()
    other = resize_rgb(partner)
    output.paste(other.crop((56, 56, 168, 168)), (56, 56))
    return output


def phase_mix(image: Image.Image, partner: Image.Image) -> Image.Image:
    source = TF.to_tensor(resize_rgb(image)).float()
    other = TF.to_tensor(resize_rgb(partner)).float()
    source_frequency = torch.fft.rfft2(source, norm="ortho")
    other_frequency = torch.fft.rfft2(other, norm="ortho")
    mixed_frequency = source_frequency.abs() * torch.exp(1j * torch.angle(other_frequency))
    mixed = torch.fft.irfft2(mixed_frequency, s=source.shape[-2:], norm="ortho").clamp(0.0, 1.0)
    return TF.to_pil_image(mixed)


def transformed(image: Image.Image, partner: Image.Image, family: str) -> Image.Image:
    if family == "canonical":
        return image.convert("RGB")
    if family == "patch_shuffle4":
        return patch_shuffle4(image)
    if family == "center_cutmix":
        return center_cutmix(image, partner)
    if family == "phase_mix":
        return phase_mix(image, partner)
    raise KeyError(family)


class ProxyDataset(Dataset):
    def __init__(self, paths: list[Path], partners: list[Path], labels: np.ndarray, family: str):
        self.paths, self.partners, self.labels, self.family = paths, partners, labels, family

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        with Image.open(self.paths[index]) as image, Image.open(self.partners[index]) as partner:
            views = fixed_views(transformed(image, partner, self.family))
        return views, int(self.labels[index]), str(self.paths[index])


def normalized_to_unit(images: torch.Tensor) -> torch.Tensor:
    mean = torch.as_tensor(IMAGENET_MEAN, dtype=images.dtype, device=images.device)[None, :, None, None]
    std = torch.as_tensor(IMAGENET_STD, dtype=images.dtype, device=images.device)[None, :, None, None]
    return (images * std + mean).clamp(0.0, 1.0)


def extract_batch(model, views: torch.Tensor, captured: list[torch.Tensor | None]):
    canonical = views[:, 0].detach().requires_grad_(True)
    spatial = model.forward_features(canonical)
    feature = model.forward_head(spatial, pre_logits=True)
    logits = model.get_classifier()(feature)
    prediction = logits.detach().argmax(dim=1)
    selected = logits[torch.arange(len(logits), device=logits.device), prediction]
    gradient = torch.autograd.grad(selected.sum(), canonical, only_inputs=True)[0]
    if any(value is None for value in captured):
        raise RuntimeError("a ResNet stage hook did not run")
    pooled = torch.cat(
        [value.detach().float().mean(dim=(-2, -1)) for value in captured if value is not None], dim=1
    )
    perturbed = low_gradient_perturb(canonical, gradient)
    with torch.no_grad():
        perturbed_spatial = model.forward_features(perturbed)
        perturbed_feature = model.forward_head(perturbed_spatial, pre_logits=True)
        shift = activation_shift_statistics(feature.detach(), perturbed_feature)["shift_ratio"]

        selected_weight = model.get_classifier().weight[prediction]
        class_cam = torch.einsum("bchw,bc->bhw", spatial.detach().float(), selected_weight)
        energy = spatial.detach().float().square().mean(dim=1).sqrt()
        crop_views = localized_views(canonical.detach(), class_cam, energy)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            crop_spatial = model.forward_features(crop_views.flatten(0, 1))
            crop_feature = model.forward_head(crop_spatial, pre_logits=True)
            extra_spatial = model.forward_features(views[:, 1:].flatten(0, 1))
            extra_feature = model.forward_head(extra_spatial, pre_logits=True)
        localized = crop_feature.float().reshape(len(views), len(VIEW_NAMES), -1)
        extra = extra_feature.float().reshape(len(views), 4, -1)
        meanview = torch.cat((feature.detach()[:, None], extra), dim=1).mean(dim=1)
    captured[:] = [None] * 4
    return pooled, shift, localized, meanview


def main() -> None:
    args = parse_args()
    cache_paths = expand(args.validation_shards)
    assert_no_final_access(cache_paths + [args.output])
    paths, partners, labels = selected_rows(
        cache_paths, args.samples_per_class, args.sample_offset
    )
    dataset = ProxyDataset(paths, partners, labels, args.family)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    import timm

    torch.backends.cudnn.benchmark = True
    model = timm.create_model("resnet50d", pretrained=True).to(args.device).eval()
    model.requires_grad_(False)
    captured: list[torch.Tensor | None] = [None] * 4
    handles = []
    for index, stage in enumerate((model.layer1, model.layer2, model.layer3, model.layer4)):
        def capture(_module, _inputs, output, position=index):
            captured[position] = output

        handles.append(stage.register_forward_hook(capture))

    output = {name: [] for name in ("base_pooled", "shift_ratio", "localized", "meanview")}
    output_labels, output_paths = [], []
    for step, (views, batch_labels, batch_paths) in enumerate(loader):
        values = extract_batch(model, views.to(args.device, non_blocking=True), captured)
        for name, value in zip(output, values):
            dtype = np.float32 if name == "shift_ratio" else np.float16
            output[name].append(value.cpu().numpy().astype(dtype))
        output_labels.append(batch_labels.numpy().astype(np.int16))
        output_paths.extend(batch_paths)
        if step % 20 == 0:
            print(f"family={args.family} batches={step + 1}/{len(loader)}", flush=True)
    for handle in handles:
        handle.remove()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        labels=np.concatenate(output_labels),
        **{name: np.concatenate(parts) for name, parts in output.items()},
    )
    args.output.with_suffix(".json").write_text(
        json.dumps(
            {
                "family": args.family,
                "model": "timm/resnet50d.ra2_in1k",
                "rows": len(dataset),
                "sample_offset": args.sample_offset,
                "samples_per_class": args.samples_per_class,
                "target_ood_accessed": False,
                "final_benchmark_accessed": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    args.output.with_suffix(".jsonl").write_text(
        "".join(json.dumps({"path": path}) + "\n" for path in output_paths), encoding="utf-8"
    )
    print(f"saved={args.output} rows={len(dataset)} family={args.family}", flush=True)


if __name__ == "__main__":
    main()
