#!/usr/bin/env python3
"""Extract one immutable final-suite shard and score it with frozen PULSE."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT.parent / "msps_sota"))

from extract_resnet50d_views import fixed_views  # noqa: E402
from prototype_stats import ReferenceStats  # noqa: E402

from build_atomic_cache import AtomicExtractor  # noqa: E402
from final_lock import (  # noqa: E402
    artifact_path,
    sha256_file,
    sha256_model_state,
    verify_locked_config,
)
from methods.atomic_components import AtomicFit  # noqa: E402
from methods.activation_shift import (  # noqa: E402
    activation_shift_statistics,
    low_gradient_perturb,
)
from methods.bilateral_distribution import bilateral_components  # noqa: E402
from methods.compact_knn import mean_neighbor_confidence  # noqa: E402
from methods.localized_views_v2 import VIEW_NAMES, localized_views  # noqa: E402
from methods.pulse import PulseState  # noqa: E402


ImageFile.LOAD_TRUNCATED_IMAGES = True
FINAL_DIM = 2048


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--locked-config", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--device", required=True)
    return parser.parse_args()


def manifest_rows(path: Path, dataset_name: str) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected = [row for row in rows if row["dataset"] == dataset_name]
    if not selected:
        raise ValueError(f"no manifest rows for {dataset_name}")
    return selected


def verify_image(path: Path, row: dict[str, Any]) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != int(row["bytes"]):
        raise RuntimeError(f"image size changed after preregistration: {path}")
    if sha256_file(path) != row["sha256"]:
        raise RuntimeError(f"image hash changed after preregistration: {path}")


class LockedImageDataset(Dataset):
    def __init__(self, image_root: Path, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.paths = [image_root / row["relative_path"] for row in rows]
        for path, row in zip(self.paths, rows):
            verify_image(path, row)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        path = self.paths[index]
        with Image.open(path) as image:
            views = fixed_views(image)
        return views, -1, self.rows[index]["relative_path"]


def make_loader(dataset: Dataset, batch_size: int, workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )


def extract_multiview(
    model: torch.nn.Module,
    dataset: Dataset,
    captured: list[torch.Tensor | None],
    *,
    device: str,
    batch_size: int,
    workers: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    pooled_chunks, mean_chunks, output_paths = [], [], []
    loader = make_loader(dataset, batch_size, workers)
    for step, (views, _labels, paths) in enumerate(loader):
        views = views.to(device, non_blocking=True)
        rows, num_views = views.shape[:2]
        with torch.inference_mode(), torch.autocast(
            device_type=views.device.type, dtype=torch.float16
        ):
            spatial = model.forward_features(views.flatten(0, 1))
            feature = model.forward_head(spatial, pre_logits=True)
        if any(value is None for value in captured):
            raise RuntimeError("a ResNet stage hook did not run")
        pooled = torch.cat(
            [
                value.float()
                .mean(dim=(-2, -1))
                .reshape(rows, num_views, -1)[:, 0]
                for value in captured
                if value is not None
            ],
            dim=1,
        )
        meanview = feature.float().reshape(rows, num_views, -1).mean(dim=1)
        pooled_chunks.append(pooled.cpu().half().numpy())
        mean_chunks.append(meanview.cpu().half().numpy())
        output_paths.extend(paths)
        captured[:] = [None] * 4
        if step % 50 == 0:
            print(f"multiview batches={step + 1}/{len(loader)}", flush=True)
    return np.concatenate(pooled_chunks), np.concatenate(mean_chunks), output_paths


def extract_shift(
    model: torch.nn.Module,
    dataset: Dataset,
    captured: list[torch.Tensor | None],
    *,
    device: str,
    batch_size: int,
    workers: int,
) -> tuple[np.ndarray, list[str]]:
    chunks, output_paths = [], []
    loader = make_loader(dataset, batch_size, workers)
    for step, (views, _labels, paths) in enumerate(loader):
        canonical = views[:, 0].to(device, non_blocking=True).detach().requires_grad_(True)
        spatial = model.forward_features(canonical)
        feature = model.forward_head(spatial, pre_logits=True)
        logits = model.get_classifier()(feature)
        prediction = logits.detach().argmax(dim=1)
        selected = logits[torch.arange(len(logits), device=canonical.device), prediction]
        gradient = torch.autograd.grad(selected.sum(), canonical, only_inputs=True)[0]
        perturbed = low_gradient_perturb(canonical, gradient)
        with torch.no_grad():
            perturbed_spatial = model.forward_features(perturbed)
            perturbed_feature = model.forward_head(perturbed_spatial, pre_logits=True)
            shift = activation_shift_statistics(feature.detach(), perturbed_feature)[
                "shift_ratio"
            ]
        chunks.append(shift.cpu().numpy().astype(np.float32))
        output_paths.extend(paths)
        captured[:] = [None] * 4
        if step % 50 == 0:
            print(f"activation batches={step + 1}/{len(loader)}", flush=True)
    return np.concatenate(chunks), output_paths


def extract_localized(
    model: torch.nn.Module,
    dataset: Dataset,
    captured: list[torch.Tensor | None],
    *,
    device: str,
    batch_size: int,
    workers: int,
) -> tuple[np.ndarray, list[str]]:
    chunks, output_paths = [], []
    loader = make_loader(dataset, batch_size, workers)
    for step, (views, _labels, paths) in enumerate(loader):
        canonical = views[:, 0].to(device, non_blocking=True)
        with torch.inference_mode(), torch.autocast(
            device_type=canonical.device.type, dtype=torch.float16
        ):
            feature_map = model.forward_features(canonical)
            logits = model.get_classifier()(feature_map.float().mean(dim=(-2, -1)))
            selected_weight = model.get_classifier().weight[logits.argmax(dim=1)]
            class_cam = torch.einsum(
                "bchw,bc->bhw", feature_map.float(), selected_weight
            )
            energy = feature_map.float().square().mean(dim=1).sqrt()
            crops = localized_views(canonical, class_cam, energy)
            crop_spatial = model.forward_features(crops.flatten(0, 1))
            crop_feature = model.forward_head(crop_spatial, pre_logits=True)
        chunks.append(
            crop_feature.float()
            .reshape(len(canonical), len(VIEW_NAMES), -1)
            .cpu()
            .half()
            .numpy()
        )
        output_paths.extend(paths)
        captured[:] = [None] * 4
        if step % 50 == 0:
            print(f"localized batches={step + 1}/{len(loader)}", flush=True)
    return np.concatenate(chunks), output_paths


def localized_prototype_confidence(
    localized: np.ndarray,
    prototypes: np.ndarray,
    device: str,
    batch_size: int = 256,
) -> np.ndarray:
    bank = F.normalize(
        torch.as_tensor(prototypes, dtype=torch.float32, device=device), dim=1
    )
    output = []
    with torch.inference_mode():
        for begin in range(0, len(localized), batch_size):
            crop = F.normalize(
                torch.as_tensor(
                    localized[begin : begin + batch_size],
                    dtype=torch.float32,
                    device=device,
                ),
                dim=2,
            )
            output.append(
                torch.einsum("bvd,cd->bvc", crop, bank)
                .amax(dim=(1, 2))
                .cpu()
                .numpy()
            )
    return np.concatenate(output).astype(np.float32, copy=False)


def make_atomic_extractor(
    lock: dict[str, Any], model: torch.nn.Module, device: str
) -> tuple[AtomicExtractor, np.ndarray]:
    stats, extra = ReferenceStats.load(artifact_path(lock, "rc_state"))
    with np.load(artifact_path(lock, "atomic_fit"), allow_pickle=False) as data:
        fit = AtomicFit.from_mapping({key: data[key] for key in data.files})
    rc_config = json.loads(
        artifact_path(lock, "rc_config").read_text(encoding="utf-8")
    )["hyperparameters"]
    classifier = model.get_classifier()
    weight = classifier.weight.detach().cpu().numpy().astype(np.float32)
    bias = classifier.bias.detach().cpu().numpy().astype(np.float32)
    extractor = AtomicExtractor(stats, extra, fit, weight, bias, rc_config, device)
    return extractor, stats.prototypes[-1]


def main() -> None:
    args = parse_args()
    started = time.time()
    lock, config_sha = verify_locked_config(args.locked_config)
    specs = {item["key"]: item for item in lock["final_suite"]["datasets"]}
    if args.dataset not in specs:
        raise ValueError(f"dataset must be one of {sorted(specs)}")
    num_shards = int(lock["execution"]["num_shards"])
    if not 0 <= args.shard_index < num_shards:
        raise ValueError("invalid shard index")
    if args.device != lock["execution"]["devices"][args.shard_index]:
        raise ValueError("device does not match the locked shard assignment")

    spec = specs[args.dataset]
    rows = manifest_rows(
        artifact_path(lock, "final_manifest"), str(spec["manifest_name"])
    )
    if len(rows) != int(spec["expected_images"]):
        raise RuntimeError(
            f"{args.dataset}: manifest has {len(rows)} rows, "
            f"expected {spec['expected_images']}"
        )
    rows = rows[args.shard_index :: num_shards]
    expected_shard_rows = len(rows)
    image_root = Path(lock["final_suite"]["data_root"]) / "images_largescale"
    dataset = LockedImageDataset(image_root, rows)
    import faiss
    import timm

    faiss.omp_set_num_threads(int(lock["execution"]["faiss_threads_per_process"]))
    torch.backends.cudnn.benchmark = True
    model = timm.create_model(lock["model"]["timm_name"], pretrained=True)
    observed_model_sha = sha256_model_state(model)
    if observed_model_sha != lock["model"]["state_dict_sha256"]:
        raise RuntimeError(
            f"model hash mismatch: expected={lock['model']['state_dict_sha256']} "
            f"observed={observed_model_sha}"
        )
    model = model.to(args.device).eval()
    model.requires_grad_(False)
    atomic, prototypes = make_atomic_extractor(lock, model, args.device)
    pulse_state = PulseState.load(artifact_path(lock, "pulse_state"))
    knn_index = faiss.read_index(str(artifact_path(lock, "compact_index")))

    captured: list[torch.Tensor | None] = [None] * 4
    handles = []
    for position, stage in enumerate(
        (model.layer1, model.layer2, model.layer3, model.layer4)
    ):
        def capture(_module, _inputs, output, index=position):
            captured[index] = output

        handles.append(stage.register_forward_hook(capture))

    workers = int(lock["execution"]["workers_per_gpu"])
    # Preserve the original standalone extractor order for CUDA kernel selection.
    shift, shift_paths = extract_shift(
        model,
        dataset,
        captured,
        device=args.device,
        batch_size=int(lock["execution"]["activation_batch_size"]),
        workers=workers,
    )
    pooled, meanview, multiview_paths = extract_multiview(
        model,
        dataset,
        captured,
        device=args.device,
        batch_size=int(lock["execution"]["multiview_batch_size"]),
        workers=workers,
    )
    localized_features, localized_paths = extract_localized(
        model,
        dataset,
        captured,
        device=args.device,
        batch_size=int(lock["execution"]["localized_batch_size"]),
        workers=workers,
    )
    for handle in handles:
        handle.remove()
    if multiview_paths != shift_paths or multiview_paths != localized_paths:
        raise RuntimeError("component extraction paths are not aligned")
    output_paths = multiview_paths
    if len(shift) != expected_shard_rows:
        raise RuntimeError("extracted row count does not match the locked manifest")
    uniform = bilateral_components(
        pooled[:, -FINAL_DIM:],
        prototypes,
        device=args.device,
        batch_size=int(lock["execution"]["score_batch_size"]),
    )["uniform_cosine"]
    localized = localized_prototype_confidence(
        localized_features, prototypes, args.device
    )
    knn = mean_neighbor_confidence(
        knn_index,
        meanview,
        k=int(lock["method"]["hyperparameters"]["knn_k"]),
        nprobe=int(lock["method"]["implementation"]["faiss_nprobe"]),
        batch_size=int(lock["execution"]["knn_batch_size"]),
    )
    rc = atomic.extract(pooled)["rc_msps"]
    pulse = pulse_state.score(shift, uniform, localized, knn)

    output_dir = Path(lock["outputs"]["score_shard_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{args.dataset}.rank{args.shard_index}.npz"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite final score shard: {output}")
    np.savez_compressed(
        output,
        labels=np.full(expected_shard_rows, -1, dtype=np.int16),
        pulse_ood_score=pulse.astype(np.float32),
        rc_msps_confidence=rc.astype(np.float32),
        shift_ratio=shift.astype(np.float32),
        uniform_cosine=uniform.astype(np.float32),
        localized_prototype_confidence=localized.astype(np.float32),
        compact_knn_confidence=knn.astype(np.float32),
    )
    output.with_suffix(".jsonl").write_text(
        "".join(json.dumps({"relative_path": path}) + "\n" for path in output_paths),
        encoding="utf-8",
    )
    output.with_suffix(".manifest.json").write_text(
        json.dumps(
            {
                "config_sha256": config_sha,
                "dataset": args.dataset,
                "final_access_authorized_by_pre_access_lock": True,
                "image_hashes_verified_before_decoding": True,
                "model_state_dict_sha256": observed_model_sha,
                "num_shards": num_shards,
                "output_sha256": sha256_file(output),
                "rows": expected_shard_rows,
                "runtime_seconds": time.time() - started,
                "same_locked_method_for_every_image": True,
                "shard_index": args.shard_index,
                "target_ood_used_for_fit_or_calibration": False,
                "test_image_sharing": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"saved={output} rows={expected_shard_rows}", flush=True)


if __name__ == "__main__":
    main()
