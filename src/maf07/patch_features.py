from __future__ import annotations

import json
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from .backbones import BackboneRunner, device_available, load_backbone_specs
from .config import load_yaml, resolve_path
from .data import load_manifest


@dataclass(frozen=True)
class ForegroundPatchCache:
    backbone: str
    data_path: Path
    metadata_path: Path
    info_path: Path


def foreground_patch_cache_paths(
    backbone: str,
    *,
    cache_dir: str | Path = "results/patch_features",
    foreground_ratio: float = 0.35,
    max_patches: int = 128,
) -> ForegroundPatchCache:
    root = resolve_path(cache_dir)
    ratio_tag = f"fg{int(round(float(foreground_ratio) * 100)):02d}"
    patch_tag = f"top{int(max_patches)}" if int(max_patches) > 0 else "topratio"
    stem = f"{backbone}.{ratio_tag}.{patch_tag}.fp16"
    return ForegroundPatchCache(
        backbone=backbone,
        data_path=root / f"{stem}.dat",
        metadata_path=root / f"{stem}.metadata.csv",
        info_path=root / f"{stem}.json",
    )


def _tokens_from_forward_features(out, num_prefix_tokens: int):
    if isinstance(out, dict):
        cls = out.get("x_norm_clstoken")
        patches = out.get("x_norm_patchtokens")
        if cls is not None and patches is not None:
            return cls, patches
        out = out.get("x_norm", out.get("tokens", out))
    if out.ndim != 3:
        raise ValueError(f"Expected ViT token tensor, got shape {tuple(out.shape)}")
    return out[:, 0], out[:, int(num_prefix_tokens) :]


def _l2_normalize_torch(x, eps: float = 1e-12):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


def _select_foreground_patches(cls, patches, foreground_ratio: float, max_patches: int):
    import torch

    cls = _l2_normalize_torch(cls)
    patches = _l2_normalize_torch(patches)
    scores = torch.einsum("bnd,bd->bn", patches, cls)
    ratio_k = max(1, int(scores.shape[1] * float(foreground_ratio)))
    k = min(scores.shape[1], ratio_k)
    if int(max_patches) > 0:
        k = min(k, int(max_patches))
    idx = torch.topk(scores, k=k, dim=1, largest=True).indices
    gather_idx = idx.unsqueeze(-1).expand(-1, -1, patches.shape[-1])
    return torch.gather(patches, dim=1, index=gather_idx)


def extract_foreground_patch_cache(
    backbone: str,
    *,
    manifest_path: str | Path = "results/manifest.csv",
    backbone_config: str | Path = "configs/backbones.yaml",
    experiments_config: str | Path = "configs/experiments.yaml",
    cache_dir: str | Path | None = None,
    foreground_ratio: float = 0.35,
    max_patches: int = 128,
    batch_size: int | None = None,
    device: str | None = None,
    resume: bool = True,
    worker_index: int = 0,
    worker_count: int = 1,
) -> ForegroundPatchCache:
    ecfg = load_yaml(experiments_config)
    runtime = ecfg.get("runtime", {})
    cache_dir = cache_dir or "results/patch_features"
    batch_size = int(batch_size or runtime.get("batch_size", 8))
    device = str(device or runtime.get("device", "cuda"))
    paths = foreground_patch_cache_paths(
        backbone,
        cache_dir=cache_dir,
        foreground_ratio=foreground_ratio,
        max_patches=max_patches,
    )
    worker_index = int(worker_index)
    worker_count = max(1, int(worker_count))
    done_path = paths.data_path.with_name(f"{paths.data_path.name}.worker{worker_index}of{worker_count}.done")
    if (
        resume
        and worker_count == 1
        and paths.data_path.exists()
        and paths.metadata_path.exists()
        and paths.info_path.exists()
    ):
        return paths
    if resume and worker_count > 1 and done_path.exists():
        return paths

    manifest = load_manifest(manifest_path)
    specs = load_backbone_specs(backbone_config)
    if backbone not in specs:
        raise KeyError(f"Unknown backbone: {backbone}")
    rows = manifest[["image_id", "class_name", "path", "rel_path"]].copy()
    image_paths = rows["path"].tolist()
    paths.data_path.parent.mkdir(parents=True, exist_ok=True)

    import torch

    if worker_count == 1 or worker_index == 0:
        runner = BackboneRunner(specs[backbone], device=device)
        if runner.spec.family != "timm":
            raise ValueError("Foreground patch extraction currently supports timm ViT backbones only")
        if runner.model is None or runner.preprocess is None:
            raise RuntimeError("Backbone is not loaded")
        dev = device if device_available(device) else "cpu"

        first_image = Image.open(image_paths[0]).convert("RGB")
        first_batch = torch.stack([runner.preprocess(first_image)]).to(dev)
        first_image.close()
        with torch.no_grad():
            first_out = runner.model.forward_features(first_batch)
            first_cls, first_patches = _tokens_from_forward_features(
                first_out,
                int(getattr(runner.model, "num_prefix_tokens", 1)),
            )
            first_fg = _select_foreground_patches(first_cls, first_patches, foreground_ratio, max_patches)
        n_images = len(rows)
        n_patches = int(first_fg.shape[1])
        dim = int(first_fg.shape[2])
        data = np.memmap(paths.data_path, mode="w+", dtype=np.float16, shape=(n_images, n_patches, dim))
        data.flush()
        rows.to_csv(paths.metadata_path, index=False)
        info = {
            "backbone": backbone,
            "dtype": "float16",
            "shape": [n_images, n_patches, dim],
            "foreground_ratio": float(foreground_ratio),
            "max_patches": int(max_patches),
        }
        paths.info_path.write_text(json.dumps(info, indent=2, sort_keys=True), encoding="utf-8")
        del data
        del runner
        if device_available(device):
            torch.cuda.empty_cache()
    else:
        wait_start = time.time()
        while not paths.info_path.exists():
            if time.time() - wait_start > 1800:
                raise TimeoutError(f"Timed out waiting for cache init: {paths.info_path}")
            time.sleep(2)

    info = json.loads(paths.info_path.read_text(encoding="utf-8"))
    n_images, n_patches, dim = (int(v) for v in info["shape"])
    data = np.memmap(paths.data_path, mode="r+", dtype=np.float16, shape=(n_images, n_patches, dim))
    runner = BackboneRunner(specs[backbone], device=device)
    if runner.spec.family != "timm":
        raise ValueError("Foreground patch extraction currently supports timm ViT backbones only")
    if runner.model is None or runner.preprocess is None:
        raise RuntimeError("Backbone is not loaded")
    dev = device if device_available(device) else "cpu"

    worker_positions = list(range(worker_index, len(image_paths), worker_count))
    with torch.no_grad():
        for pos in tqdm(
            range(0, len(worker_positions), batch_size),
            desc=f"patches:{backbone}:w{worker_index}/{worker_count}",
        ):
            batch_indices = worker_positions[pos : pos + batch_size]
            batch_paths = [image_paths[i] for i in batch_indices]
            images = [Image.open(path).convert("RGB") for path in batch_paths]
            batch = torch.stack([runner.preprocess(img) for img in images]).to(dev)
            for img in images:
                img.close()
            out = runner.model.forward_features(batch)
            cls, patches = _tokens_from_forward_features(out, int(getattr(runner.model, "num_prefix_tokens", 1)))
            fg = _select_foreground_patches(cls, patches, foreground_ratio, max_patches)
            data[np.asarray(batch_indices, dtype=int)] = fg.detach().cpu().to(torch.float16).numpy()
            if pos and pos % max(batch_size * 64, 1) == 0:
                data.flush()
    data.flush()
    if worker_count > 1:
        done_path.write_text("done\n", encoding="utf-8")
    load_foreground_patch_cache.cache_clear()
    return paths


@lru_cache(maxsize=None)
def load_foreground_patch_cache(
    backbone: str,
    *,
    cache_dir: str | Path = "results/patch_features",
    foreground_ratio: float = 0.35,
    max_patches: int = 128,
) -> tuple[pd.DataFrame, np.memmap, dict[str, object]]:
    paths = foreground_patch_cache_paths(
        backbone,
        cache_dir=cache_dir,
        foreground_ratio=foreground_ratio,
        max_patches=max_patches,
    )
    if not paths.data_path.exists() or not paths.metadata_path.exists() or not paths.info_path.exists():
        raise FileNotFoundError(f"Foreground patch cache missing for {backbone}: {paths.data_path}")
    info = json.loads(paths.info_path.read_text(encoding="utf-8"))
    shape = tuple(int(v) for v in info["shape"])
    meta = pd.read_csv(paths.metadata_path)
    if len(meta) != shape[0]:
        raise ValueError(f"Patch metadata length mismatch for {backbone}")
    data = np.memmap(paths.data_path, mode="r", dtype=np.float16, shape=shape)
    return meta, data, info


def patch_indices_for_split(
    split_df: pd.DataFrame,
    backbone: str,
    *,
    cache_dir: str | Path = "results/patch_features",
    foreground_ratio: float = 0.35,
    max_patches: int = 128,
) -> tuple[pd.DataFrame, np.ndarray, np.memmap, dict[str, object]]:
    meta, patches, info = load_foreground_patch_cache(
        backbone,
        cache_dir=cache_dir,
        foreground_ratio=foreground_ratio,
        max_patches=max_patches,
    )
    index = {image_id: i for i, image_id in enumerate(meta["image_id"].astype(str))}
    rows = split_df.copy()
    missing = [image_id for image_id in rows["image_id"].astype(str) if image_id not in index]
    if missing:
        raise KeyError(f"{len(missing)} split rows are missing from patch cache for {backbone}")
    indices = np.asarray([index[image_id] for image_id in rows["image_id"].astype(str)], dtype=int)
    return rows.reset_index(drop=True), indices, patches, info
