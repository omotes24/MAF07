from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from .backbones import BackboneRunner, load_backbone_specs
from .config import load_yaml, resolve_path
from .data import load_manifest


@dataclass(frozen=True)
class FeatureCache:
    backbone: str
    feature_path: Path
    metadata_path: Path


def cache_paths(backbone: str, cache_dir: str | Path = "results/features") -> FeatureCache:
    root = resolve_path(cache_dir)
    return FeatureCache(
        backbone=backbone,
        feature_path=root / f"{backbone}.npz",
        metadata_path=root / f"{backbone}.metadata.csv",
    )


@lru_cache(maxsize=None)
def load_feature_cache(backbone: str, cache_dir: str | Path = "results/features") -> tuple[pd.DataFrame, np.ndarray]:
    paths = cache_paths(backbone, cache_dir)
    if not paths.feature_path.exists() or not paths.metadata_path.exists():
        raise FileNotFoundError(f"Feature cache missing for {backbone}: {paths.feature_path}")
    meta = pd.read_csv(paths.metadata_path)
    data = np.load(paths.feature_path, allow_pickle=False)
    features = data["features"]
    if len(meta) != features.shape[0]:
        raise ValueError(f"Feature metadata length mismatch for {backbone}")
    return meta, features


def write_feature_cache(
    backbone: str,
    metadata: pd.DataFrame,
    features: np.ndarray,
    cache_dir: str | Path = "results/features",
) -> FeatureCache:
    paths = cache_paths(backbone, cache_dir)
    paths.feature_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(paths.feature_path, features=np.asarray(features, dtype=np.float32))
    metadata.to_csv(paths.metadata_path, index=False)
    return paths


def extract_feature_cache(
    backbone: str,
    *,
    manifest_path: str | Path = "results/manifest.csv",
    backbone_config: str | Path = "configs/backbones.yaml",
    experiments_config: str | Path = "configs/experiments.yaml",
    cache_dir: str | Path | None = None,
    batch_size: int | None = None,
    device: str | None = None,
    resume: bool = True,
) -> FeatureCache:
    ecfg = load_yaml(experiments_config)
    runtime = ecfg.get("runtime", {})
    cache_dir = cache_dir or ecfg.get("feature_cache_dir", "results/features")
    batch_size = int(batch_size or runtime.get("batch_size", 64))
    device = str(device or runtime.get("device", "cuda"))
    paths = cache_paths(backbone, cache_dir)
    if resume and paths.feature_path.exists() and paths.metadata_path.exists():
        return paths

    manifest = load_manifest(manifest_path)
    specs = load_backbone_specs(backbone_config)
    if backbone not in specs:
        raise KeyError(f"Unknown backbone: {backbone}")
    runner = BackboneRunner(specs[backbone], device=device)

    feats: list[np.ndarray] = []
    rows = manifest[["image_id", "class_name", "path", "rel_path"]].copy()
    image_paths = rows["path"].tolist()
    for start in tqdm(range(0, len(image_paths), batch_size), desc=f"features:{backbone}"):
        batch_paths = image_paths[start : start + batch_size]
        images = [Image.open(path).convert("RGB") for path in batch_paths]
        feats.append(runner.encode_pil(images, batch_size=batch_size))
        for img in images:
            img.close()
    features = np.concatenate(feats, axis=0) if feats else np.empty((0, 0), dtype=np.float32)
    return write_feature_cache(backbone, rows, features, cache_dir)


def feature_frame_for_split(
    split_df: pd.DataFrame,
    backbone: str,
    cache_dir: str | Path = "results/features",
) -> tuple[pd.DataFrame, np.ndarray]:
    meta, features = load_feature_cache(backbone, cache_dir)
    index = {image_id: i for i, image_id in enumerate(meta["image_id"].astype(str))}
    rows = split_df.copy()
    missing = [image_id for image_id in rows["image_id"].astype(str) if image_id not in index]
    if missing:
        raise KeyError(f"{len(missing)} split rows are missing from feature cache for {backbone}")
    indices = np.asarray([index[image_id] for image_id in rows["image_id"].astype(str)], dtype=int)
    return rows.reset_index(drop=True), features[indices]
