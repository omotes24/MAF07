"""Feature-cache loading, deterministic ID splits, and provenance helpers."""

from __future__ import annotations

import glob
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


STAGE_DIMS = (256, 512, 1024, 2048)
FORBIDDEN_SEARCH_BASENAME_TOKENS = (
    "imagenet_o",
    "imagenet-o",
    "inaturalist",
    "openimage",
    "texture",
)


@dataclass(frozen=True)
class IdSplit:
    prototype: np.ndarray
    calibration: np.ndarray
    proxy_query: np.ndarray


def expand_paths(patterns: Sequence[str | Path]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = sorted(glob.glob(str(pattern)))
        paths.extend(Path(match) for match in matches)
    unique = sorted(set(path.resolve() for path in paths))
    if not unique:
        raise FileNotFoundError(f"no files matched: {list(map(str, patterns))}")
    return unique


def assert_id_only_paths(paths: Iterable[Path]) -> None:
    """Reject OOD-named cache files from every search/fitting entry point."""
    rejected = []
    for path in paths:
        name = path.name.lower()
        if any(token in name for token in FORBIDDEN_SEARCH_BASENAME_TOKENS):
            rejected.append(str(path))
    if rejected:
        raise PermissionError(f"OOD cache access is forbidden during ID-only fitting: {rejected}")


def load_arrays(paths: Sequence[Path], keys: Sequence[str]) -> dict[str, np.ndarray]:
    if not paths:
        raise ValueError("at least one cache shard is required")
    chunks: dict[str, list[np.ndarray]] = {key: [] for key in keys}
    for path in sorted(paths):
        with np.load(path, allow_pickle=False) as shard:
            missing = [key for key in keys if key not in shard]
            if missing:
                raise KeyError(f"{path} is missing arrays {missing}")
            for key in keys:
                chunks[key].append(shard[key])
    result = {key: np.concatenate(values, axis=0) for key, values in chunks.items()}
    for key, value in result.items():
        if not np.isfinite(value).all():
            raise ValueError(f"non-finite values found in cache array {key}")
    return result


def load_jsonl_paths(cache_paths: Sequence[Path]) -> np.ndarray:
    rows: list[str] = []
    for cache_path in sorted(cache_paths):
        jsonl = cache_path.with_suffix(".jsonl")
        if not jsonl.is_file():
            raise FileNotFoundError(jsonl)
        for raw in jsonl.read_text(encoding="utf-8").splitlines():
            rows.append(str(json.loads(raw)["path"]))
    return np.asarray(rows)


def split_stages(features: np.ndarray) -> tuple[np.ndarray, ...]:
    if features.ndim != 2 or features.shape[1] != sum(STAGE_DIMS):
        raise ValueError(f"expected [N, {sum(STAGE_DIMS)}] pooled features, got {features.shape}")
    boundaries = np.cumsum(STAGE_DIMS)[:-1]
    return tuple(np.split(features, boundaries, axis=1))


def deterministic_id_split(
    labels: np.ndarray,
    paths: np.ndarray,
    *,
    num_classes: int = 1000,
    prototype_per_class: int = 150,
    calibration_per_class: int = 25,
    proxy_per_class: int = 25,
) -> IdSplit:
    labels = np.asarray(labels, dtype=np.int64)
    paths = np.asarray(paths).astype(str)
    if labels.shape != paths.shape:
        raise ValueError("labels and paths must have equal one-dimensional shape")
    required = prototype_per_class + calibration_per_class + proxy_per_class
    prototype, calibration, proxy = [], [], []
    for class_id in range(num_classes):
        indices = np.flatnonzero(labels == class_id)
        if len(indices) != required:
            raise ValueError(
                f"class {class_id} has {len(indices)} rows; deterministic protocol requires {required}"
            )
        ordered = indices[np.argsort(paths[indices])]
        prototype.extend(ordered[:prototype_per_class])
        calibration.extend(ordered[prototype_per_class : prototype_per_class + calibration_per_class])
        proxy.extend(ordered[prototype_per_class + calibration_per_class :])
    return IdSplit(
        prototype=np.asarray(prototype, dtype=np.int64),
        calibration=np.asarray(calibration, dtype=np.int64),
        proxy_query=np.asarray(proxy, dtype=np.int64),
    )


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def dataset_manifest(paths: Sequence[str]) -> dict[str, object]:
    rows = []
    missing = 0
    for raw_path in sorted(paths):
        path = Path(raw_path)
        if path.is_file():
            rows.append((raw_path, path.stat().st_size))
        else:
            rows.append((raw_path, None))
            missing += 1
    return {
        "n": len(rows),
        "missing": missing,
        "path_size_sha256": sha256_json(rows),
    }
