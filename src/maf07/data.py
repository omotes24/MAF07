from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

from .config import ensure_parent, load_yaml, resolve_path


@dataclass(frozen=True)
class ImageRecord:
    image_id: str
    class_name: str
    path: str
    rel_path: str
    sha256: str
    width: int
    height: int
    corrupt: bool
    duplicate_group: str


def iter_image_files(root: Path, classes: Iterable[str], extensions: Iterable[str]) -> Iterable[tuple[str, Path]]:
    suffixes = {ext.lower() for ext in extensions}
    for class_name in classes:
        class_dir = root / class_name
        if not class_dir.exists():
            continue
        for path in sorted(class_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in suffixes:
                yield class_name, path


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def inspect_image(path: Path) -> tuple[int, int, bool]:
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            width, height = img.size
        return int(width), int(height), False
    except (OSError, UnidentifiedImageError):
        return 0, 0, True


def build_manifest(
    dataset_config: str | Path = "configs/dataset.yaml",
    output_path: str | Path | None = None,
    *,
    verify_pixels: bool = True,
) -> pd.DataFrame:
    cfg = load_yaml(dataset_config)
    root = resolve_path(cfg["dataset_root"])
    classes = list(cfg["classes"])
    extensions = cfg.get("image_extensions", [".jpg", ".jpeg", ".png", ".webp"])
    output = ensure_parent(output_path or cfg.get("manifest_path", "results/manifest.csv"))

    records: list[ImageRecord] = []
    files = list(iter_image_files(root, classes, extensions))
    for class_name, path in tqdm(files, desc="manifest", unit="image"):
        rel_path = path.relative_to(root).as_posix()
        file_hash = sha256_file(path)
        width, height, corrupt = inspect_image(path) if verify_pixels else (0, 0, False)
        image_id = hashlib.sha1(f"{class_name}/{rel_path}/{file_hash}".encode("utf-8")).hexdigest()
        duplicate_group = f"sha256:{file_hash}" if file_hash else f"path:{rel_path}"
        records.append(
            ImageRecord(
                image_id=image_id,
                class_name=class_name,
                path=str(path),
                rel_path=rel_path,
                sha256=file_hash,
                width=width,
                height=height,
                corrupt=corrupt,
                duplicate_group=duplicate_group,
            )
        )
    df = pd.DataFrame([r.__dict__ for r in records])
    if not df.empty:
        df = df.sort_values(["class_name", "rel_path"]).reset_index(drop=True)
    df.to_csv(output, index=False)
    return df


def load_manifest(path: str | Path = "results/manifest.csv") -> pd.DataFrame:
    manifest_path = resolve_path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest does not exist: {manifest_path}")
    return pd.read_csv(manifest_path)


def audit_manifest(
    manifest: pd.DataFrame,
    classes: Iterable[str],
    target_per_class: int | None = None,
) -> dict[str, object]:
    if manifest.empty:
        return {
            "ok": False,
            "reason": "manifest is empty",
            "class_counts": {},
            "corrupt_count": 0,
            "duplicate_exact_count": 0,
        }
    counts = manifest.groupby("class_name").size().reindex(list(classes), fill_value=0).to_dict()
    corrupt_count = int(manifest.get("corrupt", pd.Series(dtype=bool)).fillna(False).sum())
    duplicate_exact_count = int(manifest.duplicated("sha256", keep=False).sum())
    enough = True
    if target_per_class is not None:
        enough = all(int(counts.get(c, 0)) >= target_per_class for c in classes)
    ok = corrupt_count == 0 and enough
    reason = "ok" if ok else "corrupt images or insufficient class counts"
    return {
        "ok": ok,
        "reason": reason,
        "class_counts": {k: int(v) for k, v in counts.items()},
        "corrupt_count": corrupt_count,
        "duplicate_exact_count": duplicate_exact_count,
    }


def verify_dataset(dataset_config: str | Path = "configs/dataset.yaml") -> dict[str, object]:
    cfg = load_yaml(dataset_config)
    manifest_path = cfg.get("manifest_path", "results/manifest.csv")
    if resolve_path(manifest_path).exists():
        manifest = load_manifest(manifest_path)
    else:
        manifest = build_manifest(dataset_config)
    return audit_manifest(manifest, cfg["classes"], int(cfg.get("target_per_class", 0)) or None)

