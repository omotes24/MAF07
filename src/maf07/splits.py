from __future__ import annotations

import hashlib
import itertools
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .config import load_yaml, resolve_path
from .data import load_manifest

SPLIT_ORDER = ("train", "val", "test")


def ratio_targets(n: int, ratios: Sequence[int]) -> dict[str, int]:
    total = float(sum(ratios))
    raw = np.asarray([n * r / total for r in ratios], dtype=float)
    base = np.floor(raw).astype(int)
    remainder = int(n - base.sum())
    order = np.argsort(-(raw - base))
    for idx in order[:remainder]:
        base[idx] += 1
    return dict(zip(SPLIT_ORDER, [int(x) for x in base], strict=True))


def _assign_class_groups(
    class_df: pd.DataFrame,
    seed: int,
    ratios: Sequence[int],
    group_col: str,
) -> dict[str, str]:
    group_sizes = class_df.groupby(group_col, sort=True).size().to_dict()
    groups = np.array(sorted(group_sizes))
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)

    targets = ratio_targets(len(class_df), ratios)
    counts = {split: 0 for split in SPLIT_ORDER}
    assignments: dict[str, str] = {}

    for group in groups:
        size = int(group_sizes[group])

        def split_key(split: str) -> tuple[float, int]:
            target = max(targets[split], 1)
            deficit = (targets[split] - counts[split]) / target
            overshoot = max(0, counts[split] + size - targets[split])
            return (deficit - overshoot, -counts[split])

        split = max(SPLIT_ORDER, key=split_key)
        assignments[str(group)] = split
        counts[split] += size
    return assignments


def assign_splits(
    manifest: pd.DataFrame,
    seed: int,
    ratios: Sequence[int] = (4, 1, 1),
    group_col: str = "duplicate_group",
) -> pd.DataFrame:
    required = {"image_id", "class_name", "path", group_col}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest missing required columns: {sorted(missing)}")

    frames: list[pd.DataFrame] = []
    for class_name, class_df in manifest.groupby("class_name", sort=True):
        class_df = class_df.copy()
        class_offset = int(hashlib.sha1(class_name.encode("utf-8")).hexdigest()[:8], 16)
        assignments = _assign_class_groups(
            class_df,
            seed=seed + class_offset,
            ratios=ratios,
            group_col=group_col,
        )
        class_df["split"] = class_df[group_col].astype(str).map(assignments)
        frames.append(class_df)
    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["class_name", "split", "image_id"]).reset_index(drop=True)
    assert_no_duplicate_group_cross_split(out, group_col=group_col)
    return out


def assert_no_duplicate_group_cross_split(split_df: pd.DataFrame, group_col: str = "duplicate_group") -> None:
    crossings = split_df.groupby(group_col)["split"].nunique()
    bad = crossings[crossings > 1]
    if not bad.empty:
        examples = ", ".join(map(str, bad.index[:5]))
        raise AssertionError(f"Duplicate groups cross splits: {examples}")


def make_splits(
    dataset_config: str | Path = "configs/dataset.yaml",
    manifest_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> list[Path]:
    cfg = load_yaml(dataset_config)
    manifest = load_manifest(manifest_path or cfg.get("manifest_path", "results/manifest.csv"))
    split_cfg = cfg.get("split", {})
    ratios = (int(split_cfg.get("train", 4)), int(split_cfg.get("val", 1)), int(split_cfg.get("test", 1)))
    seeds = [int(s) for s in split_cfg.get("seeds", [0, 1, 2])]
    out_dir = resolve_path(output_dir or cfg.get("split_dir", "results/splits"))
    out_dir.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []
    for seed in seeds:
        split_df = assign_splits(manifest, seed=seed, ratios=ratios)
        path = out_dir / f"splits_seed{seed}.csv"
        split_df.to_csv(path, index=False)
        paths.append(path)
    return paths


def make_ood_eval_frame(
    split_df: pd.DataFrame,
    id_classes: Iterable[str],
    *,
    protocol: str = "fair",
) -> pd.DataFrame:
    id_set = set(id_classes)
    if protocol not in {"fair", "oracle"}:
        raise ValueError(f"Unknown protocol: {protocol}")
    rows: list[pd.DataFrame] = []
    id_df = split_df[split_df["class_name"].isin(id_set)].copy()
    id_df["ood_label"] = 0
    id_df["role"] = "id_" + id_df["split"].astype(str)
    rows.append(id_df)

    ood_df = split_df[(~split_df["class_name"].isin(id_set)) & (split_df["split"] == "test")].copy()
    ood_df["ood_label"] = 1
    ood_df["role"] = "ood_test"
    rows.append(ood_df)
    out = pd.concat(rows, ignore_index=True)
    audit_no_ood_val(out, protocol=protocol)
    return out


def audit_no_ood_val(eval_df: pd.DataFrame, *, protocol: str = "fair") -> None:
    if protocol != "fair":
        return
    bad = eval_df[(eval_df["ood_label"] == 1) & (eval_df["role"].isin(["ood_train", "ood_val"]))]
    if not bad.empty:
        raise AssertionError("Fair protocol loaded OOD train/val rows")
    bad_split = eval_df[(eval_df["ood_label"] == 1) & (eval_df["split"] != "test")]
    if not bad_split.empty:
        raise AssertionError("Fair protocol loaded non-test OOD rows")


def all_id_sets(classes: Sequence[str], id_sizes: Iterable[int]) -> list[tuple[str, ...]]:
    combos: list[tuple[str, ...]] = []
    for k in id_sizes:
        combos.extend(tuple(c) for c in itertools.combinations(classes, int(k)))
    return combos
