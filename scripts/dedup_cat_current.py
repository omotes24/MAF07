#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class DSU:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


class BKNode:
    def __init__(self, value: int, index: int) -> None:
        self.value = value
        self.indices = [index]
        self.children: dict[int, BKNode] = {}

    def add(self, value: int, index: int) -> None:
        distance = (self.value ^ value).bit_count()
        if distance == 0:
            self.indices.append(index)
        elif distance in self.children:
            self.children[distance].add(value, index)
        else:
            self.children[distance] = BKNode(value, index)

    def query(self, value: int, radius: int, out: list[int]) -> None:
        distance = (self.value ^ value).bit_count()
        if distance <= radius:
            out.extend(self.indices)
        for key in range(max(1, distance - radius), distance + radius + 1):
            child = self.children.get(key)
            if child is not None:
                child.query(value, radius, out)


def load_config(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def manifest_inputs(metadata_dir: Path) -> list[Path]:
    excluded = {
        "manifest_all_candidates.csv",
        "manifest_clip_filtered.csv",
        "final_manifest.csv",
        "dedup_unique_manifest.csv",
        "dedup_rejected_duplicates.csv",
    }
    chosen: dict[str, Path] = {}
    for path in sorted(metadata_dir.glob("manifest_*.csv")):
        if path.name in excluded or path.name.startswith("manifest_final"):
            continue
        key = path.name.replace(".partial.csv", ".csv")
        if key not in chosen or ".partial." in chosen[key].name:
            chosen[key] = path
    return sorted(chosen.values())


def load_candidates(root: Path, cfg: dict[str, Any]) -> pd.DataFrame:
    metadata_dir = root / "metadata"
    frames = []
    for path in manifest_inputs(metadata_dir):
        try:
            frame = pd.read_csv(path, low_memory=False)
        except pd.errors.EmptyDataError:
            continue
        if "target_class" not in frame.columns and "class_name" in frame.columns:
            frame["target_class"] = frame["class_name"]
        frame["manifest_source"] = path.name
        frames.append(frame)
    if not frames:
        raise RuntimeError(f"No candidate manifests found in {metadata_dir}")
    data = pd.concat(frames, ignore_index=True, sort=False)
    classes = cfg.get(
        "classes",
        ["cheetah", "jaguar", "leopard", "lion", "ocelot", "puma", "serval", "tiger"],
    )
    data = data[data["target_class"].isin(classes)].copy()
    if "status" in data.columns:
        data = data[data["status"].isin(["ok", "exists"])].copy()
    required = ["pixel_sha256", "phash", "dhash", "local_path"]
    for col in required:
        data = data[data[col].notna()]
        data = data[data[col].astype(str).str.len() > 0]
    data = data.drop_duplicates(subset=["target_class", "local_path"]).reset_index(drop=True)
    priority = cfg.get("source_priority", {})
    data["source_priority"] = data["source"].map(priority).fillna(99).astype(int)
    if "clip_target_probability" in data.columns:
        clip_prob = pd.to_numeric(data["clip_target_probability"], errors="coerce")
    else:
        clip_prob = pd.Series(np.nan, index=data.index)
    data["clip_target_probability"] = clip_prob.fillna(1.0)
    return data.sort_values(
        ["source_priority", "clip_target_probability"],
        ascending=[True, False],
    ).reset_index(drop=True)


def dedup(data: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    dsu = DSU(len(data))
    for _, indices in data.groupby("pixel_sha256").groups.items():
        idx = list(indices)
        for other in idx[1:]:
            dsu.union(idx[0], other)

    ph_values = [int(str(value), 16) for value in data["phash"]]
    dh_values = [int(str(value), 16) for value in data["dhash"]]
    dedup_cfg = cfg.get("dedup", {})
    p_radius = int(dedup_cfg.get("phash_threshold", 4))
    d_radius = int(dedup_cfg.get("dhash_threshold", 6))
    root_node: BKNode | None = None
    for index, value in enumerate(ph_values):
        if root_node is None:
            root_node = BKNode(value, index)
            continue
        matches: list[int] = []
        root_node.query(value, p_radius, matches)
        for other in matches:
            if (dh_values[index] ^ dh_values[other]).bit_count() <= d_radius:
                dsu.union(index, other)
        root_node.add(value, index)
        if index and index % 10000 == 0:
            print(f"dedup {index}/{len(data)}", flush=True)

    components: dict[int, list[int]] = defaultdict(list)
    for index in range(len(data)):
        components[dsu.find(index)].append(index)

    keep: list[int] = []
    rejected_rows = []
    component_id_by_index = {}
    for component_id, indices in enumerate(components.values()):
        species = sorted(set(data.loc[indices, "target_class"].astype(str)))
        if len(species) > 1:
            for idx in indices:
                row = data.loc[idx].to_dict()
                row["duplicate_component"] = component_id
                row["dedup_decision"] = "reject"
                row["dedup_reason"] = "cross_class_duplicate"
                rejected_rows.append(row)
            continue
        chosen = indices[0]
        keep.append(chosen)
        component_id_by_index[chosen] = component_id
        for idx in indices[1:]:
            row = data.loc[idx].to_dict()
            row["duplicate_component"] = component_id
            row["dedup_decision"] = "reject"
            row["dedup_reason"] = "same_class_duplicate"
            row["kept_local_path"] = data.at[chosen, "local_path"]
            rejected_rows.append(row)

    unique = data.loc[keep].copy()
    unique["duplicate_component"] = [component_id_by_index[i] for i in keep]
    unique["dedup_decision"] = "keep"
    rejected = pd.DataFrame(rejected_rows)
    report = {
        "input_candidates": int(len(data)),
        "unique_kept": int(len(unique)),
        "rejected_duplicates": int(len(rejected)),
        "cross_class_rejected": int((rejected.get("dedup_reason", pd.Series(dtype=str)) == "cross_class_duplicate").sum()),
        "same_class_rejected": int((rejected.get("dedup_reason", pd.Series(dtype=str)) == "same_class_duplicate").sum()),
        "phash_threshold": p_radius,
        "dhash_threshold": d_radius,
    }
    return unique, rejected, report


def write_outputs(root: Path, unique: pd.DataFrame, rejected: pd.DataFrame, report: dict[str, Any]) -> None:
    metadata = root / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    unique.to_csv(metadata / "dedup_unique_manifest.csv", index=False)
    rejected.to_csv(metadata / "dedup_rejected_duplicates.csv", index=False)
    counts = unique.groupby("target_class").size().sort_index().reset_index(name="unique_count")
    counts.to_csv(metadata / "dedup_unique_counts.csv", index=False)
    report = dict(report)
    report["counts"] = {row.target_class: int(row.unique_count) for row in counts.itertuples(index=False)}
    (metadata / "dedup_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))


def quarantine(root: Path, rejected: pd.DataFrame, name: str) -> None:
    if rejected.empty:
        return
    qroot = root / "duplicates_quarantine" / name
    moved = []
    for row in rejected.itertuples(index=False):
        source = Path(str(getattr(row, "local_path")))
        if not source.exists() or not source.is_file():
            continue
        try:
            rel = source.relative_to(root)
        except ValueError:
            rel = Path(source.name)
        dest = qroot / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            source.unlink()
        else:
            shutil.move(str(source), str(dest))
        moved.append({"old_path": str(source), "quarantine_path": str(dest)})
    pd.DataFrame(moved).to_csv(root / "metadata" / f"dedup_quarantine_{name}.csv", index=False)
    print(f"quarantined {len(moved)} files to {qroot}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/home/omote/CAT")
    parser.add_argument("--config", default="/home/omote/MAF-OOD-v51/configs/cats15k_replacements.json")
    parser.add_argument("--quarantine-name", default="")
    args = parser.parse_args()

    root = Path(args.root)
    cfg = load_config(Path(args.config))
    data = load_candidates(root, cfg)
    unique, rejected, report = dedup(data, cfg)
    write_outputs(root, unique, rejected, report)
    if args.quarantine_name:
        quarantine(root, rejected, args.quarantine_name)


if __name__ == "__main__":
    main()
