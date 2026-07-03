#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import pandas as pd


CLASSES = ["cheetah", "jaguar", "leopard", "lion", "ocelot", "puma", "serval", "tiger"]


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def materialize(root: Path, manifest_path: Path, clear: bool = True) -> dict[str, object]:
    manifest = pd.read_csv(manifest_path, low_memory=False)
    manifest = manifest[manifest["target_class"].isin(CLASSES)].copy()
    manifest = manifest.sort_values(["target_class", "duplicate_component", "local_path"]).reset_index(drop=True)

    if clear:
        for class_name in CLASSES:
            shutil.rmtree(root / class_name, ignore_errors=True)

    selected = []
    missing: list[str] = []
    for class_name in CLASSES:
        subset = manifest[manifest["target_class"] == class_name].copy()
        for number, (_, row) in enumerate(subset.iterrows(), 1):
            source = Path(str(row["local_path"]))
            if not source.exists():
                missing.append(str(source))
                continue
            destination = root / class_name / f"{class_name}_{number:05d}.jpg"
            link_or_copy(source, destination)
            out = row.to_dict()
            out["final_path"] = str(destination)
            selected.append(out)

    if missing:
        raise RuntimeError(f"{len(missing)} source images are missing; first={missing[0]}")

    selected_df = pd.DataFrame(selected)
    metadata = root / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    selected_df.to_csv(metadata / "final_manifest.csv", index=False)
    counts = selected_df.groupby("target_class").size().reindex(CLASSES, fill_value=0).astype(int)
    counts.reset_index(name="count").rename(columns={"index": "target_class"}).to_csv(
        metadata / "final_counts.csv", index=False
    )

    hash_rows = []
    for row in selected_df.itertuples(index=False):
        final_path = Path(str(getattr(row, "final_path")))
        hash_rows.append(
            f"{getattr(row, 'target_class')},{final_path.name},{sha256_file(final_path)}"
        )
    data_hash = hashlib.sha256("\n".join(sorted(hash_rows)).encode("utf-8")).hexdigest()
    report = {
        "classes": CLASSES,
        "counts": {k: int(v) for k, v in counts.to_dict().items()},
        "data_hash": data_hash,
        "final_manifest": str(metadata / "final_manifest.csv"),
        "source_manifest": str(manifest_path),
        "total": int(len(selected_df)),
    }
    (metadata / "data_version.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/home/omote/CAT")
    parser.add_argument("--manifest", default="/home/omote/CAT/metadata/dedup_unique_manifest.csv")
    parser.add_argument("--no-clear", action="store_true")
    args = parser.parse_args()
    materialize(Path(args.root), Path(args.manifest), clear=not args.no_clear)


if __name__ == "__main__":
    main()

