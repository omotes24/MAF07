#!/usr/bin/env python3
"""Hash a preregistered final suite without decoding or scoring any image."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_preregistration(path: Path) -> tuple[dict[str, object], str]:
    sidecar = path.with_suffix(".sha256")
    expected = sidecar.read_text(encoding="utf-8").split()[0]
    observed = sha256_file(path)
    if observed != expected:
        raise RuntimeError("final benchmark preregistration hash mismatch")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["final_evaluation_executed"]:
        raise RuntimeError("final benchmark was already evaluated")
    return payload, observed


def parse_imglist(path: Path) -> list[str]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        relative, label = line.rsplit(maxsplit=1)
        if label != "-1":
            raise ValueError(f"unexpected OOD label in {path}: {label}")
        rows.append(relative)
    if len(rows) != len(set(rows)):
        raise ValueError(f"duplicate path in {path}")
    return rows


def materialize_manifest(
    data_root: Path,
    preregistration: Path,
    output_dir: Path,
) -> dict[str, object]:
    prereg, prereg_sha = verify_preregistration(preregistration)
    summary_path = output_dir / "final_dataset_manifest.json"
    rows_path = output_dir / "final_dataset_manifest.jsonl"
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite final manifest: {output_dir}")
    output_dir.mkdir(parents=True)
    image_root = data_root / "images_largescale"
    aggregate = hashlib.sha256()
    dataset_summaries = []
    total_images = 0
    total_bytes = 0

    with rows_path.open("x", encoding="utf-8") as stream:
        for spec in prereg["datasets"]:
            alias = str(spec["openood_alias"])
            imglist = data_root / "benchmark_imglist" / "imagenet" / f"test_{alias}.txt"
            listed = parse_imglist(imglist)
            expected = int(spec["expected_images"])
            if len(listed) != expected:
                raise RuntimeError(f"{alias}: imglist has {len(listed)} rows, expected {expected}")
            disk_paths = {
                str(path.relative_to(image_root))
                for path in (image_root / alias).rglob("*")
                if path.is_file()
            }
            if set(listed) != disk_paths:
                missing = sorted(set(listed) - disk_paths)[:5]
                extra = sorted(disk_paths - set(listed))[:5]
                raise RuntimeError(f"{alias}: imglist/files mismatch missing={missing} extra={extra}")

            dataset_digest = hashlib.sha256()
            dataset_bytes = 0
            for relative in sorted(listed):
                path = image_root / relative
                size = path.stat().st_size
                digest = sha256_file(path)
                record = {
                    "bytes": size,
                    "dataset": str(spec["name"]),
                    "relative_path": relative,
                    "sha256": digest,
                }
                encoded = json.dumps(record, sort_keys=True, separators=(",", ":"))
                stream.write(encoded + "\n")
                dataset_digest.update((encoded + "\n").encode("utf-8"))
                aggregate.update((encoded + "\n").encode("utf-8"))
                dataset_bytes += size
            dataset_summaries.append(
                {
                    "bytes": dataset_bytes,
                    "file_manifest_sha256": dataset_digest.hexdigest(),
                    "imglist_path": str(imglist),
                    "imglist_sha256": sha256_file(imglist),
                    "name": str(spec["name"]),
                    "num_images": len(listed),
                    "openood_alias": alias,
                }
            )
            total_images += len(listed)
            total_bytes += dataset_bytes

    summary = {
        "aggregate_file_manifest_sha256": aggregate.hexdigest(),
        "data_root": str(data_root.resolve()),
        "datasets": dataset_summaries,
        "feature_extraction_executed": False,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "image_decoding_executed": False,
        "preregistration_path": str(preregistration.resolve()),
        "preregistration_sha256": prereg_sha,
        "score_computation_executed": False,
        "total_bytes": total_bytes,
        "total_images": total_images,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "final_dataset_manifest.sha256").write_text(
        f"{sha256_file(summary_path)}  {summary_path.name}\n"
        f"{sha256_file(rows_path)}  {rows_path.name}\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            materialize_manifest(args.data_root, args.preregistration, args.output_dir),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
