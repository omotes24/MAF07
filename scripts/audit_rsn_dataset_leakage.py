#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from maf07.config import resolve_path
from maf07.features import load_feature_cache


PROVENANCE_FIELDS = {
    "source": ("source", "source_url", "url", "domain", "collection_source"),
    "license": ("license", "license_name", "license_url"),
    "photographer": ("photographer", "author", "creator", "owner"),
    "individual": ("individual_id", "animal_id", "identity_id"),
    "site": ("site_id", "location_id", "camera_id"),
}


def _metadata_coverage(manifest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for concept, candidates in PROVENANCE_FIELDS.items():
        present = [column for column in candidates if column in manifest]
        if present:
            nonempty = manifest[present].notna().any(axis=1)
            for column in present:
                values = manifest[column].astype(str).str.strip()
                nonempty |= ~values.isin(["", "nan", "none", "None"])
            coverage = float(nonempty.mean())
        else:
            coverage = 0.0
        rows.append(
            {
                "concept": concept,
                "available_columns": "|".join(present),
                "coverage": coverage,
                "status": (
                    "available_complete"
                    if present and coverage >= 1.0
                    else "available_partial"
                    if present and coverage > 0.0
                    else "unverified_missing_metadata"
                ),
            }
        )
    return pd.DataFrame(rows)


def _exact_duplicates(manifest: pd.DataFrame) -> pd.DataFrame:
    hash_column = next(
        (column for column in ("sha256", "pixel_sha256", "raw_sha256") if column in manifest),
        None,
    )
    if hash_column is None:
        return pd.DataFrame(
            columns=["hash", "n", "class_count", "cross_class", "image_ids", "classes"]
        )
    valid = manifest[manifest[hash_column].notna()].copy()
    valid[hash_column] = valid[hash_column].astype(str)
    valid = valid[~valid[hash_column].isin(["", "nan", "None"])]
    rows = []
    for value, block in valid.groupby(hash_column):
        if len(block) < 2:
            continue
        classes = sorted(block["class_name"].astype(str).unique())
        rows.append(
            {
                "hash": value,
                "n": int(len(block)),
                "class_count": int(len(classes)),
                "cross_class": bool(len(classes) > 1),
                "image_ids": "|".join(block["image_id"].astype(str)),
                "classes": "|".join(classes),
            }
        )
    return pd.DataFrame(
        rows,
        columns=["hash", "n", "class_count", "cross_class", "image_ids", "classes"],
    )


def _feature_pairs(
    backbone: str,
    *,
    threshold: float,
    neighbors: int,
    ef_search: int,
) -> pd.DataFrame:
    meta, features = load_feature_cache(backbone)
    values = np.asarray(features, dtype=np.float32)
    values /= np.linalg.norm(values, axis=1, keepdims=True) + 1e-12
    try:
        import faiss

        index = faiss.IndexHNSWFlat(
            values.shape[1], 32, faiss.METRIC_INNER_PRODUCT
        )
        index.hnsw.efConstruction = 80
        index.hnsw.efSearch = int(ef_search)
        index.add(values)
        similarity, indices = index.search(values, max(2, int(neighbors) + 1))
        engine = "faiss_hnsw_ip"
    except ImportError:
        if len(values) > 20_000:
            raise RuntimeError(
                "FAISS is required for the full near-duplicate audit; install "
                "the project with the 'audit' extra."
            )
        from sklearn.neighbors import NearestNeighbors

        model = NearestNeighbors(
            n_neighbors=max(2, int(neighbors) + 1), metric="cosine", n_jobs=-1
        ).fit(values)
        distances, indices = model.kneighbors(values)
        similarity = 1.0 - distances
        engine = "sklearn_cosine_exact"

    meta = meta.reset_index(drop=True)
    rows = []
    seen = set()
    for left in range(len(values)):
        for similarity_value, right in zip(similarity[left], indices[left], strict=True):
            right = int(right)
            if right == left or float(similarity_value) < threshold:
                continue
            key = tuple(sorted((left, right)))
            if key in seen:
                continue
            seen.add(key)
            a = meta.iloc[key[0]]
            b = meta.iloc[key[1]]
            rows.append(
                {
                    "backbone": backbone,
                    "engine": engine,
                    "similarity": float(similarity_value),
                    "left_image_id": str(a["image_id"]),
                    "right_image_id": str(b["image_id"]),
                    "left_class": str(a["class_name"]),
                    "right_class": str(b["class_name"]),
                    "cross_class": bool(str(a["class_name"]) != str(b["class_name"])),
                    "left_path": str(a.get("path", "")),
                    "right_path": str(b.get("path", "")),
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "backbone",
            "engine",
            "similarity",
            "left_image_id",
            "right_image_id",
            "left_class",
            "right_class",
            "cross_class",
            "left_path",
            "right_path",
        ],
    )


def _split_crossings(
    pairs: pd.DataFrame,
    split_dir: Path,
    seeds: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    details = []
    summaries = []
    for seed in seeds:
        split_path = split_dir / f"splits_seed{seed}.csv"
        split = pd.read_csv(split_path, usecols=["image_id", "split"])
        split_map = dict(zip(split["image_id"].astype(str), split["split"].astype(str), strict=True))
        block = pairs.copy()
        block["seed"] = int(seed)
        block["left_split"] = block["left_image_id"].astype(str).map(split_map)
        block["right_split"] = block["right_image_id"].astype(str).map(split_map)
        block["cross_split"] = block["left_split"] != block["right_split"]
        details.append(block[block["cross_split"]])
        summaries.append(
            {
                "seed": int(seed),
                "near_duplicate_pairs": int(len(block)),
                "cross_split_pairs": int(block["cross_split"].sum()),
                "cross_class_pairs": int(block["cross_class"].sum()),
                "cross_class_and_split_pairs": int(
                    (block["cross_class"] & block["cross_split"]).sum()
                ),
            }
        )
    detail = pd.concat(details, ignore_index=True) if details else pd.DataFrame()
    return detail, pd.DataFrame(summaries)


def _group_split_audit(
    manifest: pd.DataFrame,
    split_dir: Path,
    seeds: list[int],
) -> pd.DataFrame:
    rows = []
    for concept, candidates in PROVENANCE_FIELDS.items():
        columns = [column for column in candidates if column in manifest]
        for column in columns:
            source = manifest[["image_id", column]].dropna().copy()
            source[column] = source[column].astype(str).str.strip()
            source = source[~source[column].isin(["", "nan", "None"])]
            for seed in seeds:
                split = pd.read_csv(
                    split_dir / f"splits_seed{seed}.csv", usecols=["image_id", "split"]
                )
                merged = source.merge(split, on="image_id", how="inner")
                crossings = merged.groupby(column)["split"].nunique()
                rows.append(
                    {
                        "concept": concept,
                        "column": column,
                        "seed": int(seed),
                        "groups": int(len(crossings)),
                        "groups_crossing_splits": int((crossings > 1).sum()),
                    }
                )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", default="results/manifest.content_cleaned_conservative.csv"
    )
    parser.add_argument(
        "--split-dir", default="results/splits_content_cleaned_conservative"
    )
    parser.add_argument("--backbones", default="dinov2_vitb14,dinov2_vitl14")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--similarity-threshold", type=float, default=0.995)
    parser.add_argument("--neighbors", type=int, default=5)
    parser.add_argument("--ef-search", type=int, default=128)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output = resolve_path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(resolve_path(args.manifest), low_memory=False)
    split_dir = resolve_path(args.split_dir)
    seeds = [int(value) for value in args.seeds.replace(" ", ",").split(",") if value]
    backbones = [
        value for value in args.backbones.replace(" ", ",").split(",") if value
    ]

    coverage = _metadata_coverage(manifest)
    exact = _exact_duplicates(manifest)
    group_audit = _group_split_audit(manifest, split_dir, seeds)
    coverage.to_csv(output / "metadata_coverage.csv", index=False)
    exact.to_csv(output / "exact_duplicate_groups.csv", index=False)
    group_audit.to_csv(output / "provenance_group_split_audit.csv", index=False)

    feature_summaries = []
    for backbone in backbones:
        pairs = _feature_pairs(
            backbone,
            threshold=float(args.similarity_threshold),
            neighbors=int(args.neighbors),
            ef_search=int(args.ef_search),
        )
        pairs.to_csv(output / f"feature_near_duplicates_{backbone}.csv", index=False)
        crossing, summary = _split_crossings(pairs, split_dir, seeds)
        crossing.to_csv(output / f"cross_split_near_duplicates_{backbone}.csv", index=False)
        summary.insert(0, "backbone", backbone)
        feature_summaries.append(summary)
    feature_summary = pd.concat(feature_summaries, ignore_index=True)
    feature_summary.to_csv(output / "feature_leakage_summary.csv", index=False)

    report = {
        "manifest_rows": int(len(manifest)),
        "class_counts": {
            str(key): int(value)
            for key, value in manifest.groupby("class_name").size().items()
        },
        "exact_duplicate_groups": int(len(exact)),
        "exact_cross_class_groups": int(exact.get("cross_class", pd.Series(dtype=bool)).sum()),
        "feature_similarity_threshold": float(args.similarity_threshold),
        "metadata_missing_concepts": coverage[
            ~coverage["status"].eq("available_complete")
        ]["concept"].tolist(),
        "leakage_claim_allowed": bool(
            len(exact) == 0
            and int(feature_summary["cross_split_pairs"].sum()) == 0
            and coverage["status"].eq("available_complete").all()
        ),
    }
    (output / "audit_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
