#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from maf07.config import resolve_path


def _wilson(errors: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total == 0:
        return np.nan, np.nan
    p = errors / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total**2)) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def _sheet(block: pd.DataFrame, output: Path, columns: int = 10, tile: int = 150) -> None:
    rows = int(math.ceil(len(block) / columns))
    canvas = Image.new("RGB", (columns * tile, rows * (tile + 24)), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for index, row in enumerate(block.itertuples(index=False)):
        x = (index % columns) * tile
        y = (index // columns) * (tile + 24)
        try:
            with Image.open(str(row.path)) as image:
                thumb = ImageOps.fit(
                    image.convert("RGB"),
                    (tile, tile),
                    method=Image.Resampling.LANCZOS,
                )
            canvas.paste(thumb, (x, y))
        except OSError:
            draw.rectangle((x, y, x + tile, y + tile), fill="#dddddd")
            draw.text((x + 4, y + 4), "UNREADABLE", fill="red", font=font)
        draw.text((x + 3, y + tile + 4), str(row.audit_id), fill="black", font=font)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=92)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", default="results/manifest.content_cleaned_conservative.csv"
    )
    parser.add_argument("--samples-per-class", type=int, default=100)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--completed-csv", default="")
    args = parser.parse_args()

    output = resolve_path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(resolve_path(args.manifest), low_memory=False)
    sampled = []
    for class_name, block in manifest.groupby("class_name", sort=True):
        values = block.sample(
            min(args.samples_per_class, len(block)), random_state=20260712
        ).copy()
        values["expected_class"] = class_name
        sampled.append(values)
    audit = pd.concat(sampled, ignore_index=True)
    audit.insert(0, "audit_id", [f"A{index:05d}" for index in range(len(audit))])
    audit["reviewer_label"] = ""
    audit["is_correct"] = ""
    audit["is_clear_animal_image"] = ""
    audit["comments"] = ""
    columns = [
        "audit_id",
        "image_id",
        "expected_class",
        "path",
        "reviewer_label",
        "is_correct",
        "is_clear_animal_image",
        "comments",
    ]
    audit[columns].to_csv(output / "human_label_audit_template.csv", index=False)
    for class_name, block in audit.groupby("expected_class", sort=True):
        _sheet(block, output / f"contact_sheet_{class_name}.jpg")

    report: dict[str, object] = {
        "status": "awaiting_blinded_human_review",
        "sample_n": int(len(audit)),
        "samples_per_class": int(args.samples_per_class),
    }
    if args.completed_csv:
        completed = pd.read_csv(resolve_path(args.completed_csv))
        reviewed = completed[completed["is_correct"].notna()].copy()
        truthy = reviewed["is_correct"].astype(str).str.lower().isin(["1", "true", "yes", "y"])
        errors = int((~truthy).sum())
        low, high = _wilson(errors, len(reviewed))
        report.update(
            {
                "status": "complete",
                "reviewed_n": int(len(reviewed)),
                "label_error_n": errors,
                "label_error_rate": float(errors / len(reviewed)) if len(reviewed) else np.nan,
                "label_error_wilson95_low": float(low),
                "label_error_wilson95_high": float(high),
            }
        )
    (output / "human_label_audit_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
