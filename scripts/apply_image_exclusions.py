#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def load_exclusions(path: Path) -> set[str]:
    if not path.exists():
        return set()
    out: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        item = line.strip()
        if item and not item.startswith("#"):
            out.add(item)
    return out


def filter_csv(input_path: Path, output_path: Path, excluded_rel_paths: set[str]) -> tuple[int, int]:
    df = pd.read_csv(input_path)
    if "rel_path" not in df:
        raise ValueError(f"{input_path} has no rel_path column")
    before = len(df)
    out = df[~df["rel_path"].astype(str).isin(excluded_rel_paths)].copy()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    return before, len(out)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exclude-list", default="configs/excluded_images.txt")
    parser.add_argument("--manifest", default="results/manifest.csv")
    parser.add_argument("--manifest-output", default="results/manifest.cleaned.csv")
    parser.add_argument("--split-dir", default="results/splits")
    parser.add_argument("--split-output-dir", default="results/splits_cleaned")
    parser.add_argument("--seeds", default="0,1,2")
    args = parser.parse_args()

    excluded = load_exclusions(Path(args.exclude_list))
    rows = []
    before, after = filter_csv(Path(args.manifest), Path(args.manifest_output), excluded)
    rows.append({"file": args.manifest, "output": args.manifest_output, "before": before, "after": after})
    for raw_seed in args.seeds.split(","):
        seed = raw_seed.strip()
        if not seed:
            continue
        inp = Path(args.split_dir) / f"splits_seed{seed}.csv"
        out = Path(args.split_output_dir) / f"splits_seed{seed}.csv"
        before, after = filter_csv(inp, out, excluded)
        rows.append({"file": str(inp), "output": str(out), "before": before, "after": after})
    report = pd.DataFrame(rows)
    report_path = Path(args.manifest_output).with_suffix(".exclusion_report.csv")
    report.to_csv(report_path, index=False)
    print(report.to_string(index=False))
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
