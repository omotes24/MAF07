#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _existing_path(candidates: list[str]) -> Path:
    for item in candidates:
        path = Path(item)
        if path.exists():
            return path
    return Path(candidates[0])


def _metric_columns(df: pd.DataFrame) -> dict[str, str]:
    cols = {}
    for metric in ["AUROC", "FPR95", "AUPR_OUT"]:
        if metric in df:
            cols[metric] = metric
        elif f"{metric}_mean" in df:
            cols[metric] = f"{metric}_mean"
        else:
            raise ValueError(f"Missing metric column for {metric}")
    return cols


def _load_rsn_summary(path: Path, backbone: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "variant" in df:
        df = df[df["variant"].astype(str).isin(["main", "diag_huber_raw"])]
    df = df[df["backbone"].astype(str) == backbone].copy()
    metrics = _metric_columns(df)
    out = df[["protocol", "id_size", "backbone", "n"]].copy()
    out["rsn_n"] = out.pop("n")
    for metric, col in metrics.items():
        out[f"rsn_{metric}"] = df[col].astype(float).to_numpy()
    return out


def _load_baseline_summary(path: Path, backbone: str, methods: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "variant" in df:
        df = df[df["variant"].astype(str).isin(["main", ""])]
    df = df[df["backbone"].astype(str) == backbone].copy()
    df = df[df["method"].astype(str).isin(methods)].copy()
    metrics = _metric_columns(df)
    if all(f"{m}_mean" in df for m in ["AUROC", "FPR95", "AUPR_OUT"]) and "n" in df:
        grouped = df[["protocol", "id_size", "backbone", "method", "n"] + list(metrics.values())].copy()
        rename = {v: f"base_{k}" for k, v in metrics.items()}
        grouped = grouped.rename(columns=rename).rename(columns={"n": "baseline_n"})
        return grouped

    grouped = (
        df.groupby(["protocol", "id_size", "backbone", "method"], dropna=False)
        .agg(
            baseline_n=(metrics["AUROC"], "size"),
            base_AUROC=(metrics["AUROC"], "mean"),
            base_FPR95=(metrics["FPR95"], "mean"),
            base_AUPR_OUT=(metrics["AUPR_OUT"], "mean"),
        )
        .reset_index()
    )
    return grouped


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rsn-summary",
        default=str(
            _existing_path(
                [
                    "results/quick/rsn_full_summary_by_id_size.csv",
                    "results/quick/diagcard_full_huber_raw_summary_by_id_size.csv",
                ]
            )
        ),
    )
    parser.add_argument("--baseline-summary", default="results/ood/fair/summary_by_setting.csv")
    parser.add_argument("--methods", default="knn,card")
    parser.add_argument("--backbone", default="dinov2_vitb14")
    parser.add_argument("--output", default="results/analysis/rsn_vitb14_vs_knn_card_by_id_size.csv")
    args = parser.parse_args()

    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    rsn = _load_rsn_summary(Path(args.rsn_summary), args.backbone)
    base = _load_baseline_summary(Path(args.baseline_summary), args.backbone, methods)
    out = base.merge(rsn, on=["protocol", "id_size", "backbone"], how="inner")
    for metric in ["AUROC", "FPR95", "AUPR_OUT"]:
        out[f"diff_{metric}"] = out[f"rsn_{metric}"] - out[f"base_{metric}"]
    out = out.sort_values(["protocol", "id_size", "method"]).reset_index(drop=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
