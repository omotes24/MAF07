from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import resolve_path
from .jobs import audit_coverage


def aggregate_scores(protocol: str = "fair") -> Path | None:
    root = resolve_path(f"results/ood/{protocol}/scores/jobs")
    out = resolve_path(f"results/ood/{protocol}/all_scores.parquet")
    if not root.exists():
        return None
    paths = sorted(root.glob("*.parquet"))
    if not paths:
        return None
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        frames = [pd.read_parquet(path) for path in paths]
        pd.concat(frames, ignore_index=True).to_parquet(out, index=False)
        return out

    writer: pq.ParquetWriter | None = None
    try:
        for path in paths:
            frame = pd.read_parquet(path)
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(out, table.schema)
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    return out


def aggregate_fair_summaries() -> dict[str, Path]:
    summary = resolve_path("results/ood/fair/summary_by_setting.csv")
    outputs: dict[str, Path] = {}
    if not summary.exists():
        return outputs
    df = pd.read_csv(summary)
    groupings = {
        "summary_by_id_size.csv": ["id_size"],
        "summary_by_backbone.csv": ["backbone"],
        "summary_by_method.csv": ["method"],
        "summary_by_ood_class.csv": ["ood_set"],
        "summary_by_protocol.csv": ["protocol"],
    }
    metrics = [c for c in ["AUROC", "FPR95", "AUPR_IN", "AUPR_OUT", "DetectionError", "OSCR"] if c in df]
    for filename, cols in groupings.items():
        out = resolve_path(f"results/ood/fair/{filename}")
        grouped = df.groupby(cols, dropna=False)[metrics].agg(["mean", "std", "count"])
        grouped.columns = ["_".join(col).strip("_") for col in grouped.columns.to_flat_index()]
        grouped.reset_index().to_csv(out, index=False)
        outputs[filename] = out
    return outputs


def make_tables() -> list[Path]:
    outputs: list[Path] = []
    tables_dir = resolve_path("results/tables")
    tables_dir.mkdir(parents=True, exist_ok=True)
    fair = resolve_path("results/ood/fair/summary_by_setting.csv")
    if fair.exists():
        df = pd.read_csv(fair)
        if not df.empty:
            best_auroc = df.sort_values("AUROC", ascending=False).head(20)
            out = tables_dir / "best_fair_by_auroc.csv"
            best_auroc.to_csv(out, index=False)
            outputs.append(out)
            best_fpr = df.sort_values("FPR95", ascending=True).head(20)
            out = tables_dir / "best_fair_by_fpr95.csv"
            best_fpr.to_csv(out, index=False)
            outputs.append(out)
    oracle = resolve_path("results/ood/oracle/oracle_summary.csv")
    if oracle.exists():
        df = pd.read_csv(oracle)
        if not df.empty:
            out = tables_dir / "oracle_upper_bound.csv"
            df.assign(note="Oracle upper bound; not deployable").to_csv(out, index=False)
            outputs.append(out)
    audit_coverage(strict=False)
    return outputs
