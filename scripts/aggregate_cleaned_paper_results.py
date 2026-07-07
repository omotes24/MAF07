#!/usr/bin/env python
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from maf07 import TARGET_CLASSES
from maf07.config import load_yaml


METRICS = ["AUROC", "FPR95", "AUPR_OUT"]
EXPECTED_COUNTS = {2: 168, 3: 336, 4: 420, 5: 336, 6: 168, 7: 48}
DISCARDED = {"card", "cqs", "lar"}


def _classes() -> list[str]:
    dcfg = load_yaml("configs/dataset.yaml")
    return list(dcfg.get("classes", TARGET_CLASSES))


def _id_set_ids(classes: list[str], id_size: int) -> dict[str, int]:
    return {"|".join(id_set): idx for idx, id_set in enumerate(itertools.combinations(classes, id_size))}


def _normalize_fold_frame(path: Path, *, source: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        return df
    if "protocol" in df.columns and "scope" not in df.columns:
        df["scope"] = df["protocol"]
    if "scope" not in df.columns:
        df["scope"] = "fair"
    if "variant" not in df.columns:
        df["variant"] = "main"
    if "method" in df.columns:
        df["method"] = df["method"].replace({"rsn": "diag_huber_raw"})
    if "variant" in df.columns:
        # Legacy DiagCARD outputs used method=diagcard, variant=diag_raw, etc.
        mask = df.get("method", "").eq("diagcard") & df["variant"].astype(str).str.startswith("diag_")
        if mask.any():
            df.loc[mask, "method"] = df.loc[mask, "variant"]
            df.loc[mask, "variant"] = "main"
    classes = _classes()
    if "id_set_id" not in df.columns:
        ids = []
        for _, row in df.iterrows():
            ids.append(_id_set_ids(classes, int(row["id_size"]))[str(row["id_set"])])
        df["id_set_id"] = ids
    if "n" not in df.columns:
        df["n"] = 1
    df["source"] = source
    keep = [
        "job_id",
        "scope",
        "backbone",
        "id_size",
        "id_set_id",
        "seed",
        "method",
        "variant",
        "id_set",
        "ood_set",
        "n",
        "AUROC",
        "FPR95",
        "AUPR_OUT",
        "source",
    ]
    for col in keep:
        if col not in df.columns:
            df[col] = np.nan
    df = df[keep].copy()
    df = df[~df["method"].isin(DISCARDED)].copy()
    return df


def _read_all(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        if path.exists() and path.stat().st_size > 0:
            frames.append(_normalize_fold_frame(path, source=path.name))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True, sort=False)
    df = df.drop_duplicates(["backbone", "id_size", "id_set_id", "seed", "method"], keep="last")
    return df.sort_values(["id_size", "backbone", "seed", "id_set_id", "method"]).reset_index(drop=True)


def _summary(df: pd.DataFrame, *, methods: list[str] | None = None, id_sizes: list[int] | None = None) -> pd.DataFrame:
    sub = df.copy()
    if methods is not None:
        sub = sub[sub["method"].isin(methods)].copy()
    if id_sizes is not None:
        sub = sub[sub["id_size"].isin(id_sizes)].copy()
    rows = []
    for keys, block in sub.groupby(["backbone", "id_size", "method"], dropna=False):
        rows.append(
            {
                "backbone": keys[0],
                "id_size": int(keys[1]),
                "method": keys[2],
                "n": int(len(block)),
                **{metric: float(block[metric].mean()) for metric in METRICS},
            }
        )
    for keys, block in sub.groupby(["id_size", "method"], dropna=False):
        rows.append(
            {
                "backbone": "ALL",
                "id_size": int(keys[0]),
                "method": keys[1],
                "n": int(len(block)),
                **{metric: float(block[metric].mean()) for metric in METRICS},
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    for metric in METRICS:
        out[metric] = out[metric].round(6)
    return out.sort_values(["id_size", "backbone", "method"]).reset_index(drop=True)


def _paired(
    df: pd.DataFrame,
    method_a: str,
    method_b: str,
    *,
    by_id_size: bool = False,
    alpha: float = 0.05,
) -> pd.DataFrame:
    keys = ["backbone", "seed", "id_size", "id_set_id"]
    piv = df[df["method"].isin([method_a, method_b])].pivot_table(
        index=keys,
        columns="method",
        values=METRICS,
        aggfunc="first",
    )
    rows = []
    for metric in METRICS:
        if (metric, method_a) not in piv or (metric, method_b) not in piv:
            continue
        base = piv[[(metric, method_a), (metric, method_b)]].dropna().copy()
        base.columns = ["a", "b"]
        groups = [(None, base)]
        if by_id_size:
            groups = [(int(k), g) for k, g in base.groupby(level="id_size")]
        for id_size, block in groups:
            if metric == "FPR95":
                diff = block["b"].to_numpy(dtype=float) - block["a"].to_numpy(dtype=float)
            else:
                diff = block["a"].to_numpy(dtype=float) - block["b"].to_numpy(dtype=float)
            n = len(diff)
            mean = float(np.mean(diff))
            if n > 1:
                se = float(stats.sem(diff))
                half = float(stats.t.ppf(1.0 - alpha / 2.0, n - 1) * se)
            else:
                half = float("nan")
            row = {
                "method_a": method_a,
                "method_b": method_b,
                "metric": metric,
                "n": int(n),
                "mean_diff": mean,
                "ci_low": mean - half,
                "ci_high": mean + half,
                "good_rate": float(np.mean(diff > 0)),
            }
            if by_id_size:
                row["id_size"] = id_size
            rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        for col in ["mean_diff", "ci_low", "ci_high", "good_rate"]:
            out[col] = out[col].round(6)
    return out


def _bit_identical_checks(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    checks = [
        ("mahalanobis_family", ["mahalanobis", "mah_mindist", "mahalanobispp"]),
        ("clip_family", ["clip_zeroshot_msp", "mcm"]),
    ]
    keys = ["backbone", "seed", "id_size", "id_set_id"]
    for name, methods in checks:
        sub = df[df["method"].isin(methods)].copy()
        present = sorted(sub["method"].dropna().unique().tolist())
        row = {
            "check": name,
            "methods": "|".join(methods),
            "present_methods": "|".join(present),
            "complete": set(methods).issubset(present),
            "n_pairs": 0,
            "all_metrics_bit_identical": False,
        }
        if set(methods).issubset(present):
            piv = sub.pivot_table(index=keys, columns="method", values=METRICS, aggfunc="first")
            complete_mask = np.ones(len(piv), dtype=bool)
            for method in methods:
                for metric in METRICS:
                    complete_mask &= piv[(metric, method)].notna().to_numpy()
            piv = piv.loc[complete_mask]
            row["n_pairs"] = int(len(piv))
            all_equal = True
            for metric in METRICS:
                ref = piv[(metric, methods[0])].to_numpy()
                metric_equal = all(np.array_equal(ref, piv[(metric, method)].to_numpy()) for method in methods[1:])
                row[f"{metric}_bit_identical"] = bool(metric_equal)
                all_equal &= metric_equal
            row["all_metrics_bit_identical"] = bool(all_equal)
        rows.append(row)
    return pd.DataFrame(rows)


def _read_per_ood(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["method"] = df["method"].replace({"rsn": "diag_huber_raw"})
    df = df[~df["method"].isin(DISCARDED)].copy()
    return df.drop_duplicates(["job_id", "ood_class"], keep="last")


def _per_ood_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    rows = []
    groups = ["backbone", "id_size", "method", "ood_class"]
    for keys, block in df.groupby(groups, dropna=False):
        rows.append(
            {
                "backbone": keys[0],
                "id_size": int(keys[1]),
                "method": keys[2],
                "ood_class": keys[3],
                "n": int(len(block)),
                **{metric: float(block[metric].mean()) for metric in METRICS},
            }
        )
    for keys, block in df.groupby(["id_size", "method", "ood_class"], dropna=False):
        rows.append(
            {
                "backbone": "ALL",
                "id_size": int(keys[0]),
                "method": keys[1],
                "ood_class": keys[2],
                "n": int(len(block)),
                **{metric: float(block[metric].mean()) for metric in METRICS},
            }
        )
    for keys, block in df.groupby(["backbone", "method", "ood_class"], dropna=False):
        rows.append(
            {
                "backbone": keys[0],
                "id_size": "ALL",
                "method": keys[1],
                "ood_class": keys[2],
                "n": int(len(block)),
                **{metric: float(block[metric].mean()) for metric in METRICS},
            }
        )
    for keys, block in df.groupby(["method", "ood_class"], dropna=False):
        rows.append(
            {
                "backbone": "ALL",
                "id_size": "ALL",
                "method": keys[0],
                "ood_class": keys[1],
                "n": int(len(block)),
                **{metric: float(block[metric].mean()) for metric in METRICS},
            }
        )
    out = pd.DataFrame(rows)
    for metric in METRICS:
        out[metric] = out[metric].round(6)
    return out.sort_values(["method", "backbone", "id_size", "ood_class"]).reset_index(drop=True)


def _write_figures(out_dir: Path, tables: dict[str, pd.DataFrame]) -> list[Path]:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    made: list[Path] = []

    main = tables.get("B_main_idsize")
    if main is not None and not main.empty:
        sub = main[(main["backbone"] == "ALL") & (main["method"].isin(["diag_huber_raw", "knn", "psm"]))]
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), dpi=160)
        for method, g in sub.groupby("method"):
            label = "RSN" if method == "diag_huber_raw" else method.upper()
            axes[0].plot(g["id_size"], g["AUROC"], marker="o", label=label)
            axes[1].plot(g["id_size"], g["FPR95"], marker="o", label=label)
        axes[0].set_title("AUROC vs ID class size")
        axes[1].set_title("FPR95 vs ID class size")
        for ax in axes:
            ax.set_xlabel("ID class size m")
            ax.grid(alpha=0.25)
            ax.legend(frameon=False)
        axes[0].set_ylabel("AUROC")
        axes[1].set_ylabel("FPR95")
        fig.tight_layout()
        path = fig_dir / "fig_idsize_trend.png"
        fig.savefig(path)
        plt.close(fig)
        made.append(path)

        bsub = main[(main["backbone"] != "ALL") & (main["method"].isin(["diag_huber_raw", "knn", "psm"]))]
        fig, ax = plt.subplots(figsize=(8, 4.2), dpi=160)
        for (method, backbone), g in bsub.groupby(["method", "backbone"]):
            label = f"{'RSN' if method == 'diag_huber_raw' else method.upper()} / {backbone.replace('dinov2_', '')}"
            ax.plot(g["id_size"], g["AUROC"], marker="o", label=label)
        ax.set_title("Backbone consistency")
        ax.set_xlabel("ID class size m")
        ax.set_ylabel("AUROC")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8, ncol=2)
        fig.tight_layout()
        path = fig_dir / "fig_backbone_consistency.png"
        fig.savefig(path)
        plt.close(fig)
        made.append(path)

    per_ood = tables.get("D_per_ood")
    if per_ood is not None and not per_ood.empty:
        sub = per_ood[
            (per_ood["backbone"] == "ALL")
            & (per_ood["id_size"].astype(str) == "ALL")
            & (per_ood["method"] == "diag_huber_raw")
        ].sort_values("AUROC")
        fig, ax = plt.subplots(figsize=(7, 4.2), dpi=160)
        ax.barh(sub["ood_class"], sub["AUROC"], color="#4C78A8")
        ax.set_title("Per-OOD-class AUROC (RSN)")
        ax.set_xlabel("AUROC")
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        path = fig_dir / "fig_per_ood_class.png"
        fig.savefig(path)
        plt.close(fig)
        made.append(path)

        wide = per_ood[
            (per_ood["backbone"] == "ALL")
            & (per_ood["id_size"].astype(str) == "ALL")
            & (per_ood["method"].isin(["diag_huber_raw", "diag_raw"]))
        ].pivot_table(index="ood_class", columns="method", values="AUROC", aggfunc="first")
        if {"diag_huber_raw", "diag_raw"}.issubset(wide.columns):
            delta = (wide["diag_huber_raw"] - wide["diag_raw"]).sort_values()
            fig, ax = plt.subplots(figsize=(7, 4.2), dpi=160)
            colors = ["#D55E00" if v < 0 else "#009E73" for v in delta]
            ax.barh(delta.index, delta.values, color=colors)
            ax.axvline(0, color="black", linewidth=1)
            ax.set_title("Huber contribution by OOD species")
            ax.set_xlabel("AUROC(RSN) - AUROC(diag_raw)")
            ax.grid(axis="x", alpha=0.25)
            fig.tight_layout()
            path = fig_dir / "fig_huber_species_delta.png"
            fig.savefig(path)
            plt.close(fig)
            made.append(path)

    ranking = tables.get("C_full_ranking")
    if ranking is not None and not ranking.empty:
        sub = ranking.sort_values("AUROC", ascending=True)
        fig_h = max(5.0, 0.22 * len(sub))
        fig, ax = plt.subplots(figsize=(8, fig_h), dpi=160)
        colors = ["#009E73" if m.startswith("diag_huber_raw") else "#4C78A8" for m in sub["method"]]
        ax.barh(sub["method"], sub["AUROC"], color=colors)
        ax.set_title("Full method ranking (m=2, AUROC)")
        ax.set_xlabel("AUROC")
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        path = fig_dir / "fig_full_ranking.png"
        fig.savefig(path)
        plt.close(fig)
        made.append(path)

    cdf_path = out_dir / "shared_shape_representative_cdf.csv"
    if cdf_path.exists():
        cdf = pd.read_csv(cdf_path)
        fig, ax = plt.subplots(figsize=(7, 4.2), dpi=160)
        for cls, g in cdf.groupby("class_name"):
            ax.plot(g["distance"], g["cdf"], label=cls)
        ax.set_title("Shared-shape check: validation CDFs")
        ax.set_xlabel("RSN validation distance")
        ax.set_ylabel("Empirical CDF")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8, ncol=2)
        fig.tight_layout()
        path = fig_dir / "fig_shared_shape_cdf.png"
        fig.savefig(path)
        plt.close(fig)
        made.append(path)

    return made


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--fold-input", action="append", default=[])
    parser.add_argument("--per-ood-input", default="")
    parser.add_argument("--ks-summary", default="")
    parser.add_argument("--make-figures", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fold = _read_all([Path(p) for p in args.fold_input])
    if fold.empty:
        raise SystemExit("No fold-level inputs found")
    fold.to_csv(out_dir / "fold_level.csv", index=False)

    tables: dict[str, pd.DataFrame] = {}
    tables["A_ablation_m2"] = _summary(
        fold,
        methods=["diag_raw", "diag_calib", "diag_huber_raw", "diag_huber_calib"],
        id_sizes=[2],
    )
    tables["B_main_idsize"] = _summary(fold, methods=["diag_huber_raw", "knn", "psm"])
    ranking = _summary(fold[fold["id_size"] == 2])
    ranking = ranking[ranking["backbone"] == "ALL"].sort_values("AUROC", ascending=False).reset_index(drop=True)
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
    tables["C_full_ranking"] = ranking

    per_ood = _read_per_ood(Path(args.per_ood_input)) if args.per_ood_input else pd.DataFrame()
    tables["D_per_ood"] = _per_ood_summary(per_ood)
    tables["E_pair_rsn_vs_knn_psm"] = pd.concat(
        [
            _paired(fold, "diag_huber_raw", "knn"),
            _paired(fold, "diag_huber_raw", "psm"),
        ],
        ignore_index=True,
    )
    tables["F_huber_vs_raw"] = _paired(fold, "diag_huber_raw", "diag_raw")
    tables["G_huber_fpr95_by_idsize"] = _paired(fold, "diag_huber_raw", "diag_raw", by_id_size=True)
    if not tables["G_huber_fpr95_by_idsize"].empty:
        tables["G_huber_fpr95_by_idsize"] = tables["G_huber_fpr95_by_idsize"][
            tables["G_huber_fpr95_by_idsize"]["metric"] == "FPR95"
        ].copy()
    tables["H_backbone_main_comparison"] = tables["B_main_idsize"][
        tables["B_main_idsize"]["backbone"] != "ALL"
    ].copy()
    tables["H_backbone_per_ood"] = tables["D_per_ood"][
        tables["D_per_ood"].get("backbone", pd.Series(dtype=str)) != "ALL"
    ].copy() if not tables["D_per_ood"].empty else pd.DataFrame()

    if args.ks_summary and Path(args.ks_summary).exists():
        ks = pd.read_csv(args.ks_summary).drop_duplicates("job_id", keep="last")
        tables["I_shared_shape"] = pd.DataFrame(
            [
                {
                    "backbone": "ALL",
                    "id_size": "ALL",
                    "n": int(len(ks)),
                    "ks_mean": float(ks["ks_mean"].mean()),
                    "ks_median": float(ks["ks_median"].mean()),
                    "ks_max": float(ks["ks_max"].mean()),
                    "mean_distance_cv": float(ks["mean_distance_cv"].mean()),
                }
            ]
        )
    else:
        tables["I_shared_shape"] = pd.DataFrame()

    pair_id = pd.concat(
        [
            _paired(fold, "diag_huber_raw", "knn", by_id_size=True, alpha=0.05 / 18.0),
            _paired(fold, "diag_huber_raw", "psm", by_id_size=True, alpha=0.05 / 18.0),
            _paired(fold, "diag_huber_raw", "diag_raw", by_id_size=True, alpha=0.05 / 18.0),
        ],
        ignore_index=True,
    )
    if not pair_id.empty:
        alpha = 0.05 / 18.0
        pair_id["bonferroni_alpha"] = alpha
        pair_id["bonferroni_significant"] = ~((pair_id["ci_low"] <= 0) & (pair_id["ci_high"] >= 0))
    tables["J_bonferroni_checks"] = pair_id
    tables["implementation_checks"] = _bit_identical_checks(fold)

    for name, table in tables.items():
        table.to_csv(out_dir / f"{name}.csv", index=False)

    manifest = {
        "fold_rows": int(len(fold)),
        "methods": sorted(fold["method"].dropna().unique().tolist()),
        "outputs": sorted(str(p.name) for p in out_dir.glob("*.csv")),
    }
    if args.make_figures:
        manifest["figures"] = [str(p) for p in _write_figures(out_dir, tables)]
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
