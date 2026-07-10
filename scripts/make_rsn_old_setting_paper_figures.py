#!/usr/bin/env python3
"""Build the figures used by the RSN old-setting paper rewrite.

All quantitative panels are derived from the cleaned CAT outputs or the
verified-v1 baseline audit. The script intentionally does not read the
invalidated archived full-ranking table.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "paper" / "figures"
CLEAN = ROOT / "docs" / "results" / "rsn_cleaned_20260708"
VERIFIED = ROOT / "docs" / "results" / "rsn_baseline_audit_20260710"

COLORS = {
    "RSN": "#007f73",
    "KNN": "#e1812c",
    "PSM": "#5b6f9f",
    "rosette": "#c65d3a",
    "spot": "#d7a62f",
    "plain": "#4f8f76",
    "stripe": "#6a6a6a",
    "distance": "#4778a8",
    "classifier": "#8a8a8a",
}

METHOD_LABELS = {
    "rsn_reported": "RSN (ours)",
    "knn": "KNN",
    "vim": "ViM",
    "mahalanobis": "Mahalanobis",
    "mahalanobispp": "Mahalanobis++",
    "dice": "DICE",
    "entropy": "Entropy",
    "gen": "GEN",
    "msp": "MSP",
    "openmax": "OpenMax",
    "maxlogit": "MaxLogit",
    "energy": "Energy",
    "gradnorm": "GradNorm",
    "scale": "SCALE",
    "rmd": "RMD",
    "kl_matching": "KL-Matching",
    "nci": "NCI",
    "react": "ReAct",
    "ashb": "ASH-B",
    "ashp": "ASH-P",
    "ashs": "ASH-S",
}

SPECIES_PATTERN = {
    "cheetah": "spot",
    "jaguar": "rosette",
    "leopard": "rosette",
    "lion": "plain",
    "ocelot": "rosette",
    "puma": "plain",
    "serval": "spot",
    "tiger": "stripe",
}


def configure() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.fontsize": 8,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save(fig: plt.Figure, stem: str) -> None:
    for suffix in ("pdf", "png"):
        fig.savefig(OUT / f"{stem}.{suffix}", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def pipeline_figure() -> None:
    fig = plt.figure(figsize=(12.4, 3.7))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)

    sample_path = ROOT / "docs" / "paper" / "assets" / "lion_example.jpg"
    with Image.open(sample_path) as image:
        sample = ImageOps.fit(image.convert("RGB"), (430, 430), method=Image.Resampling.LANCZOS)
    ax.imshow(sample, extent=(0.02, 0.14, 0.31, 0.69), zorder=1, aspect="auto")
    ax.text(0.08, 0.255, "Input image", ha="center", weight="bold", fontsize=9)

    boxes = [
        (0.18, 0.36, 0.12, 0.28, "DINOv2\nraw h(x)", "#d8e9f0"),
        (0.34, 0.34, 0.15, 0.32, "class bank\nB_c, sigma_c", "#f5e6bd"),
        (0.53, 0.34, 0.14, 0.32, "scaled kNN\nk = 150", "#dce9cc"),
        (0.71, 0.36, 0.12, 0.28, "Huber\ndelta = 1.345", "#ead8e1"),
        (0.87, 0.36, 0.10, 0.28, "OOD\nmin_c r_c", "#dedbea"),
    ]
    for x, y, w, h, label, color in boxes:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.012,rounding_size=0.016",
                linewidth=1.2,
                edgecolor="#333333",
                facecolor=color,
                zorder=3,
            )
        )
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", weight="bold", fontsize=6.4, zorder=4)

    def arrow(start: tuple[float, float], end: tuple[float, float], rad: float = 0.0) -> None:
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=15,
                linewidth=1.5,
                color="#333333",
                connectionstyle=f"arc3,rad={rad}",
            )
        )

    arrow((0.14, 0.50), (0.18, 0.50))
    arrow((0.30, 0.50), (0.34, 0.50))
    arrow((0.49, 0.50), (0.53, 0.50))
    arrow((0.67, 0.50), (0.71, 0.50))
    arrow((0.83, 0.50), (0.87, 0.50))

    ax.text(0.5, 0.88, "RSN: Robust Scale-Normalized kNN", ha="center", fontsize=16, weight="bold")
    ax.text(
        0.5,
        0.09,
        "Main setting: no L2 normalization, no classifier candidate pruning, and no empirical calibration.",
        ha="center",
        fontsize=9,
    )
    save(fig, "fig01_rsn_old_pipeline")


def idsize_and_backbone_figure() -> None:
    main = pd.read_csv(CLEAN / "B_main_idsize.csv")
    methods = {"diag_huber_raw": "RSN", "knn": "KNN", "psm": "PSM"}
    pooled = main[(main["backbone"] == "ALL") & main["method"].isin(methods)].copy()
    pooled["label"] = pooled["method"].map(methods)

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.55))
    for metric, ax, better in [("AUROC", axes[0], "higher is better"), ("FPR95", axes[1], "lower is better")]:
        for label in ("RSN", "KNN", "PSM"):
            block = pooled[pooled["label"] == label].sort_values("id_size")
            ax.plot(
                block["id_size"],
                block[metric],
                marker="o",
                linewidth=2.1,
                markersize=5,
                label=label,
                color=COLORS[label],
            )
        ax.set_xticks(range(2, 8))
        ax.set_xlabel("number of ID classes m")
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} ({better})")
        ax.grid(alpha=0.22)
    axes[0].legend(frameon=False, loc="lower left")

    b2 = main[(main["id_size"] == 2) & (main["backbone"] != "ALL") & main["method"].isin(methods)].copy()
    b2["label"] = b2["method"].map(methods)
    x = np.arange(2)
    width = 0.23
    for offset, label in zip((-width, 0.0, width), ("RSN", "KNN", "PSM"), strict=True):
        vals = []
        for backbone in ("dinov2_vitb14", "dinov2_vitl14"):
            vals.append(float(b2[(b2["backbone"] == backbone) & (b2["label"] == label)]["AUROC"].iloc[0]))
        axes[2].bar(x + offset, vals, width, label=label, color=COLORS[label])
        for xpos, value in zip(x + offset, vals, strict=True):
            axes[2].text(xpos, value + 0.0014, f"{value:.3f}", ha="center", fontsize=7, rotation=90)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(["ViT-B/14", "ViT-L/14"])
    axes[2].set_ylim(0.87, 0.935)
    axes[2].set_ylabel("AUROC")
    axes[2].set_title("Backbone check at m=2")
    axes[2].grid(axis="y", alpha=0.22)
    axes[2].legend(frameon=False, loc="lower right")
    fig.suptitle("RSN remains ahead across ID-set size and both DINOv2 backbones", fontsize=13, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, "fig02_idsize_backbone")


def verified_ranking_figure() -> None:
    summary = pd.read_csv(VERIFIED / "summary_verified.csv")
    block = summary[(summary["backbone"] == "ALL") & (summary["id_size"] == 2)].copy()
    block = block[block["method"] != "rsn_paper"].sort_values("AUROC", ascending=True)
    block["label"] = block["method"].map(METHOD_LABELS)
    distance = {"knn", "vim", "mahalanobis", "mahalanobispp", "rmd"}
    colors = []
    for method in block["method"]:
        if method == "rsn_reported":
            colors.append(COLORS["RSN"])
        elif method == "knn":
            colors.append(COLORS["KNN"])
        elif method in distance:
            colors.append(COLORS["distance"])
        else:
            colors.append(COLORS["classifier"])

    fig, ax = plt.subplots(figsize=(7.5, 6.8))
    y = np.arange(len(block))
    bars = ax.barh(y, block["AUROC"], color=colors, edgecolor="white", linewidth=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels(block["label"])
    ax.set_xlim(0.45, 0.93)
    ax.set_xlabel("AUROC")
    ax.set_title("Verified DINOv2 baseline ranking (m=2, n=168)", weight="bold")
    ax.grid(axis="x", alpha=0.2)
    for bar, value in zip(bars, block["AUROC"], strict=True):
        ax.text(value + 0.003, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center", fontsize=7)
    ax.text(
        0.01,
        -0.085,
        "All rows use verified_v1 implementations. ODIN and CLIP-only methods are excluded from the cached-DINO table.",
        transform=ax.transAxes,
        fontsize=8,
    )
    fig.tight_layout()
    save(fig, "fig03_verified_ranking")


def per_ood_figure() -> None:
    per_ood = pd.read_csv(CLEAN / "D_per_ood.csv")
    block = per_ood[
        (per_ood["backbone"] == "ALL")
        & (per_ood["id_size"].astype(str) == "2")
        & (per_ood["method"] == "diag_huber_raw")
    ].sort_values("AUROC")
    labels = block["ood_class"].str.title().tolist()
    colors = [COLORS[SPECIES_PATTERN[name]] for name in block["ood_class"]]

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.7), sharey=True)
    y = np.arange(len(block))
    for ax, metric, title in [
        (axes[0], "AUROC", "AUROC (higher is better)"),
        (axes[1], "FPR95", "FPR95 (lower is better)"),
    ]:
        bars = ax.barh(y, block[metric], color=colors, edgecolor="#333333", linewidth=0.5)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.set_xlabel(metric)
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.2)
        for bar, value in zip(bars, block[metric], strict=True):
            ax.text(value + 0.008, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center", fontsize=7)
    axes[0].set_xlim(0.80, 1.01)
    axes[1].set_xlim(0.0, 0.78)
    fig.suptitle("Per-OOD-species performance of RSN at m=2", fontsize=13, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, "fig04_per_ood_m2")


def shared_shape_figure() -> None:
    cdf = pd.read_csv(CLEAN / "shared_shape_representative_cdf.csv")
    summary = pd.read_csv(CLEAN / "I_shared_shape.csv").iloc[0]
    fig, ax = plt.subplots(figsize=(7.1, 4.2))
    palette = plt.get_cmap("tab10")
    for i, (name, block) in enumerate(cdf.groupby("class_name")):
        ax.plot(block["distance"], block["cdf"], label=name.title(), linewidth=1.5, color=palette(i))
    ax.set_xlabel("RSN validation distance")
    ax.set_ylabel("empirical CDF")
    ax.set_title("Representative fold: class-conditional validation CDFs", weight="bold")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, ncol=2, loc="lower right")
    ax.text(
        0.03,
        0.96,
        f"all-fold mean KS = {summary['ks_mean']:.3f}\nall-fold mean CV = {summary['mean_distance_cv']:.3f}",
        transform=ax.transAxes,
        va="top",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#555555", alpha=0.9),
    )
    fig.tight_layout()
    save(fig, "fig05_shared_shape_cdf")


def huber_species_figure() -> None:
    per_ood = pd.read_csv(CLEAN / "D_per_ood.csv")
    block = per_ood[
        (per_ood["backbone"] == "ALL")
        & (per_ood["id_size"].astype(str) == "ALL")
        & per_ood["method"].isin(["diag_huber_raw", "diag_raw"])
    ]
    wide = block.pivot(index="ood_class", columns="method", values="AUROC")
    delta = (wide["diag_huber_raw"] - wide["diag_raw"]).sort_values()
    colors = [COLORS["RSN"] if value >= 0 else COLORS["KNN"] for value in delta]
    fig, ax = plt.subplots(figsize=(6.7, 3.8))
    bars = ax.barh(np.arange(len(delta)), delta.to_numpy(), color=colors)
    ax.set_yticks(np.arange(len(delta)))
    ax.set_yticklabels([name.title() for name in delta.index])
    ax.axvline(0, color="#222222", linewidth=0.9)
    ax.set_xlim(-0.00055, 0.0013)
    ax.set_xlabel("AUROC(RSN) - AUROC(RSN without Huber)")
    ax.set_title("Huber contribution by OOD species", weight="bold")
    ax.grid(axis="x", alpha=0.2)
    for bar, value in zip(bars, delta.to_numpy(), strict=True):
        ha = "left" if value >= 0 else "right"
        ax.text(value + (0.00002 if value >= 0 else -0.00002), bar.get_y() + bar.get_height() / 2, f"{value:+.4f}", va="center", ha=ha, fontsize=7)
    fig.tight_layout()
    save(fig, "fig06_huber_species")


def main() -> None:
    configure()
    pipeline_figure()
    idsize_and_backbone_figure()
    verified_ranking_figure()
    per_ood_figure()
    shared_shape_figure()
    huber_species_figure()


if __name__ == "__main__":
    main()
