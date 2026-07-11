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
import pandas as pd
from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "paper" / "figures"
CLEAN = ROOT / "docs" / "results" / "rsn_cleaned_20260708"

COLORS = {
    "RSN": "#007f73",
    "KNN": "#e1812c",
    "PSM": "#5b6f9f",
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

    sample_path = ROOT / "docs" / "paper" / "assets" / "ocelot_example.jpg"
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
    save(fig, "fig01_rsn_pipeline")


def idsize_trend_figure() -> None:
    main = pd.read_csv(CLEAN / "B_main_idsize.csv")
    methods = {"diag_huber_raw": "RSN", "knn": "KNN", "psm": "PSM"}
    pooled = main[(main["backbone"] == "ALL") & main["method"].isin(methods)].copy()
    pooled["label"] = pooled["method"].map(methods)

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.45))
    panels = [
        ("AUROC", axes[0], "higher is better"),
        ("FPR95", axes[1], "lower is better"),
        ("AUPR_OUT", axes[2], "higher is better"),
    ]
    for metric, ax, better in panels:
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
        label = "AUPR-OUT" if metric == "AUPR_OUT" else metric
        ax.set_ylabel(label)
        ax.set_title(f"{label} ({better})")
        ax.grid(alpha=0.22)
    axes[0].legend(frameon=False, loc="lower left")

    fig.suptitle("Performance as the number of ID classes increases", fontsize=13, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, "fig02_idsize_trend")


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
    save(fig, "fig03_shared_shape_cdf")


def main() -> None:
    configure()
    pipeline_figure()
    idsize_trend_figure()
    shared_shape_figure()


if __name__ == "__main__":
    main()
