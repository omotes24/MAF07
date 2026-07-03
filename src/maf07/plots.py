from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import resolve_path


def make_figures() -> list[Path]:
    import matplotlib.pyplot as plt
    import seaborn as sns

    out_dir = resolve_path("results/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    fair = resolve_path("results/ood/fair/summary_by_setting.csv")
    if fair.exists():
        df = pd.read_csv(fair)
        if not df.empty:
            plt.figure(figsize=(8, 5))
            sns.lineplot(data=df, x="id_size", y="AUROC", hue="method", errorbar=None, legend=False)
            plt.tight_layout()
            out = out_dir / "auroc_vs_id_size.png"
            plt.savefig(out, dpi=200)
            plt.close()
            outputs.append(out)

            pivot = df.pivot_table(index="backbone", columns="method", values="AUROC", aggfunc="mean")
            plt.figure(figsize=(12, max(4, 0.35 * len(pivot))))
            sns.heatmap(pivot, cmap="viridis")
            plt.tight_layout()
            out = out_dir / "backbone_method_auroc_heatmap.png"
            plt.savefig(out, dpi=200)
            plt.close()
            outputs.append(out)
    closed = resolve_path("results/closed/results.csv")
    if closed.exists():
        df = pd.read_csv(closed)
        if not df.empty:
            plt.figure(figsize=(8, 4))
            sns.barplot(data=df, x="method", y="top1_accuracy", errorbar=None)
            plt.xticks(rotation=30, ha="right")
            plt.tight_layout()
            out = out_dir / "closed_top1_by_method.png"
            plt.savefig(out, dpi=200)
            plt.close()
            outputs.append(out)
    return outputs

