#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from maf07.methods.baselines_distance import knn_score
from maf07.methods.rsn import RSNDetector
from maf07.runner import _load_split
from maf07.splits import make_ood_eval_frame


CLASSES = ["cheetah", "jaguar", "leopard", "lion", "ocelot", "puma", "serval", "tiger"]
CLASS_LABELS = {
    "cheetah": "Cheetah",
    "jaguar": "Jaguar",
    "leopard": "Leopard",
    "lion": "Lion",
    "ocelot": "Ocelot",
    "puma": "Puma",
    "serval": "Serval",
    "tiger": "Tiger",
}
COLORS = {
    "cheetah": "#e3b448",
    "jaguar": "#7a4f2a",
    "leopard": "#b67f2f",
    "lion": "#d98c3a",
    "ocelot": "#8e6a3a",
    "puma": "#6f7f8f",
    "serval": "#d6b15f",
    "tiger": "#e56b2f",
    "RSN": "#1f77b4",
    "KNN": "#6c757d",
    "PSM": "#d95f02",
}
PATTERN_GROUP = {
    "jaguar": "rosette",
    "leopard": "rosette",
    "ocelot": "rosette",
    "cheetah": "spot",
    "serval": "spot",
    "puma": "plain",
    "lion": "plain/mane",
    "tiger": "stripe",
}
ID_EXAMPLE = ["cheetah", "puma", "lion", "tiger"]


def _mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 130,
            "savefig.dpi": 300,
        }
    )
    return plt


def _save(fig, out_dir: Path, name: str, *, pdf: bool = True) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / f"{name}.png"]
    fig.savefig(paths[0], bbox_inches="tight", facecolor="white")
    if pdf:
        paths.append(out_dir / f"{name}.pdf")
        fig.savefig(paths[1], bbox_inches="tight", facecolor="white")
    return paths


def _load_image(path: str | Path, size: int | tuple[int, int] = 224) -> Image.Image:
    with Image.open(path) as img:
        img = img.convert("RGB")
        if isinstance(size, int):
            return ImageOps.fit(img, (size, size), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
        return ImageOps.contain(img, size, method=Image.Resampling.LANCZOS)


def _image_quality_score(path: str | Path) -> float:
    try:
        with Image.open(path) as img:
            img = ImageOps.contain(img.convert("RGB"), (96, 96), method=Image.Resampling.BILINEAR)
            arr = np.asarray(img, dtype=np.float32) / 255.0
    except Exception:
        return -1.0
    if arr.size == 0:
        return -1.0
    brightness = float(arr.mean())
    contrast = float(arr.std())
    mx = arr.max(axis=2)
    mn = arr.min(axis=2)
    saturation = float(np.mean((mx - mn) / np.maximum(mx, 1e-3)))
    penalty = 0.0
    if brightness < 0.18:
        penalty += (0.18 - brightness) * 2.5
    if brightness > 0.86:
        penalty += (brightness - 0.86) * 2.0
    if saturation < 0.08:
        penalty += (0.08 - saturation) * 3.0
    return 0.50 * saturation + 0.35 * contrast + 0.15 * (1.0 - abs(brightness - 0.50)) - penalty


def _sample_rows(manifest: pd.DataFrame, split: pd.DataFrame | None = None) -> dict[str, pd.Series]:
    source = split if split is not None else manifest
    rows: dict[str, pd.Series] = {}
    for cls in CLASSES:
        sub = source[(source["class_name"] == cls) & source["path"].map(lambda p: Path(str(p)).exists())]
        if sub.empty:
            sub = manifest[(manifest["class_name"] == cls) & manifest["path"].map(lambda p: Path(str(p)).exists())]
        if sub.empty:
            continue
        sub = sub.sort_values("image_id").reset_index(drop=True)
        if len(sub) > 360:
            take = np.linspace(0, len(sub) - 1, 360, dtype=int)
            candidates = sub.iloc[take].copy()
        else:
            candidates = sub.copy()
        scores = candidates["path"].map(_image_quality_score).to_numpy()
        rows[cls] = candidates.iloc[int(np.argmax(scores))]
    return rows


def _feature_cache(backbone: str) -> tuple[pd.DataFrame, np.ndarray]:
    meta = pd.read_csv(f"results/features/{backbone}.metadata.csv")
    features = np.load(f"results/features/{backbone}.npz", allow_pickle=False)["features"].astype(np.float32)
    return meta, features


def _load_exclusions(path: str | Path | None) -> set[str]:
    if path is None:
        return set()
    p = Path(path)
    if not p.exists():
        return set()
    out: set[str] = set()
    for line in p.read_text(encoding="utf-8").splitlines():
        item = line.strip()
        if not item or item.startswith("#"):
            continue
        out.add(item)
    return out


def _apply_exclusions(df: pd.DataFrame, excluded_rel_paths: set[str]) -> pd.DataFrame:
    if not excluded_rel_paths or "rel_path" not in df:
        return df
    return df[~df["rel_path"].astype(str).isin(excluded_rel_paths)].reset_index(drop=True)


def _feature_rows(split_df: pd.DataFrame, meta: pd.DataFrame, features: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    index = {str(image_id): i for i, image_id in enumerate(meta["image_id"].astype(str))}
    ids = split_df["image_id"].astype(str).tolist()
    take = np.asarray([index[image_id] for image_id in ids], dtype=int)
    return split_df.reset_index(drop=True), features[take]


def figure_rsn_concept(out_dir: Path, manifest: pd.DataFrame, split: pd.DataFrame) -> list[Path]:
    plt = _mpl()
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    samples = _sample_rows(manifest, split)
    sample = samples["jaguar"] if "jaguar" in samples else samples[CLASSES[0]]
    img = _load_image(sample["path"], 210)
    fig = plt.figure(figsize=(14, 7))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()

    ax.imshow(img, extent=(0.03, 0.17, 0.55, 0.83), zorder=1)
    ax.text(0.10, 0.50, "Input image", ha="center", va="top", fontsize=11, weight="bold")

    boxes = [
        (0.23, 0.56, 0.13, 0.20, "DINOv2\nfeature z", "#d9edf7"),
        (0.42, 0.62, 0.18, 0.18, "Class-wise\nfeature banks", "#f6e8c3"),
        (0.42, 0.34, 0.18, 0.18, "Per-class scale\nsigma_c = std(z_c)", "#f6e8c3"),
        (0.66, 0.56, 0.15, 0.20, "Scaled residuals\n(z - z_i^c) / sigma_c", "#dcecc9"),
        (0.66, 0.30, 0.15, 0.18, "Huber aggregation\nsum psi_delta", "#dcecc9"),
        (0.86, 0.48, 0.10, 0.18, "min over\nclasses", "#ead5dc"),
    ]
    for x, y, w, h, text, color in boxes:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.012,rounding_size=0.018",
                linewidth=1.2,
                edgecolor="#333333",
                facecolor=color,
            )
        )
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=11, weight="bold")

    def arrow(a, b, rad=0.0):
        ax.add_patch(
            FancyArrowPatch(
                a,
                b,
                arrowstyle="-|>",
                mutation_scale=16,
                linewidth=1.6,
                color="#333333",
                connectionstyle=f"arc3,rad={rad}",
            )
        )

    arrow((0.17, 0.69), (0.23, 0.66))
    arrow((0.36, 0.66), (0.42, 0.70))
    arrow((0.36, 0.62), (0.42, 0.43), -0.15)
    arrow((0.60, 0.70), (0.66, 0.66))
    arrow((0.60, 0.43), (0.66, 0.61), 0.15)
    arrow((0.735, 0.56), (0.735, 0.48))
    arrow((0.81, 0.39), (0.86, 0.54))
    ax.text(
        0.92,
        0.36,
        "OOD score\n"
        "o(x) = min_c d_c^RSN(z)\n"
        "ID score = -o(x)",
        ha="center",
        va="center",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f2f2f2", edgecolor="#333333"),
    )

    # Small feature bank glyphs.
    rng = np.random.default_rng(3)
    for j, cls in enumerate(["cheetah", "puma", "lion", "tiger"]):
        cx = 0.445 + (j % 2) * 0.08
        cy = 0.68 - (j // 2) * 0.07
        pts = rng.normal(size=(18, 2)) * 0.012 + [cx, cy]
        ax.scatter(pts[:, 0], pts[:, 1], s=13, color=COLORS[cls], edgecolor="white", linewidth=0.4, zorder=5)
        ax.text(cx, cy - 0.035, cls, ha="center", va="center", fontsize=7)

    ax.text(0.5, 0.92, "Robust Scale-Normalized kNN (RSN)", ha="center", fontsize=17, weight="bold")
    ax.text(
        0.5,
        0.10,
        "Class-conditional diagonal scaling makes the distance comparable across species; "
        "Huber aggregation suppresses single-coordinate outliers.",
        ha="center",
        fontsize=11,
    )
    return _save(fig, out_dir, "01_rsn_concept")


def figure_ultra_near_ood(out_dir: Path, manifest: pd.DataFrame, split: pd.DataFrame) -> list[Path]:
    plt = _mpl()
    samples = _sample_rows(manifest, split)
    fig, axes = plt.subplots(2, 4, figsize=(13, 7))
    ood_near = {"jaguar", "leopard", "ocelot"}
    for ax, cls in zip(axes.ravel(), CLASSES, strict=True):
        ax.imshow(_load_image(samples[cls]["path"], 256))
        role = "ID" if cls in ID_EXAMPLE else "OOD"
        color = "#1b9e77" if role == "ID" else "#d95f02"
        ax.set_title(f"{CLASS_LABELS[cls]}  ({role})", color=color, fontsize=12, weight="bold")
        ax.text(
            0.02,
            0.05,
            PATTERN_GROUP[cls],
            transform=ax.transAxes,
            color="white",
            fontsize=9,
            bbox=dict(facecolor="black", alpha=0.65, pad=3),
        )
        if cls in ood_near:
            ax.text(
                0.98,
                0.05,
                "near OOD\nrosette",
                transform=ax.transAxes,
                ha="right",
                color="white",
                fontsize=9,
                bbox=dict(facecolor="#b2182b", alpha=0.8, pad=3),
            )
        ax.set_axis_off()
    fig.suptitle("Ultra-Near-OOD split example: visual relatives inside the same wild-cat family", fontsize=15, weight="bold")
    fig.text(
        0.5,
        0.03,
        "Example ID set: cheetah, puma, lion, tiger. "
        "Jaguar/leopard/ocelot share rosette-like texture cues, making them near-OOD rather than open-world outliers.",
        ha="center",
        fontsize=11,
    )
    return _save(fig, out_dir, "02_ultra_near_ood_setup")


def figure_feature_space(out_dir: Path, split: pd.DataFrame, backbone: str) -> list[Path]:
    plt = _mpl()
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler

    meta, features = _feature_cache(backbone)
    rng = np.random.default_rng(7)
    picks: list[int] = []
    for cls in CLASSES:
        idx = np.flatnonzero(meta["class_name"].to_numpy() == cls)
        idx = rng.choice(idx, size=min(320, len(idx)), replace=False)
        picks.extend(idx.tolist())
    picks = np.asarray(picks, dtype=int)
    x = features[picks]
    labels = meta.iloc[picks]["class_name"].to_numpy()
    x = StandardScaler(with_mean=True, with_std=True).fit_transform(x)
    x50 = PCA(n_components=min(50, x.shape[1]), random_state=7).fit_transform(x)
    emb = TSNE(n_components=2, init="pca", learning_rate="auto", perplexity=35, random_state=7).fit_transform(x50)

    fig, ax = plt.subplots(figsize=(9, 7))
    for cls in CLASSES:
        m = labels == cls
        ax.scatter(emb[m, 0], emb[m, 1], s=10, alpha=0.65, color=COLORS[cls], label=CLASS_LABELS[cls])
    for group, text, color in [
        (["jaguar", "leopard", "ocelot"], "rosette overlap", "#b2182b"),
        (["lion", "tiger"], "more separated", "#2166ac"),
    ]:
        pts = emb[np.isin(labels, group)]
        center = pts.mean(axis=0)
        ax.text(
            center[0],
            center[1],
            text,
            ha="center",
            va="center",
            fontsize=11,
            weight="bold",
            color=color,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor=color, alpha=0.85),
        )
    ax.set_title(f"DINOv2 feature space ({backbone}, t-SNE after PCA)", fontsize=14, weight="bold")
    ax.set_xlabel("embedding dim 1")
    ax.set_ylabel("embedding dim 2")
    ax.legend(ncol=4, fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.08))
    return _save(fig, out_dir, "03_dino_feature_space_tsne")


def _summary_ci(df: pd.DataFrame, method_name: str, metric: str) -> pd.DataFrame:
    out = (
        df.groupby("id_size")[metric]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "mean", "std": "std", "count": "n"})
    )
    out["method"] = method_name
    out["ci"] = 1.96 * out["std"].fillna(0.0) / np.sqrt(out["n"].clip(lower=1))
    return out


def figure_metrics_vs_m(out_dir: Path) -> list[Path]:
    plt = _mpl()
    rsn = pd.read_csv("results/quick/diagcard_full_huber_raw_results.csv").drop_duplicates("job_id", keep="last")
    rsn = rsn[rsn["protocol"] == "fair"].copy()
    rsn["method_plot"] = "RSN"
    psm = pd.read_csv("results/quick/psm_full_results.csv").drop_duplicates("job_id", keep="last")
    psm = psm[psm["protocol"] == "fair"].copy()
    psm["method_plot"] = "PSM"
    knn = pd.read_csv("results/ood/fair/summary_by_setting.csv").drop_duplicates("job_id", keep="last")
    knn = knn[knn["method"] == "knn"].copy()
    knn["method_plot"] = "KNN"
    data = pd.concat([rsn, knn, psm], ignore_index=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharex=True)
    for ax, metric, ylabel in [(axes[0], "AUROC", "AUROC (higher is better)"), (axes[1], "FPR95", "FPR95 (lower is better)")]:
        for method in ["RSN", "KNN", "PSM"]:
            sub = _summary_ci(data[data["method_plot"] == method], method, metric)
            x = sub["id_size"].to_numpy(dtype=float)
            y = sub["mean"].to_numpy(dtype=float)
            ci = sub["ci"].to_numpy(dtype=float)
            ax.plot(x, y, marker="o", linewidth=2.2, color=COLORS[method], label=method)
            ax.fill_between(x, y - ci, y + ci, color=COLORS[method], alpha=0.16, linewidth=0)
        ax.set_xlabel("ID class size m")
        ax.set_ylabel(ylabel)
        ax.set_xticks([2, 3, 4, 5, 6, 7])
        ax.grid(True, alpha=0.25)
    axes[0].legend(frameon=False)
    fig.suptitle("OOD performance vs ID class size", fontsize=14, weight="bold")
    return _save(fig, out_dir, "04_metrics_vs_id_size")


def figure_per_ood_bar(out_dir: Path) -> list[Path]:
    plt = _mpl()
    df = pd.read_csv("results/analysis/per_ood_class_quick_fair.csv")
    rsn = df[df["method"] == "rsn"].copy()
    agg = (
        rsn.groupby("ood_class")
        .agg(AUROC=("AUROC", "mean"), FPR95=("FPR95", "mean"), AUPR_OUT=("AUPR_OUT", "mean"), n=("AUROC", "size"))
        .reset_index()
    )
    order = agg.sort_values("AUROC")["ood_class"].tolist()
    agg = agg.set_index("ood_class").loc[order].reset_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    hatches = {"rosette": "///", "spot": "...", "plain": "", "plain/mane": "", "stripe": "xx"}
    bars = ax.bar(
        np.arange(len(agg)),
        agg["AUROC"],
        color=[COLORS[c] for c in agg["ood_class"]],
        edgecolor="#333333",
        linewidth=0.8,
    )
    for bar, cls in zip(bars, agg["ood_class"], strict=True):
        bar.set_hatch(hatches[PATTERN_GROUP[cls]])
    ax.axvline(2.5, color="#333333", linestyle="--", alpha=0.55)
    ax.text(1.0, 0.97, "rosette near-OOD", ha="center", va="top", transform=ax.get_xaxis_transform(), fontsize=10)
    ax.text(4.7, 0.97, "larger semantic/texture gap", ha="center", va="top", transform=ax.get_xaxis_transform(), fontsize=10)
    ax.set_xticks(np.arange(len(agg)))
    ax.set_xticklabels([CLASS_LABELS[c] for c in agg["ood_class"]], rotation=25, ha="right")
    ax.set_ylim(0.65, 1.0)
    ax.set_ylabel("RSN AUROC")
    ax.set_title("Per-OOD-class AUROC: rosette species are the hardest cases", fontsize=14, weight="bold")
    for i, row in agg.iterrows():
        ax.text(i, row["AUROC"] + 0.008, f"{row['AUROC']:.3f}", ha="center", fontsize=8)
    return _save(fig, out_dir, "05_per_ood_class_auroc")


@dataclass
class ScoreContext:
    rows: pd.DataFrame
    train_rows: pd.DataFrame
    train_x: np.ndarray
    train_y: np.ndarray
    val_x: np.ndarray
    val_y: np.ndarray
    eval_x: np.ndarray
    rsn_scores: np.ndarray
    knn_scores: np.ndarray
    rsn_threshold: float
    knn_threshold: float


def _score_context(backbone: str = "dinov2_vitb14", seed: int = 0) -> ScoreContext:
    from sklearn.preprocessing import LabelEncoder

    meta, features = _feature_cache(backbone)
    split = _load_split(seed)
    frame = make_ood_eval_frame(split, ID_EXAMPLE, protocol="fair")
    rows, feats = _feature_rows(frame, meta, features)
    train_mask = rows["role"].eq("id_train").to_numpy()
    val_mask = rows["role"].eq("id_val").to_numpy()
    eval_mask = rows["role"].isin(["id_test", "ood_test"]).to_numpy()
    enc = LabelEncoder().fit(rows.loc[train_mask, "class_name"])
    train_y = enc.transform(rows.loc[train_mask, "class_name"])
    val_y = enc.transform(rows.loc[val_mask, "class_name"])
    train_x = feats[train_mask]
    val_x = feats[val_mask]
    eval_rows = rows.loc[eval_mask].reset_index(drop=True)
    eval_x_all = feats[eval_mask]

    rng = np.random.default_rng(11)
    id_idx = np.flatnonzero(eval_rows["ood_label"].to_numpy() == 0)
    ood_idx = np.flatnonzero(eval_rows["ood_label"].to_numpy() == 1)
    take = np.concatenate(
        [
            rng.choice(id_idx, size=min(4500, len(id_idx)), replace=False),
            rng.choice(ood_idx, size=min(4500, len(ood_idx)), replace=False),
        ]
    )
    take.sort()
    eval_rows = eval_rows.iloc[take].reset_index(drop=True)
    eval_x = eval_x_all[take]

    detector = RSNDetector(k=150, delta=1.345, device="cuda", score_batch=96).fit(
        train_x, train_y, z_cal=val_x, y_cal=val_y
    )
    rsn_scores = detector.id_scores(eval_x)
    knn_scores = knn_score(train_x, eval_x)
    id_mask = eval_rows["ood_label"].to_numpy() == 0
    rsn_threshold = float(np.quantile(rsn_scores[id_mask], 0.05, method="lower"))
    knn_threshold = float(np.quantile(knn_scores[id_mask], 0.05, method="lower"))
    train_rows = rows.loc[train_mask].reset_index(drop=True)
    return ScoreContext(
        rows=eval_rows,
        train_rows=train_rows,
        train_x=train_x,
        train_y=train_y,
        val_x=val_x,
        val_y=val_y,
        eval_x=eval_x,
        rsn_scores=rsn_scores,
        knn_scores=knn_scores,
        rsn_threshold=rsn_threshold,
        knn_threshold=knn_threshold,
    )


def figure_score_distributions(out_dir: Path, ctx: ScoreContext) -> list[Path]:
    plt = _mpl()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, scores, threshold, title, color in [
        (axes[0], ctx.knn_scores, ctx.knn_threshold, "KNN ID score", COLORS["KNN"]),
        (axes[1], ctx.rsn_scores, ctx.rsn_threshold, "RSN ID score", COLORS["RSN"]),
    ]:
        id_scores = scores[ctx.rows["ood_label"].to_numpy() == 0]
        ood_scores = scores[ctx.rows["ood_label"].to_numpy() == 1]
        bins = np.linspace(np.percentile(scores, 1), np.percentile(scores, 99), 60)
        ax.hist(id_scores, bins=bins, density=True, alpha=0.55, color="#2ca25f", label="ID test")
        ax.hist(ood_scores, bins=bins, density=True, alpha=0.55, color=color, label="OOD test")
        ax.axvline(threshold, color="#111111", linestyle="--", linewidth=1.5, label="95% ID TPR threshold")
        ax.set_title(title, fontsize=12, weight="bold")
        ax.set_xlabel("ID-like score")
        ax.set_ylabel("density")
        ax.grid(True, alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Score distributions: RSN shifts near-OOD farther below the ID threshold", fontsize=14, weight="bold")
    return _save(fig, out_dir, "06_score_distributions_rsn_vs_knn")


def _cosine_neighbors(train_x: np.ndarray, query: np.ndarray, n: int = 5) -> np.ndarray:
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import normalize

    nn = NearestNeighbors(n_neighbors=n, metric="cosine").fit(normalize(train_x))
    _, idx = nn.kneighbors(normalize(query.reshape(1, -1)))
    return idx[0]


def _rsn_neighbors(train_rows: pd.DataFrame, train_x: np.ndarray, query: np.ndarray, n: int = 5) -> tuple[str, np.ndarray, float]:
    best_cls = ""
    best_score = math.inf
    best_idx = np.arange(n)
    for cls in ID_EXAMPLE:
        mask = train_rows["class_name"].to_numpy() == cls
        bank = train_x[mask].astype(np.float32)
        idx_global = np.flatnonzero(mask)
        std = bank.std(axis=0, ddof=1)
        std = np.maximum(std, 1e-3)
        qs = query.astype(np.float32) / std
        bs = bank / std
        d2 = np.sum((bs - qs[None, :]) ** 2, axis=1)
        k_top = min(150, len(d2))
        top = np.argpartition(d2, k_top - 1)[:k_top]
        delta = 1.345
        diffs = np.abs(bs[top] - qs[None, :])
        psi = np.where(diffs <= delta, diffs * diffs, 2 * delta * diffs - delta * delta).sum(axis=1)
        score = float(np.mean(psi))
        if score < best_score:
            best_score = score
            best_cls = cls
            best_idx = idx_global[np.argsort(d2)[:n]]
    return best_cls, best_idx, best_score


def _draw_image_row(axs, paths: list[str], titles: list[str]) -> None:
    for ax, path, title in zip(axs, paths, titles, strict=True):
        ax.imshow(_load_image(path, 180))
        ax.set_title(title, fontsize=8)
        ax.set_axis_off()


def figure_nearest_neighbor_panel(out_dir: Path, ctx: ScoreContext) -> list[Path]:
    plt = _mpl()
    is_ood = ctx.rows["ood_label"].to_numpy() == 1
    near = ctx.rows["class_name"].isin(["jaguar", "leopard", "ocelot"]).to_numpy()
    knn_accept = ctx.knn_scores >= ctx.knn_threshold
    rsn_reject = ctx.rsn_scores < ctx.rsn_threshold
    candidates = np.flatnonzero(is_ood & near & knn_accept & rsn_reject)
    if len(candidates) == 0:
        candidates = np.flatnonzero(is_ood & near)
    score_gap = (ctx.knn_scores[candidates] - ctx.knn_threshold) + (ctx.rsn_threshold - ctx.rsn_scores[candidates])
    order = candidates[np.argsort(-score_gap)]
    qidx = int(order[0])
    for idx in order[:80]:
        if _image_quality_score(ctx.rows.iloc[int(idx)]["path"]) > 0.08:
            qidx = int(idx)
            break
    qrow = ctx.rows.iloc[qidx]
    q = ctx.eval_x[qidx]
    knn_idx = _cosine_neighbors(ctx.train_x, q, 5)
    rsn_cls, rsn_idx, rsn_distance = _rsn_neighbors(ctx.train_rows, ctx.train_x, q, 5)

    fig, axes = plt.subplots(3, 6, figsize=(13, 7))
    for ax in axes.ravel():
        ax.set_axis_off()
    axes[0, 0].imshow(_load_image(qrow["path"], 220))
    axes[0, 0].set_title(
        f"OOD query: {CLASS_LABELS[qrow['class_name']]}\nKNN accepts, RSN rejects",
        fontsize=9,
        weight="bold",
    )
    axes[0, 1].text(
        0.05,
        0.55,
        f"KNN score: {ctx.knn_scores[qidx]:.3f}\nthreshold: {ctx.knn_threshold:.3f}\n\n"
        f"RSN score: {ctx.rsn_scores[qidx]:.1f}\nthreshold: {ctx.rsn_threshold:.1f}\n\n"
        f"RSN nearest class: {CLASS_LABELS[rsn_cls]}\nHuber distance: {rsn_distance:.1f}",
        va="center",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f2f2f2", edgecolor="#555555"),
    )
    _draw_image_row(
        axes[1, :5],
        ctx.train_rows.iloc[knn_idx]["path"].tolist(),
        [f"KNN #{i+1}\n{CLASS_LABELS[c]}" for i, c in enumerate(ctx.train_rows.iloc[knn_idx]["class_name"])],
    )
    axes[1, 5].text(0.5, 0.5, "cosine\nneighbors", ha="center", va="center", fontsize=12, weight="bold")
    _draw_image_row(
        axes[2, :5],
        ctx.train_rows.iloc[rsn_idx]["path"].tolist(),
        [f"RSN #{i+1}\n{CLASS_LABELS[c]}" for i, c in enumerate(ctx.train_rows.iloc[rsn_idx]["class_name"])],
    )
    axes[2, 5].text(0.5, 0.5, "scale-normalized\nneighbors", ha="center", va="center", fontsize=12, weight="bold")
    fig.suptitle("Qualitative nearest-neighbor example", fontsize=14, weight="bold")
    return _save(fig, out_dir, "07_nearest_neighbor_qualitative")


def figure_failure_cases(out_dir: Path, ctx: ScoreContext) -> list[Path]:
    plt = _mpl()
    is_id = ctx.rows["ood_label"].to_numpy() == 0
    is_ood = ~is_id
    near = ctx.rows["class_name"].isin(["jaguar", "leopard", "ocelot"]).to_numpy()
    false_accept = np.flatnonzero(is_ood & near & (ctx.rsn_scores >= ctx.rsn_threshold))
    false_reject = np.flatnonzero(is_id & (ctx.rsn_scores < ctx.rsn_threshold))
    if len(false_accept) < 4:
        false_accept = np.flatnonzero(is_ood & near)
    if len(false_reject) < 4:
        false_reject = np.flatnonzero(is_id)

    def choose(candidates: np.ndarray, values: np.ndarray, reverse: bool) -> np.ndarray:
        ordered = candidates[np.argsort(values[candidates])]
        if reverse:
            ordered = ordered[::-1]
        good = [int(i) for i in ordered[:160] if _image_quality_score(ctx.rows.iloc[int(i)]["path"]) > 0.08]
        if len(good) < 4:
            good = [int(i) for i in ordered[:4]]
        return np.asarray(good[:4], dtype=int)

    false_accept = choose(false_accept, ctx.rsn_scores, reverse=True)
    false_reject = choose(false_reject, ctx.rsn_scores, reverse=False)

    fig, axes = plt.subplots(2, 4, figsize=(11, 6))
    for ax, idx in zip(axes[0], false_accept):
        row = ctx.rows.iloc[int(idx)]
        ax.imshow(_load_image(row["path"], 220))
        ax.set_title(f"OOD accepted\n{CLASS_LABELS[row['class_name']]}  s={ctx.rsn_scores[idx]:.1f}", fontsize=8)
        ax.set_axis_off()
    for ax, idx in zip(axes[1], false_reject):
        row = ctx.rows.iloc[int(idx)]
        ax.imshow(_load_image(row["path"], 220))
        ax.set_title(f"ID rejected\n{CLASS_LABELS[row['class_name']]}  s={ctx.rsn_scores[idx]:.1f}", fontsize=8)
        ax.set_axis_off()
    fig.suptitle("RSN failure cases at the FPR95 operating point", fontsize=14, weight="bold")
    fig.text(0.5, 0.03, "Top: near-OOD rosette cats accepted as ID. Bottom: difficult ID images rejected.", ha="center")
    return _save(fig, out_dir, "08_failure_cases")


def figure_ablation_heatmap(out_dir: Path) -> list[Path]:
    plt = _mpl()
    df = pd.read_csv("results/quick/diagcard_full_2x2_summary_by_id_size.csv")
    df = df[(df["scope"] == "fair") & (df["backbone"] == "ALL")].copy()
    order = ["diag_calib", "diag_huber_calib", "diag_raw", "diag_huber_raw"]
    labels = ["scaled + calib", "scaled + Huber + calib", "scaled raw", "scaled + Huber raw (RSN)"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, metric, title, cmap in [
        (axes[0], "AUROC_mean", "AUROC", "YlGnBu"),
        (axes[1], "FPR95_mean", "FPR95", "YlOrRd_r"),
    ]:
        pivot = df.pivot(index="variant", columns="id_size", values=metric).loc[order]
        im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap=cmap)
        ax.set_yticks(np.arange(len(order)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels([str(c) for c in pivot.columns])
        ax.set_xlabel("ID class size m")
        ax.set_title(title, weight="bold")
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                ax.text(j, i, f"{pivot.iloc[i, j]:.3f}", ha="center", va="center", fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Ablation heatmap (completed full sweep: k=150, q=3, delta=1.345)", fontsize=14, weight="bold")
    fig.text(
        0.5,
        0.01,
        "This completed grid isolates standardization, Huber aggregation, and empirical calibration. "
        "A separate k/q/delta sensitivity sweep is still needed for hyperparameter-dependence claims.",
        ha="center",
        fontsize=9,
    )
    return _save(fig, out_dir, "09_ablation_heatmap")


def figure_dataset_flow(out_dir: Path, manifest: pd.DataFrame, split: pd.DataFrame) -> list[Path]:
    plt = _mpl()
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    counts = manifest.groupby("class_name").size().reindex(CLASSES).astype(int)
    split_counts = split.groupby("split").size().reindex(["train", "val", "test"]).astype(int)
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.set_axis_off()

    boxes = [
        (0.03, 0.58, 0.16, 0.20, "Image collection\nweb search + filtering", "#e8f4f8"),
        (0.25, 0.58, 0.16, 0.20, "Deduplication\nSHA/pixel groups", "#f5ead7"),
        (0.47, 0.58, 0.16, 0.20, "Final CAT dataset\n8 wild-cat classes", "#e8f0d9"),
        (0.69, 0.58, 0.16, 0.20, "Train/val/test split\nduplicate-safe", "#eadff0"),
        (0.69, 0.22, 0.16, 0.20, "Ultra-Near-OOD\nID/OOD set generation", "#f2e0e0"),
    ]
    for x, y, w, h, text, color in boxes:
        ax.add_patch(
            FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.015", facecolor=color, edgecolor="#333333")
        )
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=11, weight="bold")
    for a, b in [((0.19, 0.68), (0.25, 0.68)), ((0.41, 0.68), (0.47, 0.68)), ((0.63, 0.68), (0.69, 0.68)), ((0.77, 0.58), (0.77, 0.42))]:
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=16, linewidth=1.5, color="#333333"))

    # Class distribution inset.
    inset = fig.add_axes([0.07, 0.08, 0.45, 0.32])
    inset.bar(np.arange(len(counts)), counts.to_numpy(), color=[COLORS[c] for c in counts.index], edgecolor="#333333")
    inset.set_xticks(np.arange(len(counts)))
    inset.set_xticklabels([CLASS_LABELS[c] for c in counts.index], rotation=30, ha="right", fontsize=8)
    inset.set_ylabel("images")
    inset.set_title("Class distribution after deduplication", fontsize=10, weight="bold")
    inset.grid(axis="y", alpha=0.25)

    text = "\n".join([f"{k}: {int(v):,}" for k, v in split_counts.to_dict().items()])
    ax.text(
        0.90,
        0.67,
        f"Seed-0 split\n{text}",
        ha="center",
        va="center",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f8f8f8", edgecolor="#333333"),
    )
    ax.text(
        0.90,
        0.31,
        "For each m:\nchoose ID classes\nremaining species are OOD\nrepeat over all combinations\nand seeds 0,1,2",
        ha="center",
        va="center",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f8f8f8", edgecolor="#333333"),
    )
    ax.text(0.5, 0.92, "CAT dataset construction and evaluation split flow", ha="center", fontsize=15, weight="bold")
    return _save(fig, out_dir, "10_dataset_construction_flow")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results/paper_figures")
    parser.add_argument("--backbone", default="dinov2_vitb14")
    parser.add_argument("--exclude-list", default="configs/excluded_images.txt")
    parser.add_argument("--skip-heavy", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    excluded = _load_exclusions(args.exclude_list)
    manifest = _apply_exclusions(pd.read_csv("results/manifest.csv"), excluded)
    split = _apply_exclusions(pd.read_csv("results/splits/splits_seed0.csv"), excluded)
    outputs: list[Path] = []

    outputs += figure_rsn_concept(out_dir, manifest, split)
    outputs += figure_ultra_near_ood(out_dir, manifest, split)
    outputs += figure_metrics_vs_m(out_dir)
    outputs += figure_per_ood_bar(out_dir)
    outputs += figure_ablation_heatmap(out_dir)
    outputs += figure_dataset_flow(out_dir, manifest, split)

    if not args.skip_heavy:
        outputs += figure_feature_space(out_dir, split, args.backbone)
        ctx = _score_context(args.backbone)
        outputs += figure_score_distributions(out_dir, ctx)
        outputs += figure_nearest_neighbor_panel(out_dir, ctx)
        outputs += figure_failure_cases(out_dir, ctx)

    manifest_path = out_dir / "figure_manifest.json"
    manifest_path.write_text(json.dumps([str(p) for p in outputs], indent=2), encoding="utf-8")
    print(manifest_path)
    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
