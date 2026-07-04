from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, normalize

from .config import load_yaml, resolve_path
from .features import feature_frame_for_split
from .heads import fit_closed_head
from .jobs import append_completed_job, append_csv_rows, read_completed_jobs, read_expected_jobs
from .metrics import closed_classification_metrics, logits_to_scores, ood_metrics
from .methods.baselines_activation import (
    ash_features,
    dice_features,
    feature_energy,
    nci_score,
    react_features,
    scale_score,
)
from .methods.baselines_distance import (
    class_means,
    knn_score,
    mahalanobis_score,
    mahalanobispp_score,
    rmd_score,
)
from .methods.baselines_logit import gradnorm, kl_matching
from .methods.baselines_vlm import clip_text_energy, clip_zero_shot_msp, mcm_score, tip_adapter_score
from .methods.maf import MAFScorer, distance_variant_score, maf_fusion
from .splits import make_ood_eval_frame


def _load_split(seed: int, split_dir: str | Path = "results/splits") -> pd.DataFrame:
    path = resolve_path(split_dir) / f"splits_seed{seed}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Split file missing: {path}")
    return pd.read_csv(path)


def _pending_jobs(kind: str, protocol: str | None = None) -> pd.DataFrame:
    expected = read_expected_jobs()
    completed = read_completed_jobs()
    done = set(completed.loc[completed["status"] == "completed", "job_id"]) if not completed.empty else set()
    jobs = expected[(expected["job_kind"] == kind) & (~expected["job_id"].isin(done))]
    if protocol is not None:
        jobs = jobs[jobs["protocol"] == protocol]
    backbone_filter = _selected_backbones_from_env()
    if backbone_filter is not None:
        jobs = jobs[jobs["backbone"].astype(str).isin(backbone_filter)]
    return jobs.sort_values("job_id").reset_index(drop=True)


def _selected_backbones_from_env() -> set[str] | None:
    raw = os.environ.get("MAF07_BACKBONES") or os.environ.get("MAF07_BACKBONE_FILTER")
    if not raw:
        return None
    values = {item.strip() for chunk in raw.split(",") for item in chunk.split() if item.strip()}
    return values or None


def _job_shard(
    jobs: pd.DataFrame,
    worker_index: int | None = None,
    worker_count: int | None = None,
) -> pd.DataFrame:
    if worker_index is None:
        raw = os.environ.get("MAF07_JOB_WORKER_INDEX")
        worker_index = int(raw) if raw is not None else 0
    if worker_count is None:
        raw = os.environ.get("MAF07_JOB_WORKER_COUNT")
        worker_count = int(raw) if raw is not None else 1
    if worker_count <= 1:
        return jobs.reset_index(drop=True)
    if worker_index < 0 or worker_index >= worker_count:
        raise ValueError(f"Invalid worker shard {worker_index}/{worker_count}")
    return jobs.iloc[worker_index::worker_count].reset_index(drop=True)


def _class_prototypes(features: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    classes, means = class_means(features, labels)
    return classes, normalize(means)


def _logit_training(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    logits = fit_closed_head("linear_probe", train_x, train_y, test_x)
    train_logits = fit_closed_head("linear_probe", train_x, train_y, train_x)
    by_class = {int(c): train_logits[train_y == c] for c in sorted(np.unique(train_y))}
    return logits, by_class


def _vim_score(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> np.ndarray:
    _, means = class_means(train_x, train_y)
    logits = test_x @ normalize(means).T
    centered = train_x - train_x.mean(axis=0, keepdims=True)
    dim = max(1, min(centered.shape[1] - 1, len(np.unique(train_y)) * 2))
    pca = PCA(n_components=dim, svd_solver="randomized", random_state=0).fit(centered)
    residual = test_x - test_x.mean(axis=0, keepdims=True)
    proj = pca.inverse_transform(pca.transform(residual))
    residual_norm = np.linalg.norm(residual - proj, axis=1)
    return logits.max(axis=1) - residual_norm


def _openmax_score(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> np.ndarray:
    _, means = class_means(train_x, train_y)
    logits = test_x @ normalize(means).T
    nearest = -mahalanobis_score(train_x, train_y, test_x)
    adjusted = logits.max(axis=1) - nearest
    return adjusted


def _score_method(
    method: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    eval_x: np.ndarray,
    *,
    oracle_labels: np.ndarray | None = None,
) -> np.ndarray:
    method = method.lower()
    if method == "maf":
        fit_x = val_x if len(val_x) else train_x
        fit_y = val_y if len(val_y) else train_y
        scorer = MAFScorer().fit(fit_x, fit_y)
        return scorer.score(eval_x)
    if method in {"msp", "entropy", "energy", "maxlogit", "odin", "gen"}:
        logits, _ = _logit_training(train_x, train_y, eval_x)
        return logits_to_scores(logits, method)
    if method == "gradnorm":
        logits, _ = _logit_training(train_x, train_y, eval_x)
        return gradnorm(logits)
    if method == "kl_matching":
        logits, by_class = _logit_training(train_x, train_y, eval_x)
        return kl_matching(logits, by_class)
    if method == "openmax":
        return _openmax_score(train_x, train_y, eval_x)
    if method in {"mahalanobis", "mah_mindist"}:
        return mahalanobis_score(train_x, train_y, eval_x)
    if method == "rmd":
        return rmd_score(train_x, train_y, eval_x)
    if method == "knn":
        return knn_score(train_x, eval_x)
    if method == "mahalanobispp":
        return mahalanobispp_score(train_x, train_y, val_x, eval_x)
    if method == "vim":
        return _vim_score(train_x, train_y, eval_x)

    _, prototypes = _class_prototypes(train_x, train_y)
    if method == "react":
        return feature_energy(react_features(train_x, eval_x), prototypes)
    if method == "ashp":
        return feature_energy(ash_features(eval_x, variant="p"), prototypes)
    if method == "ashs":
        return feature_energy(ash_features(eval_x, variant="s"), prototypes)
    if method == "ashb":
        return feature_energy(ash_features(eval_x, variant="b"), prototypes)
    if method == "dice":
        return feature_energy(dice_features(train_x, eval_x), prototypes)
    if method == "scale":
        return scale_score(eval_x, prototypes)
    if method == "nci":
        return nci_score(eval_x, prototypes)
    if method == "mcm":
        return mcm_score(eval_x, prototypes)
    if method == "clip_zeroshot_msp":
        return clip_zero_shot_msp(eval_x, prototypes)
    if method == "clip_text_energy":
        return clip_text_energy(eval_x, prototypes)
    if method == "tip_adapter":
        onehot = np.eye(len(np.unique(train_y)))[train_y]
        return tip_adapter_score(eval_x, train_x, onehot, prototypes)
    raise ValueError(f"Unknown OOD method: {method}")


def run_closed_jobs(
    max_jobs: int | None = None,
    split_dir: str | Path = "results/splits",
    worker_index: int | None = None,
    worker_count: int | None = None,
) -> int:
    jobs = _job_shard(_pending_jobs("closed"), worker_index, worker_count)
    if max_jobs is not None:
        jobs = jobs.head(max_jobs)
    out_path = resolve_path("results/closed/results.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    for _, job in jobs.iterrows():
        split = _load_split(int(job["seed"]), split_dir)
        rows, features = feature_frame_for_split(split, str(job["backbone"]))
        train_mask = rows["split"] == "train"
        test_mask = rows["split"] == "test"
        enc = LabelEncoder().fit(rows.loc[train_mask, "class_name"])
        train_y = enc.transform(rows.loc[train_mask, "class_name"])
        test_y = enc.transform(rows.loc[test_mask, "class_name"])
        logits = fit_closed_head(
            str(job["method"]),
            features[train_mask.to_numpy()],
            train_y,
            features[test_mask.to_numpy()],
        )
        metrics = closed_classification_metrics(logits, test_y, [str(c) for c in enc.classes_])
        flat = {
            k: v
            for k, v in metrics.items()
            if not isinstance(v, (dict, list))
        }
        row = {**job.to_dict(), **flat}
        append_csv_rows(pd.DataFrame([row]), out_path)
        cm_path = resolve_path("results/closed/confusion_matrices") / f"{job['job_id']}.json"
        cm_path.parent.mkdir(parents=True, exist_ok=True)
        cm_path.write_text(
            json.dumps(
                {
                    "job_id": str(job["job_id"]),
                    "classes": [str(c) for c in enc.classes_],
                    "confusion_matrix": metrics["confusion_matrix"],
                    "per_class_precision": metrics["per_class_precision"],
                    "per_class_recall": metrics["per_class_recall"],
                    "per_class_f1": metrics["per_class_f1"],
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        append_completed_job(job, artifact=str(out_path))
        count += 1
    return count


def run_ood_jobs(
    protocol: str,
    max_jobs: int | None = None,
    split_dir: str | Path = "results/splits",
    worker_index: int | None = None,
    worker_count: int | None = None,
) -> int:
    jobs = _job_shard(_pending_jobs("ood", protocol), worker_index, worker_count)
    if max_jobs is not None:
        jobs = jobs.head(max_jobs)
    summary_dir = resolve_path(f"results/ood/{protocol}")
    scores_dir = summary_dir / "scores" / "jobs"
    scores_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / ("oracle_summary.csv" if protocol == "oracle" else "summary_by_setting.csv")
    count = 0
    for _, job in jobs.iterrows():
        split = _load_split(int(job["seed"]), split_dir)
        id_classes = str(job["id_set"]).split("|")
        eval_frame = make_ood_eval_frame(split, id_classes, protocol=protocol)
        rows, features = feature_frame_for_split(eval_frame, str(job["backbone"]))
        train_mask = rows["role"] == "id_train"
        val_mask = rows["role"] == "id_val"
        eval_mask = rows["role"].isin(["id_test", "ood_test"])
        enc = LabelEncoder().fit(rows.loc[train_mask, "class_name"])
        train_y = enc.transform(rows.loc[train_mask, "class_name"])
        val_y = enc.transform(rows.loc[val_mask, "class_name"]) if val_mask.any() else train_y
        scores = _score_method(
            str(job["method"]),
            features[train_mask.to_numpy()],
            train_y,
            features[val_mask.to_numpy()] if val_mask.any() else features[train_mask.to_numpy()],
            val_y,
            features[eval_mask.to_numpy()],
            oracle_labels=rows.loc[eval_mask, "ood_label"].to_numpy(),
        )
        score_rows = rows.loc[eval_mask, ["image_id", "class_name", "split", "role", "ood_label"]].copy()
        score_rows["is_id"] = (score_rows["ood_label"] == 0).astype(int)
        score_rows["score"] = scores
        for col in job.index:
            score_rows[col] = job[col]
        score_path = scores_dir / f"{job['job_id']}.parquet"
        score_rows.to_parquet(score_path, index=False)
        metrics = ood_metrics(score_rows["is_id"].to_numpy(), score_rows["score"].to_numpy())
        summary = {**job.to_dict(), **metrics}
        append_csv_rows(pd.DataFrame([summary]), summary_path)
        append_completed_job(job, artifact=str(score_path))
        count += 1
    return count


def _parse_variant(variant: str) -> dict[str, object]:
    cfg: dict[str, object] = {}
    if variant == "main":
        return cfg
    if "=" not in variant:
        return {"fusion": variant}
    key, value = variant.split("=", 1)
    if key == "alpha":
        cfg["alpha"] = float(value)
    elif key == "pca":
        cfg["pca_dim"] = int(value)
    else:
        cfg[key] = value
    return cfg


def run_ablation_jobs(
    max_jobs: int | None = None,
    split_dir: str | Path = "results/splits",
    worker_index: int | None = None,
    worker_count: int | None = None,
) -> int:
    jobs = _job_shard(_pending_jobs("ablation", "fair"), worker_index, worker_count)
    if max_jobs is not None:
        jobs = jobs.head(max_jobs)
    out_path = resolve_path("results/ablation/maf_ablation.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    for _, job in jobs.iterrows():
        split = _load_split(int(job["seed"]), split_dir)
        id_classes = str(job["id_set"]).split("|")
        eval_frame = make_ood_eval_frame(split, id_classes, protocol="fair")
        rows, features = feature_frame_for_split(eval_frame, str(job["backbone"]))
        train_mask = rows["role"] == "id_train"
        val_mask = rows["role"] == "id_val"
        eval_mask = rows["role"].isin(["id_test", "ood_test"])
        enc = LabelEncoder().fit(rows.loc[train_mask, "class_name"])
        train_y = enc.transform(rows.loc[train_mask, "class_name"])
        val_y = enc.transform(rows.loc[val_mask, "class_name"]) if val_mask.any() else train_y
        cfg = _parse_variant(str(job["variant"]))
        fit_x = features[val_mask.to_numpy()] if val_mask.any() else features[train_mask.to_numpy()]
        fit_y = val_y
        scorer = MAFScorer(
            alpha=float(cfg.get("alpha", 0.5)),
            covariance=str(cfg.get("covariance", "tied_ledoit_wolf")),
            feature_norm=str(cfg.get("feature_norm", "raw")),
            distance=str(cfg.get("distance", "mahalanobis")),
            prototype=str(cfg.get("prototype", "class_mean")),
            pca_dim=cfg.get("pca_dim"),
        ).fit(fit_x, fit_y)
        eval_x = features[eval_mask.to_numpy()]
        if "distance_score" in cfg:
            scores = distance_variant_score(scorer.distances(eval_x), str(cfg["distance_score"]))
        else:
            comp = scorer.score_components(eval_x)
            scores = maf_fusion(
                comp["s_conf"],
                comp["s_cons"],
                alpha=float(cfg.get("alpha", 0.5)),
                mode=str(cfg.get("fusion", "alpha")),
            )
        score_rows = rows.loc[eval_mask, ["class_name", "ood_label"]].copy()
        score_rows["is_id"] = (score_rows["ood_label"] == 0).astype(int)
        metrics = ood_metrics(score_rows["is_id"].to_numpy(), scores)
        result = {**job.to_dict(), **metrics}
        append_csv_rows(pd.DataFrame([result]), out_path)
        append_completed_job(job, artifact=str(out_path))
        count += 1
    return count


def extract_configured_features(
    *,
    tiers: list[str] | None = None,
    resume: bool = True,
) -> int:
    from .features import extract_feature_cache

    bcfg = load_yaml("configs/backbones.yaml")
    ecfg = load_yaml("configs/experiments.yaml")
    tiers = tiers or list(ecfg.get("default_backbone_tiers", ["tier0"]))
    names: list[str] = []
    for tier in tiers:
        names.extend(str(item["name"]) for item in bcfg.get(tier, []))
    for name in names:
        extract_feature_cache(name, resume=resume)
    return len(names)
