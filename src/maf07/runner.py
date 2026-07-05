from __future__ import annotations

import json
import os
from functools import lru_cache
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
from .methods.lar import lar_from_env
from .methods.lantern import LANTERNDetector
from .methods.maf import MAFScorer, distance_variant_score, maf_fusion
from .splits import make_ood_eval_frame


@lru_cache(maxsize=None)
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
    method_filter = _selected_methods_from_env()
    if method_filter is not None and "method" in jobs:
        jobs = jobs[jobs["method"].astype(str).isin(method_filter)]
    preferred_order = [
        "backbone",
        "seed",
        "id_size",
        "id_set",
        "protocol",
        "method",
        "variant",
        "job_id",
    ]
    sort_cols = [col for col in preferred_order if col in jobs.columns]
    return jobs.sort_values(sort_cols).reset_index(drop=True)


def _selected_backbones_from_env() -> set[str] | None:
    raw = os.environ.get("MAF07_BACKBONES") or os.environ.get("MAF07_BACKBONE_FILTER")
    if not raw:
        return None
    values = {item.strip() for chunk in raw.split(",") for item in chunk.split() if item.strip()}
    return values or None


def _selected_methods_from_env() -> set[str] | None:
    raw = os.environ.get("MAF07_METHODS") or os.environ.get("MAF07_METHOD_FILTER")
    if not raw:
        return None
    values = {item.strip().lower() for chunk in raw.split(",") for item in chunk.split() if item.strip()}
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
    start = len(jobs) * worker_index // worker_count
    end = len(jobs) * (worker_index + 1) // worker_count
    return jobs.iloc[start:end].reset_index(drop=True)


def _class_prototypes(features: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    classes, means = class_means(features, labels)
    return classes, normalize(means)


def _logit_training(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    backend = os.environ.get("MAF07_LOGIT_BACKEND", "torch_ridge").lower()
    if backend in {"torch", "torch_ridge", "cuda", "gpu"}:
        try:
            return _torch_ridge_logits(train_x, train_y, test_x)
        except Exception:
            if os.environ.get("MAF07_STRICT_TORCH_LOGITS", "0") == "1":
                raise
    logits = fit_closed_head("linear_probe", train_x, train_y, test_x)
    train_logits = fit_closed_head("linear_probe", train_x, train_y, train_x)
    by_class = {int(c): train_logits[train_y == c] for c in sorted(np.unique(train_y))}
    return logits, by_class


def _split_logits(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    eval_x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_train = len(train_x)
    n_val = len(val_x)
    all_x = np.vstack([train_x, val_x, eval_x])
    logits, _ = _logit_training(train_x, train_y, all_x)
    return logits[:n_train], logits[n_train : n_train + n_val], logits[n_train + n_val :]


def _torch_ridge_logits(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    import torch

    device_name = os.environ.get("MAF07_TORCH_DEVICE")
    if device_name is None:
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    dtype = torch.float32
    x = torch.as_tensor(np.asarray(train_x, dtype=np.float32), device=device, dtype=dtype)
    xe = torch.as_tensor(np.asarray(test_x, dtype=np.float32), device=device, dtype=dtype)
    y_np = np.asarray(train_y, dtype=int)
    classes = np.asarray(sorted(np.unique(y_np)))
    class_to_col = {int(cls): i for i, cls in enumerate(classes)}
    y_cols = torch.as_tensor([class_to_col[int(v)] for v in y_np], device=device, dtype=torch.long)
    y = torch.nn.functional.one_hot(y_cols, num_classes=len(classes)).to(dtype)
    ones = torch.ones((x.shape[0], 1), device=device, dtype=dtype)
    x_aug = torch.cat([x, ones], dim=1)
    xe_aug = torch.cat([xe, torch.ones((xe.shape[0], 1), device=device, dtype=dtype)], dim=1)
    reg = float(os.environ.get("MAF07_RIDGE_LAMBDA", "1e-3"))
    gram = x_aug.T @ x_aug
    eye = torch.eye(gram.shape[0], device=device, dtype=dtype)
    eye[-1, -1] = 0.0
    rhs = x_aug.T @ y
    weights = torch.linalg.solve(gram + reg * eye, rhs)
    eval_logits = xe_aug @ weights
    train_logits = x_aug @ weights
    eval_np = eval_logits.detach().cpu().numpy()
    train_np = train_logits.detach().cpu().numpy()
    by_class = {int(c): train_np[y_np == int(c)] for c in classes}
    return eval_np, by_class


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


def _lantern_score(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    eval_x: np.ndarray,
    *,
    train_logits: np.ndarray | None = None,
    val_logits: np.ndarray | None = None,
    eval_logits: np.ndarray | None = None,
) -> np.ndarray:
    if train_logits is None or val_logits is None or eval_logits is None:
        train_logits, val_logits, eval_logits = _split_logits(train_x, train_y, val_x, eval_x)
    detector = LANTERNDetector().fit(
        train_x,
        train_y,
        logits_train=train_logits,
        z_cal=val_x,
        y_cal=val_y,
        logits_cal=val_logits,
    )
    return detector.id_scores(eval_x, eval_logits)


def _completed_rows(jobs: list[pd.Series], artifacts: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for job, artifact in zip(jobs, artifacts, strict=True):
        row = dict(job)
        row["status"] = "completed"
        row["artifact"] = artifact
        rows.append(row)
    return pd.DataFrame(rows)


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
    if method == "lar":
        return lar_from_env().fit(train_x).id_scores(eval_x)
    if method == "lantern":
        train_logits, val_logits, eval_logits = _split_logits(train_x, train_y, val_x, eval_x)
        return _lantern_score(
            train_x,
            train_y,
            val_x,
            val_y,
            eval_x,
            train_logits=train_logits,
            val_logits=val_logits,
            eval_logits=eval_logits,
        )
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


def _score_ood_group_method(
    method: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    eval_x: np.ndarray,
    cache: dict[str, object],
) -> np.ndarray:
    method = method.lower()

    if method in {"msp", "entropy", "energy", "maxlogit", "odin", "gen", "gradnorm", "kl_matching"}:
        if "eval_logits" not in cache:
            logits, by_class = _logit_training(train_x, train_y, eval_x)
            cache["eval_logits"] = logits
            cache["train_logits_by_class"] = by_class
        logits = cache["eval_logits"]
        if method in {"msp", "entropy", "energy", "maxlogit", "odin", "gen"}:
            return logits_to_scores(logits, method)
        if method == "gradnorm":
            return gradnorm(logits)
        return kl_matching(logits, cache["train_logits_by_class"])

    if method == "maf":
        if "maf_main" not in cache:
            fit_x = val_x if len(val_x) else train_x
            fit_y = val_y if len(val_y) else train_y
            cache["maf_main"] = MAFScorer().fit(fit_x, fit_y)
        return cache["maf_main"].score(eval_x)

    if method == "openmax":
        if "openmax" not in cache:
            cache["openmax"] = _openmax_score(train_x, train_y, eval_x)
        return cache["openmax"]
    if method in {"mahalanobis", "mah_mindist"}:
        if "mahalanobis" not in cache:
            cache["mahalanobis"] = mahalanobis_score(train_x, train_y, eval_x)
        return cache["mahalanobis"]
    if method == "rmd":
        if "rmd" not in cache:
            cache["rmd"] = rmd_score(train_x, train_y, eval_x)
        return cache["rmd"]
    if method == "knn":
        if "knn" not in cache:
            cache["knn"] = knn_score(train_x, eval_x)
        return cache["knn"]
    if method == "lar":
        if "lar" not in cache:
            cache["lar"] = lar_from_env().fit(train_x).id_scores(eval_x)
        return cache["lar"]
    if method == "lantern":
        if "lantern" not in cache:
            if "split_logits" not in cache:
                train_logits, val_logits, eval_logits = _split_logits(train_x, train_y, val_x, eval_x)
                cache["split_logits"] = (train_logits, val_logits, eval_logits)
                cache["eval_logits"] = eval_logits
                cache["train_logits_by_class"] = {
                    int(c): train_logits[train_y == c] for c in sorted(np.unique(train_y))
                }
            else:
                train_logits, val_logits, eval_logits = cache["split_logits"]
            cache["lantern"] = _lantern_score(
                train_x,
                train_y,
                val_x,
                val_y,
                eval_x,
                train_logits=train_logits,
                val_logits=val_logits,
                eval_logits=eval_logits,
            )
        return cache["lantern"]
    if method == "mahalanobispp":
        if "mahalanobispp" not in cache:
            cache["mahalanobispp"] = mahalanobispp_score(train_x, train_y, val_x, eval_x)
        return cache["mahalanobispp"]
    if method == "vim":
        if "vim" not in cache:
            cache["vim"] = _vim_score(train_x, train_y, eval_x)
        return cache["vim"]

    if "prototypes" not in cache:
        _, cache["prototypes"] = _class_prototypes(train_x, train_y)
    prototypes = cache["prototypes"]
    if method == "react":
        if "react" not in cache:
            cache["react"] = feature_energy(react_features(train_x, eval_x), prototypes)
        return cache["react"]
    if method == "ashp":
        if "ashp" not in cache:
            cache["ashp"] = feature_energy(ash_features(eval_x, variant="p"), prototypes)
        return cache["ashp"]
    if method == "ashs":
        if "ashs" not in cache:
            cache["ashs"] = feature_energy(ash_features(eval_x, variant="s"), prototypes)
        return cache["ashs"]
    if method == "ashb":
        if "ashb" not in cache:
            cache["ashb"] = feature_energy(ash_features(eval_x, variant="b"), prototypes)
        return cache["ashb"]
    if method == "dice":
        if "dice" not in cache:
            cache["dice"] = feature_energy(dice_features(train_x, eval_x), prototypes)
        return cache["dice"]
    if method == "scale":
        if "scale" not in cache:
            cache["scale"] = scale_score(eval_x, prototypes)
        return cache["scale"]
    if method == "nci":
        if "nci" not in cache:
            cache["nci"] = nci_score(eval_x, prototypes)
        return cache["nci"]
    if method == "mcm":
        if "mcm" not in cache:
            cache["mcm"] = mcm_score(eval_x, prototypes)
        return cache["mcm"]
    if method == "clip_zeroshot_msp":
        if "clip_zeroshot_msp" not in cache:
            cache["clip_zeroshot_msp"] = clip_zero_shot_msp(eval_x, prototypes)
        return cache["clip_zeroshot_msp"]
    if method == "clip_text_energy":
        if "clip_text_energy" not in cache:
            cache["clip_text_energy"] = clip_text_energy(eval_x, prototypes)
        return cache["clip_text_energy"]
    if method == "tip_adapter":
        if "tip_adapter" not in cache:
            onehot = np.eye(len(np.unique(train_y)))[train_y]
            cache["tip_adapter"] = tip_adapter_score(eval_x, train_x, onehot, prototypes)
        return cache["tip_adapter"]
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
    group_cols = ["backbone", "seed", "id_set"]
    for _, group in jobs.groupby(group_cols, sort=False):
        first = group.iloc[0]
        split = _load_split(int(first["seed"]), split_dir)
        id_classes = str(first["id_set"]).split("|")
        eval_frame = make_ood_eval_frame(split, id_classes, protocol=protocol)
        rows, features = feature_frame_for_split(eval_frame, str(first["backbone"]))
        train_mask = rows["role"] == "id_train"
        val_mask = rows["role"] == "id_val"
        eval_mask = rows["role"].isin(["id_test", "ood_test"])
        enc = LabelEncoder().fit(rows.loc[train_mask, "class_name"])
        train_y = enc.transform(rows.loc[train_mask, "class_name"])
        val_y = enc.transform(rows.loc[val_mask, "class_name"]) if val_mask.any() else train_y
        train_x = features[train_mask.to_numpy()]
        val_x = features[val_mask.to_numpy()] if val_mask.any() else train_x
        eval_x = features[eval_mask.to_numpy()]
        base_score_rows = rows.loc[eval_mask, ["image_id", "class_name", "split", "role", "ood_label"]].copy()
        base_score_rows["is_id"] = (base_score_rows["ood_label"] == 0).astype(int)
        cache: dict[str, object] = {}
        summaries: list[dict[str, object]] = []
        completed_jobs: list[pd.Series] = []
        artifacts: list[str] = []
        for _, job in group.iterrows():
            scores = _score_ood_group_method(
                str(job["method"]),
                train_x,
                train_y,
                val_x,
                val_y,
                eval_x,
                cache,
            )
            score_rows = base_score_rows.copy()
            score_rows["score"] = scores
            for col in job.index:
                score_rows[col] = job[col]
            score_path = scores_dir / f"{job['job_id']}.parquet"
            score_rows.to_parquet(score_path, index=False)
            metrics = ood_metrics(score_rows["is_id"].to_numpy(), score_rows["score"].to_numpy())
            summaries.append({**job.to_dict(), **metrics})
            completed_jobs.append(job)
            artifacts.append(str(score_path))
        if summaries:
            append_csv_rows(pd.DataFrame(summaries), summary_path)
            append_csv_rows(_completed_rows(completed_jobs, artifacts), "results/coverage/completed_jobs.csv")
            count += len(summaries)
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
    group_cols = ["backbone", "seed", "id_set"]
    for _, group in jobs.groupby(group_cols, sort=False):
        first = group.iloc[0]
        split = _load_split(int(first["seed"]), split_dir)
        id_classes = str(first["id_set"]).split("|")
        eval_frame = make_ood_eval_frame(split, id_classes, protocol="fair")
        rows, features = feature_frame_for_split(eval_frame, str(first["backbone"]))
        train_mask = rows["role"] == "id_train"
        val_mask = rows["role"] == "id_val"
        eval_mask = rows["role"].isin(["id_test", "ood_test"])
        enc = LabelEncoder().fit(rows.loc[train_mask, "class_name"])
        train_y = enc.transform(rows.loc[train_mask, "class_name"])
        val_y = enc.transform(rows.loc[val_mask, "class_name"]) if val_mask.any() else train_y
        train_x = features[train_mask.to_numpy()]
        fit_x = features[val_mask.to_numpy()] if val_mask.any() else features[train_mask.to_numpy()]
        fit_y = val_y
        eval_x = features[eval_mask.to_numpy()]
        score_rows = rows.loc[eval_mask, ["class_name", "ood_label"]].copy()
        score_rows["is_id"] = (score_rows["ood_label"] == 0).astype(int)
        results: list[dict[str, object]] = []
        completed_jobs: list[pd.Series] = []
        artifacts: list[str] = []
        scorer_cache: dict[tuple[object, ...], MAFScorer] = {}
        component_cache: dict[tuple[object, ...], dict[str, np.ndarray]] = {}
        distance_cache: dict[tuple[object, ...], np.ndarray] = {}
        for _, job in group.iterrows():
            cfg = _parse_variant(str(job["variant"]))
            key = (
                str(cfg.get("covariance", "tied_ledoit_wolf")),
                str(cfg.get("feature_norm", "raw")),
                str(cfg.get("distance", "mahalanobis")),
                str(cfg.get("prototype", "class_mean")),
                cfg.get("pca_dim"),
            )
            scorer = scorer_cache.get(key)
            if scorer is None:
                scorer = MAFScorer(
                    alpha=float(cfg.get("alpha", 0.5)),
                    covariance=str(cfg.get("covariance", "tied_ledoit_wolf")),
                    feature_norm=str(cfg.get("feature_norm", "raw")),
                    distance=str(cfg.get("distance", "mahalanobis")),
                    prototype=str(cfg.get("prototype", "class_mean")),
                    pca_dim=cfg.get("pca_dim"),
                ).fit(fit_x, fit_y)
                scorer_cache[key] = scorer
            if "distance_score" in cfg:
                distances = distance_cache.get(key)
                if distances is None:
                    distances = scorer.distances(eval_x)
                    distance_cache[key] = distances
                scores = distance_variant_score(distances, str(cfg["distance_score"]))
            else:
                comp = component_cache.get(key)
                if comp is None:
                    comp = scorer.score_components(eval_x)
                    component_cache[key] = comp
                scores = maf_fusion(
                    comp["s_conf"],
                    comp["s_cons"],
                    alpha=float(cfg.get("alpha", 0.5)),
                    mode=str(cfg.get("fusion", "alpha")),
                )
            metrics = ood_metrics(score_rows["is_id"].to_numpy(), scores)
            results.append({**job.to_dict(), **metrics})
            completed_jobs.append(job)
            artifacts.append(str(out_path))
        if results:
            append_csv_rows(pd.DataFrame(results), out_path)
            append_csv_rows(_completed_rows(completed_jobs, artifacts), "results/coverage/completed_jobs.csv")
            count += len(results)
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
