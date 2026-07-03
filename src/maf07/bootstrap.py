from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from .metrics import ood_metrics


def stratified_paired_sample_indices(
    strata: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    strata = np.asarray(strata)
    sampled: list[np.ndarray] = []
    for value in sorted(pd.unique(strata)):
        idx = np.flatnonzero(strata == value)
        sampled.append(rng.choice(idx, size=len(idx), replace=True))
    return np.concatenate(sampled)


def bootstrap_metric_ci(
    frame: pd.DataFrame,
    score_col: str,
    *,
    y_col: str = "is_id",
    strata_cols: tuple[str, ...] = ("is_id", "class_name"),
    n_boot: int = 2000,
    seed: int = 0,
    metric_fn: Callable[[np.ndarray, np.ndarray], dict[str, float]] | None = None,
) -> pd.DataFrame:
    if n_boot < 1:
        raise ValueError("n_boot must be positive")
    rng = np.random.default_rng(seed)
    strata = frame.loc[:, list(strata_cols)].astype(str).agg("|".join, axis=1).to_numpy()
    y = frame[y_col].to_numpy(dtype=int)
    s = frame[score_col].to_numpy(dtype=float)
    metric_fn = metric_fn or (lambda yy, ss: ood_metrics(yy, ss))
    rows: list[dict[str, float]] = []
    for _ in range(n_boot):
        idx = stratified_paired_sample_indices(strata, rng)
        rows.append(metric_fn(y[idx], s[idx]))
    boot = pd.DataFrame(rows)
    out_rows = []
    for metric in boot.columns:
        values = boot[metric].to_numpy(dtype=float)
        out_rows.append(
            {
                "metric": metric,
                "mean": float(np.mean(values)),
                "ci_low": float(np.quantile(values, 0.025)),
                "ci_high": float(np.quantile(values, 0.975)),
                "n_boot": int(n_boot),
            }
        )
    return pd.DataFrame(out_rows)


def paired_bootstrap_pvalue(
    frame: pd.DataFrame,
    score_a: str,
    score_b: str,
    *,
    metric: str = "AUROC",
    y_col: str = "is_id",
    strata_cols: tuple[str, ...] = ("is_id", "class_name"),
    n_boot: int = 2000,
    seed: int = 0,
) -> float:
    rng = np.random.default_rng(seed)
    strata = frame.loc[:, list(strata_cols)].astype(str).agg("|".join, axis=1).to_numpy()
    y = frame[y_col].to_numpy(dtype=int)
    a = frame[score_a].to_numpy(dtype=float)
    b = frame[score_b].to_numpy(dtype=float)
    diffs = []
    for _ in range(n_boot):
        idx = stratified_paired_sample_indices(strata, rng)
        diffs.append(ood_metrics(y[idx], a[idx])[metric] - ood_metrics(y[idx], b[idx])[metric])
    diffs_arr = np.asarray(diffs)
    return float(2.0 * min(np.mean(diffs_arr <= 0), np.mean(diffs_arr >= 0)))

