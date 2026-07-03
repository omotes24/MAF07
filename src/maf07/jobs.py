from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from . import TARGET_CLASSES
from .config import load_yaml, resolve_path


def _stable_job_id(row: dict[str, object]) -> str:
    payload = json.dumps(row, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _flatten_backbones(backbone_cfg: dict[str, object], tiers: Iterable[str]) -> list[str]:
    names: list[str] = []
    for tier in tiers:
        for item in backbone_cfg.get(tier, []):
            names.append(str(item["name"]))
    return names


def _id_set_text(classes: Iterable[str]) -> str:
    return "|".join(classes)


def _ood_set_text(classes: Iterable[str], id_set: Iterable[str]) -> str:
    id_classes = set(id_set)
    return "|".join(c for c in classes if c not in id_classes)


def ablation_variants(method_cfg: dict[str, object]) -> list[str]:
    cfg = method_cfg["maf_ablation"]
    variants = ["main"]
    for name in cfg.get("fusion", []):
        variants.append(f"fusion={name}")
    for value in cfg.get("alpha_grid", []):
        variants.append(f"alpha={value:.1f}")
    for name in cfg.get("feature_norm", []):
        variants.append(f"feature_norm={name}")
    for dim in cfg.get("pca_dims", []):
        variants.append(f"pca={int(dim)}")
    for name in cfg.get("distance", []):
        variants.append(f"distance={name}")
    for name in cfg.get("covariance", []):
        variants.append(f"covariance={name}")
    for name in cfg.get("prototype", []):
        variants.append(f"prototype={name}")
    for name in cfg.get("prototype_source", []):
        variants.append(f"prototype_source={name}")
    for name in cfg.get("distance_scores", []):
        variants.append(f"distance_score={name}")
    return sorted(set(variants))


def generate_expected_jobs(
    dataset_config: str | Path = "configs/dataset.yaml",
    backbone_config: str | Path = "configs/backbones.yaml",
    methods_config: str | Path = "configs/methods.yaml",
    experiments_config: str | Path = "configs/experiments.yaml",
) -> pd.DataFrame:
    dcfg = load_yaml(dataset_config)
    bcfg = load_yaml(backbone_config)
    mcfg = load_yaml(methods_config)
    ecfg = load_yaml(experiments_config)

    classes = list(dcfg.get("classes", TARGET_CLASSES))
    seeds = [int(s) for s in ecfg.get("seeds", [0, 1, 2])]
    id_sizes = [int(k) for k in ecfg.get("id_sizes", [2, 3, 4, 5, 6, 7])]
    backbone_tiers = ecfg.get("default_backbone_tiers", ["tier0"])
    backbones = _flatten_backbones(bcfg, backbone_tiers)

    rows: list[dict[str, object]] = []
    for seed, backbone, method in itertools.product(seeds, backbones, mcfg["closed_methods"]):
        rows.append(
            {
                "job_kind": "closed",
                "protocol": "closed",
                "seed": seed,
                "backbone": backbone,
                "method": method,
                "variant": "main",
                "id_size": len(classes),
                "id_set": _id_set_text(classes),
                "ood_set": "",
            }
        )

    id_sets = []
    for k in id_sizes:
        id_sets.extend(itertools.combinations(classes, k))

    for protocol in ecfg.get("protocols", ["fair", "oracle"]):
        method_names = mcfg["ood_methods"][protocol]
        for seed, backbone, method, id_set in itertools.product(seeds, backbones, method_names, id_sets):
            rows.append(
                {
                    "job_kind": "ood",
                    "protocol": protocol,
                    "seed": seed,
                    "backbone": backbone,
                    "method": method,
                    "variant": "main",
                    "id_size": len(id_set),
                    "id_set": _id_set_text(id_set),
                    "ood_set": _ood_set_text(classes, id_set),
                }
            )

    for seed, backbone, variant, id_set in itertools.product(
        seeds, backbones, ablation_variants(mcfg), id_sets
    ):
        rows.append(
            {
                "job_kind": "ablation",
                "protocol": "fair",
                "seed": seed,
                "backbone": backbone,
                "method": "maf",
                "variant": variant,
                "id_size": len(id_set),
                "id_set": _id_set_text(id_set),
                "ood_set": _ood_set_text(classes, id_set),
            }
        )

    df = pd.DataFrame(rows)
    df["job_id"] = [_stable_job_id(r) for r in df.to_dict(orient="records")]
    df = df[["job_id"] + [c for c in df.columns if c != "job_id"]]
    if df["job_id"].duplicated().any():
        raise AssertionError("Expected job IDs are not unique")
    return df.sort_values("job_id").reset_index(drop=True)


def write_expected_jobs(output_path: str | Path = "results/coverage/expected_jobs.csv", **kwargs) -> pd.DataFrame:
    df = generate_expected_jobs(**kwargs)
    out = resolve_path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return df


def read_expected_jobs(path: str | Path = "results/coverage/expected_jobs.csv") -> pd.DataFrame:
    expected_path = resolve_path(path)
    if not expected_path.exists():
        return write_expected_jobs(expected_path)
    return pd.read_csv(expected_path)


def append_completed_job(
    job: pd.Series | dict[str, object],
    output_path: str | Path = "results/coverage/completed_jobs.csv",
    *,
    status: str = "completed",
    artifact: str = "",
) -> None:
    row = dict(job)
    row["status"] = status
    row["artifact"] = artifact
    out = resolve_path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    header = not out.exists()
    pd.DataFrame([row]).to_csv(out, mode="a", header=header, index=False)


def read_completed_jobs(path: str | Path = "results/coverage/completed_jobs.csv") -> pd.DataFrame:
    completed_path = resolve_path(path)
    if not completed_path.exists():
        return pd.DataFrame(columns=["job_id", "status"])
    return pd.read_csv(completed_path)


def audit_coverage(
    expected_path: str | Path = "results/coverage/expected_jobs.csv",
    completed_path: str | Path = "results/coverage/completed_jobs.csv",
    missing_path: str | Path = "results/coverage/missing_jobs.csv",
    report_path: str | Path = "results/coverage/coverage_report.json",
    *,
    strict: bool = False,
) -> dict[str, object]:
    expected = read_expected_jobs(expected_path)
    completed = read_completed_jobs(completed_path)
    completed_good = completed[completed.get("status", "completed") == "completed"].copy()
    expected_ids = set(expected["job_id"])
    completed_ids = set(completed_good["job_id"]) if not completed_good.empty else set()
    missing_ids = expected_ids - completed_ids
    extra_ids = completed_ids - expected_ids
    missing = expected[expected["job_id"].isin(missing_ids)].copy()

    miss_out = resolve_path(missing_path)
    miss_out.parent.mkdir(parents=True, exist_ok=True)
    missing.to_csv(miss_out, index=False)

    report = {
        "expected_jobs": int(len(expected)),
        "completed_jobs": int(len(completed_ids & expected_ids)),
        "missing_jobs": int(len(missing_ids)),
        "extra_completed_jobs": int(len(extra_ids)),
        "coverage": float((len(completed_ids & expected_ids) / len(expected)) if len(expected) else 0.0),
        "strict_pass": len(missing_ids) == 0 and len(extra_ids) == 0,
    }
    rep_out = resolve_path(report_path)
    rep_out.parent.mkdir(parents=True, exist_ok=True)
    rep_out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    if strict and not report["strict_pass"]:
        raise SystemExit(
            f"Coverage incomplete: {report['completed_jobs']}/{report['expected_jobs']} completed, "
            f"{report['missing_jobs']} missing, {report['extra_completed_jobs']} extra"
        )
    return report

