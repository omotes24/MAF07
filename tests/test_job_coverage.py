from __future__ import annotations

from pathlib import Path

import pandas as pd

from maf07.jobs import append_csv_rows, audit_coverage, generate_expected_jobs
from maf07.runner import _job_shard, _selected_backbones_from_env, _selected_methods_from_env


def test_expected_jobs_include_all_ood_split_settings() -> None:
    jobs = generate_expected_jobs()
    ood = jobs[(jobs["job_kind"] == "ood") & (jobs["protocol"] == "fair")]
    base = ood[["seed", "id_set"]].drop_duplicates()
    assert len(base) == 246 * 3


def test_coverage_detects_missing_jobs(tmp_path: Path) -> None:
    expected = pd.DataFrame(
        [
            {"job_id": "a", "job_kind": "closed"},
            {"job_id": "b", "job_kind": "closed"},
        ]
    )
    completed = pd.DataFrame([{"job_id": "a", "status": "completed"}])
    exp = tmp_path / "expected.csv"
    comp = tmp_path / "completed.csv"
    miss = tmp_path / "missing.csv"
    rep = tmp_path / "report.json"
    expected.to_csv(exp, index=False)
    completed.to_csv(comp, index=False)
    report = audit_coverage(exp, comp, miss, rep)
    assert report["expected_jobs"] == 2
    assert report["completed_jobs"] == 1
    assert report["missing_jobs"] == 1
    assert pd.read_csv(miss)["job_id"].tolist() == ["b"]


def test_job_shards_are_disjoint() -> None:
    jobs = pd.DataFrame({"job_id": [str(i) for i in range(10)]})
    shards = [_job_shard(jobs, i, 4)["job_id"].tolist() for i in range(4)]
    flattened = [job_id for shard in shards for job_id in shard]
    assert sorted(flattened) == [str(i) for i in range(10)]
    assert len(flattened) == len(set(flattened))


def test_append_csv_rows_adds_one_header(tmp_path: Path) -> None:
    out = tmp_path / "rows.csv"
    append_csv_rows(pd.DataFrame([{"job_id": "a", "status": "completed"}]), out)
    append_csv_rows(pd.DataFrame([{"job_id": "b", "status": "completed"}]), out)
    lines = out.read_text(encoding="utf-8").splitlines()
    assert lines.count("job_id,status") == 1
    assert pd.read_csv(out)["job_id"].tolist() == ["a", "b"]


def test_backbone_filter_env_accepts_commas_and_spaces(monkeypatch) -> None:
    monkeypatch.setenv("MAF07_BACKBONES", "dinov2_vitb14, dinov2_vitl14 openai_clip_vitb16")
    assert _selected_backbones_from_env() == {
        "dinov2_vitb14",
        "dinov2_vitl14",
        "openai_clip_vitb16",
    }


def test_method_filter_env_accepts_commas_and_spaces(monkeypatch) -> None:
    monkeypatch.setenv("MAF07_METHODS", "lantern, lar knn")
    assert _selected_methods_from_env() == {"lantern", "lar", "knn"}
