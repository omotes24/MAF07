from __future__ import annotations

from pathlib import Path

import pandas as pd

from maf07.jobs import audit_coverage, generate_expected_jobs


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

