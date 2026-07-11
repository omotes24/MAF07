from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_verified_baseline_audit import _summarize


def test_equivalence_audit_pairs_rows_within_id_size(tmp_path: Path) -> None:
    rows = []
    for id_size in (2, 3):
        for method in ("method_a", "method_b"):
            rows.append(
                {
                    "job_id": f"{id_size}-{method}",
                    "backbone": "dinov2_vitb14",
                    "seed": 0,
                    "id_size": id_size,
                    "id_set_id": 0,
                    "method": method,
                    "implementation_version": "verified_v1",
                    "score_sha256": "same" if id_size == 2 else method,
                    "AUROC": 0.9 if id_size == 2 else 0.8 + 0.01 * (method == "method_b"),
                    "FPR95": 0.4 if id_size == 2 else 0.5,
                    "AUPR_OUT": 0.9 if id_size == 2 else 0.8,
                }
            )

    input_path = tmp_path / "fold.csv"
    summary_path = tmp_path / "summary.csv"
    equivalence_path = tmp_path / "equivalence.csv"
    pd.DataFrame(rows).to_csv(input_path, index=False)

    _summarize(input_path, summary_path, equivalence_path)

    equivalence = pd.read_csv(equivalence_path)
    assert len(equivalence) == 1
    assert equivalence.loc[0, "n_common"] == 2
    assert not bool(equivalence.loc[0, "all_score_fingerprints_equal"])
    assert not bool(equivalence.loc[0, "all_metrics_equal"])
