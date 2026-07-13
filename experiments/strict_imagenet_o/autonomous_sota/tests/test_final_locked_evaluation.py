from pathlib import Path

import numpy as np

from evaluate_final_locked import load_scores
from extract_final_locked import localized_prototype_confidence, manifest_rows


def test_load_scores_preserves_score_orientation(tmp_path: Path) -> None:
    path = tmp_path / "scores.npz"
    np.savez(
        path,
        pulse_ood_score=np.asarray([1.0, 2.0], dtype=np.float32),
        rc_msps_confidence=np.asarray([0.8, 0.2], dtype=np.float32),
    )
    scores = load_scores(path)
    np.testing.assert_array_equal(scores["pulse"], [1.0, 2.0])
    np.testing.assert_allclose(scores["rc"], [0.8, 0.2])


def test_manifest_rows_selects_only_locked_dataset(tmp_path: Path) -> None:
    path = tmp_path / "manifest.jsonl"
    path.write_text(
        '{"dataset":"NINCO","relative_path":"ninco/a.jpg"}\n'
        '{"dataset":"SSB-hard","relative_path":"ssb_hard/b.jpg"}\n',
        encoding="utf-8",
    )
    assert manifest_rows(path, "NINCO") == [
        {"dataset": "NINCO", "relative_path": "ninco/a.jpg"}
    ]


def test_localized_confidence_takes_best_view_and_class() -> None:
    localized = np.asarray([[[1.0, 0.0], [0.0, 1.0]]], dtype=np.float32)
    prototypes = np.asarray([[1.0, 0.0], [-1.0, 0.0]], dtype=np.float32)
    confidence = localized_prototype_confidence(
        localized, prototypes, device="cpu", batch_size=1
    )
    np.testing.assert_allclose(confidence, [1.0])
