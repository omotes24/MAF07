from __future__ import annotations

import pandas as pd
import pytest

from maf07.splits import audit_no_ood_val, make_ood_eval_frame


def test_fair_protocol_uses_only_ood_test() -> None:
    rows = []
    for cls in ["cheetah", "tiger", "lion"]:
        for split in ["train", "val", "test"]:
            rows.append(
                {
                    "image_id": f"{cls}-{split}",
                    "class_name": cls,
                    "split": split,
                    "path": f"/tmp/{cls}-{split}.jpg",
                    "duplicate_group": f"{cls}-{split}",
                }
            )
    frame = make_ood_eval_frame(pd.DataFrame(rows), ["cheetah", "tiger"], protocol="fair")
    ood = frame[frame["ood_label"] == 1]
    assert set(ood["role"]) == {"ood_test"}
    assert set(ood["split"]) == {"test"}


def test_no_ood_val_audit_fails_on_ood_val() -> None:
    frame = pd.DataFrame(
        [
            {"ood_label": 1, "role": "ood_val", "split": "val"},
            {"ood_label": 0, "role": "id_val", "split": "val"},
        ]
    )
    with pytest.raises(AssertionError):
        audit_no_ood_val(frame, protocol="fair")

