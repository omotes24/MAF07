from __future__ import annotations

import pandas as pd

from maf07.splits import assign_splits, assert_no_duplicate_group_cross_split


def _manifest(n_per_class: int = 12) -> pd.DataFrame:
    rows = []
    for cls in ["cheetah", "tiger"]:
        for i in range(n_per_class):
            rows.append(
                {
                    "image_id": f"{cls}-{i}",
                    "class_name": cls,
                    "path": f"/tmp/{cls}/{i}.jpg",
                    "duplicate_group": f"{cls}-g{i}",
                }
            )
    return pd.DataFrame(rows)


def test_split_ratio_is_4_1_1_per_class() -> None:
    split = assign_splits(_manifest(), seed=0)
    counts = split.groupby(["class_name", "split"]).size().unstack(fill_value=0)
    for _, row in counts.iterrows():
        assert int(row["train"]) == 8
        assert int(row["val"]) == 2
        assert int(row["test"]) == 2


def test_duplicate_group_does_not_cross_splits() -> None:
    df = _manifest()
    df.loc[[0, 1], "duplicate_group"] = "dup-a"
    split = assign_splits(df, seed=1)
    assert_no_duplicate_group_cross_split(split)
    assert split[split["duplicate_group"] == "dup-a"]["split"].nunique() == 1

