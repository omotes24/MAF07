#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from maf07.config import resolve_path


def _view(path: str, mode: str, size: int) -> np.ndarray:
    with Image.open(path) as image:
        values = np.asarray(
            image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR),
            dtype=np.float32,
        ) / 255.0
    border = max(1, int(round(size * 0.20)))
    if mode == "full_thumbnail":
        out = values
    elif mode == "border_only":
        out = values.copy()
        out[border:-border, border:-border] = 0.5
    elif mode == "center_only":
        out = np.full_like(values, 0.5)
        out[border:-border, border:-border] = values[border:-border, border:-border]
    elif mode == "corners_only":
        out = np.full_like(values, 0.5)
        out[:border, :border] = values[:border, :border]
        out[:border, -border:] = values[:border, -border:]
        out[-border:, :border] = values[-border:, :border]
        out[-border:, -border:] = values[-border:, -border:]
    else:
        raise ValueError(mode)
    return out.reshape(-1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", default="results/manifest.content_cleaned_conservative.csv"
    )
    parser.add_argument(
        "--split", default="results/splits_content_cleaned_conservative/splits_seed0.csv"
    )
    parser.add_argument("--samples-per-class", type=int, default=2000)
    parser.add_argument("--image-size", type=int, default=16)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output = resolve_path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(resolve_path(args.manifest), low_memory=False)
    split = pd.read_csv(resolve_path(args.split), usecols=["image_id", "split"])
    frame = manifest.merge(split, on="image_id", how="inner")
    sampled = []
    for _, block in frame.groupby(["class_name", "split"], sort=True):
        sampled.append(
            block.sample(
                min(len(block), args.samples_per_class), random_state=20260712
            )
        )
    frame = pd.concat(sampled, ignore_index=True)
    encoder = LabelEncoder().fit(frame["class_name"])
    labels = encoder.transform(frame["class_name"])
    train = frame["split"].eq("train").to_numpy()
    test = frame["split"].eq("test").to_numpy()
    modes = ["full_thumbnail", "border_only", "center_only", "corners_only"]
    rows = []
    for mode in modes:
        features = np.stack(
            [_view(path, mode, args.image_size) for path in frame["path"].astype(str)]
        )
        model = make_pipeline(
            StandardScaler(),
            SGDClassifier(
                loss="log_loss",
                alpha=1e-4,
                max_iter=2000,
                tol=1e-4,
                random_state=20260712,
                class_weight="balanced",
            ),
        )
        model.fit(features[train], labels[train])
        prediction = model.predict(features[test])
        rows.append(
            {
                "view": mode,
                "train_n": int(train.sum()),
                "test_n": int(test.sum()),
                "accuracy": float(accuracy_score(labels[test], prediction)),
                "balanced_accuracy": float(
                    balanced_accuracy_score(labels[test], prediction)
                ),
                "macro_f1": float(
                    f1_score(labels[test], prediction, average="macro")
                ),
                "chance_accuracy": float(1.0 / len(encoder.classes_)),
            }
        )
    results = pd.DataFrame(rows)
    results.to_csv(output / "visual_shortcut_linear_probe.csv", index=False)
    report = {
        "classes": encoder.classes_.tolist(),
        "samples": int(len(frame)),
        "interpretation": (
            "Border/corner accuracy above chance is evidence of exploitable visual shortcuts; "
            "near-chance accuracy does not prove their absence."
        ),
    }
    (output / "visual_shortcut_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
