#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torchvision.models import ResNet50_Weights, resnet50


CAT_TERMS = (
    "cat",
    "tiger",
    "lion",
    "leopard",
    "jaguar",
    "cheetah",
    "cougar",
    "lynx",
    "snow leopard",
)


def _load_existing(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _load_exclusions(path: Path | None) -> set[str]:
    if path is None or not path.exists():
        return set()
    out: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        item = line.strip()
        if item and not item.startswith("#"):
            out.add(item)
    return out


def _write_exclusion_list(path: Path, rel_paths: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Auto-generated candidate exclusions from scripts/score_image_content.py.",
        "# Paths are relative to the CAT dataset root.",
    ]
    lines.extend(sorted(set(rel_paths)))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_model(checkpoint: Path | None, device: torch.device):
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=None)
    if checkpoint is not None and checkpoint.exists():
        try:
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state)
    else:
        model = resnet50(weights=weights)
    model.to(device).eval()
    labels = weights.meta["categories"]
    cat_idx = torch.as_tensor(
        [i for i, label in enumerate(labels) if any(term in label.lower() for term in CAT_TERMS)],
        dtype=torch.long,
        device=device,
    )
    return model, weights.transforms(), labels, cat_idx


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="results/manifest.csv")
    parser.add_argument("--output", default="results/content_filter/imagenet_cat_scores.csv")
    parser.add_argument("--exclude-output", default="results/content_filter/auto_excluded_images.txt")
    parser.add_argument("--existing-exclude-list", default="configs/excluded_images.txt")
    parser.add_argument("--checkpoint", default=os.environ.get("MAF07_IMAGENET_RESNET50", "/home/omote/OODD/checkpoints/resnet50-0676ba61.pth"))
    parser.add_argument("--threshold", type=float, default=0.20)
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--worker-index", type=int, default=int(os.environ.get("MAF07_JOB_WORKER_INDEX", "0")))
    parser.add_argument("--worker-count", type=int, default=int(os.environ.get("MAF07_JOB_WORKER_COUNT", "1")))
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    if args.limit is not None:
        manifest = manifest.head(args.limit).copy()
    if args.worker_count > 1:
        if args.worker_index < 0 or args.worker_index >= args.worker_count:
            raise ValueError("--worker-index must be in [0, --worker-count)")
        manifest = manifest.iloc[args.worker_index :: args.worker_count].reset_index(drop=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    existing = _load_existing(output)
    done = set(existing["rel_path"].astype(str)) if "rel_path" in existing else set()

    device = torch.device(args.device)
    checkpoint = Path(args.checkpoint) if args.checkpoint else None
    model, preprocess, labels, cat_idx = _load_model(checkpoint, device)

    pending = manifest[~manifest["rel_path"].astype(str).isin(done)].copy()
    rows: list[dict[str, object]] = []

    def flush() -> None:
        nonlocal rows
        if not rows:
            return
        frame = pd.DataFrame(rows)
        frame.to_csv(output, mode="a", header=not output.exists(), index=False)
        rows = []

    batch_images = []
    batch_meta = []
    processed = 0
    with torch.no_grad():
        for row in pending.itertuples(index=False):
            rel_path = str(getattr(row, "rel_path"))
            path = str(getattr(row, "path"))
            try:
                img = Image.open(path).convert("RGB")
                batch_images.append(preprocess(img))
                batch_meta.append(row)
            except Exception as exc:
                rows.append(
                    {
                        "image_id": str(getattr(row, "image_id")),
                        "class_name": str(getattr(row, "class_name")),
                        "rel_path": rel_path,
                        "path": path,
                        "cat_score": 0.0,
                        "top1_label": "read_error",
                        "top1_prob": 0.0,
                        "top5_labels": json.dumps(["read_error"]),
                        "error": str(exc)[:200],
                    }
                )
            if len(batch_images) < int(args.batch_size):
                continue
            x = torch.stack(batch_images).to(device)
            prob = model(x).softmax(1)
            cat_scores = prob.index_select(1, cat_idx).sum(1).detach().cpu().numpy()
            top = torch.topk(prob, k=5, dim=1)
            for i, meta in enumerate(batch_meta):
                top_indices = top.indices[i].detach().cpu().tolist()
                top_values = top.values[i].detach().cpu().tolist()
                rows.append(
                    {
                        "image_id": str(getattr(meta, "image_id")),
                        "class_name": str(getattr(meta, "class_name")),
                        "rel_path": str(getattr(meta, "rel_path")),
                        "path": str(getattr(meta, "path")),
                        "cat_score": float(cat_scores[i]),
                        "top1_label": labels[int(top_indices[0])],
                        "top1_prob": float(top_values[0]),
                        "top5_labels": json.dumps([labels[int(j)] for j in top_indices]),
                        "error": "",
                    }
                )
            processed += len(batch_images)
            batch_images = []
            batch_meta = []
            if len(rows) >= 2048:
                flush()
                print(json.dumps({"processed": processed, "output": str(output)}), flush=True)

        if batch_images:
            x = torch.stack(batch_images).to(device)
            prob = model(x).softmax(1)
            cat_scores = prob.index_select(1, cat_idx).sum(1).detach().cpu().numpy()
            top = torch.topk(prob, k=5, dim=1)
            for i, meta in enumerate(batch_meta):
                top_indices = top.indices[i].detach().cpu().tolist()
                top_values = top.values[i].detach().cpu().tolist()
                rows.append(
                    {
                        "image_id": str(getattr(meta, "image_id")),
                        "class_name": str(getattr(meta, "class_name")),
                        "rel_path": str(getattr(meta, "rel_path")),
                        "path": str(getattr(meta, "path")),
                        "cat_score": float(cat_scores[i]),
                        "top1_label": labels[int(top_indices[0])],
                        "top1_prob": float(top_values[0]),
                        "top5_labels": json.dumps([labels[int(j)] for j in top_indices]),
                        "error": "",
                    }
                )
            processed += len(batch_images)
    flush()

    scores = pd.read_csv(output)
    existing_exclusions = _load_exclusions(Path(args.existing_exclude_list) if args.existing_exclude_list else None)
    proposed = scores.loc[scores["cat_score"] < float(args.threshold), "rel_path"].astype(str).tolist()
    proposed = sorted(set(proposed) | existing_exclusions)
    _write_exclusion_list(Path(args.exclude_output), proposed)

    summary = (
        scores.assign(auto_excluded=scores["rel_path"].astype(str).isin(proposed))
        .groupby("class_name", dropna=False)
        .agg(n=("rel_path", "size"), excluded=("auto_excluded", "sum"), cat_score_mean=("cat_score", "mean"))
        .reset_index()
    )
    summary["excluded"] = summary["excluded"].astype(int)
    summary["cat_score_mean"] = summary["cat_score_mean"].round(6)
    summary_path = Path(args.exclude_output).with_suffix(".summary.csv")
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False))
    print(json.dumps({"scores": str(output), "exclusions": str(args.exclude_output), "summary": str(summary_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
