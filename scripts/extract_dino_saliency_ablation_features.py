#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from maf07.backbones import BackboneRunner, load_backbone_specs
from maf07.config import resolve_path
from maf07.data import load_manifest
from maf07.features import cache_paths, write_feature_cache


def _tokens(model, batch):
    output = model.forward_features(batch)
    if isinstance(output, dict):
        cls = output.get("x_norm_clstoken")
        patches = output.get("x_norm_patchtokens")
        if cls is None or patches is None:
            raise RuntimeError(f"Unsupported forward_features dictionary: {sorted(output)}")
        return cls, patches
    if output.ndim != 3:
        raise RuntimeError(f"Expected token output, got shape={tuple(output.shape)}")
    prefix = int(getattr(model, "num_prefix_tokens", 1))
    if prefix < 1 or output.shape[1] <= prefix:
        raise RuntimeError("Backbone does not expose CLS and patch tokens")
    return output[:, 0], output[:, prefix:]


def _model_features(model, batch):
    output = model(batch)
    if isinstance(output, (tuple, list)):
        output = output[0]
    if output.ndim == 4:
        output = output.mean(dim=(-2, -1))
    if output.ndim != 2:
        raise RuntimeError(f"Unexpected model output shape: {tuple(output.shape)}")
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="dinov2_vitb14")
    parser.add_argument(
        "--manifest", default="results/manifest.content_cleaned_conservative.csv"
    )
    parser.add_argument("--output-prefix", default="")
    parser.add_argument("--mode", choices=["foreground", "background", "both"], default="both")
    parser.add_argument("--keep-quantile", type=float, default=0.70)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache-dir", default="results/features")
    parser.add_argument("--mask-stats", default="")
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Limit rows for a smoke test; use a distinct --output-prefix.",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    import torch
    import torch.nn.functional as functional

    if not 0.0 < args.keep_quantile < 1.0:
        raise SystemExit("--keep-quantile must be in (0, 1)")
    specs = load_backbone_specs()
    if args.backbone not in specs:
        raise SystemExit(f"Unknown backbone: {args.backbone}")
    spec = specs[args.backbone]
    if spec.family != "timm" or "dinov2" not in spec.model_name:
        raise SystemExit("Saliency masking currently requires a timm DINOv2 backbone")

    prefix = args.output_prefix or args.backbone
    output_names = []
    if args.mode in {"foreground", "both"}:
        output_names.append(("foreground", f"{prefix}_saliency_fg"))
    if args.mode in {"background", "both"}:
        output_names.append(("background", f"{prefix}_saliency_bg"))
    if not args.force and all(
        cache_paths(name, args.cache_dir).feature_path.exists()
        and cache_paths(name, args.cache_dir).metadata_path.exists()
        for _, name in output_names
    ):
        print(json.dumps({"status": "already_complete", "outputs": [name for _, name in output_names]}))
        return 0

    manifest = load_manifest(args.manifest)
    rows = manifest[["image_id", "class_name", "path", "rel_path"]].copy()
    if args.max_images is not None:
        if args.max_images < 1:
            raise SystemExit("--max-images must be positive")
        rows = rows.head(args.max_images).reset_index(drop=True)
    runner = BackboneRunner(spec, device=args.device)
    model = runner.model
    preprocess = runner.preprocess
    if model is None or preprocess is None:
        raise RuntimeError("Backbone failed to load")
    model.eval()

    features = {mode: [] for mode, _ in output_names}
    mask_rows = []
    paths = rows["path"].astype(str).tolist()
    with torch.no_grad():
        for start in tqdm(range(0, len(paths), args.batch_size), desc=f"saliency:{args.backbone}"):
            batch_paths = paths[start : start + args.batch_size]
            images = [Image.open(path).convert("RGB") for path in batch_paths]
            batch = torch.stack([preprocess(image) for image in images]).to(args.device)
            cls, patches = _tokens(model, batch)
            patch_count = int(patches.shape[1])
            grid_h, grid_w = getattr(model.patch_embed, "grid_size", (0, 0))
            grid_h, grid_w = int(grid_h), int(grid_w)
            if grid_h * grid_w != patch_count:
                side = int(round(patch_count**0.5))
                if side * side != patch_count:
                    raise RuntimeError(
                        f"Cannot reshape {patch_count} patch tokens into a spatial grid"
                    )
                grid_h = grid_w = side

            similarity = functional.cosine_similarity(
                patches, cls[:, None, :], dim=2
            )
            threshold = torch.quantile(
                similarity, float(args.keep_quantile), dim=1, keepdim=True
            )
            patch_mask = (similarity >= threshold).to(batch.dtype)
            spatial = patch_mask.reshape(-1, 1, grid_h, grid_w)
            spatial = functional.interpolate(
                spatial, size=batch.shape[-2:], mode="nearest"
            )
            foreground = batch * spatial
            background = batch * (1.0 - spatial)
            for mode, _ in output_names:
                masked = foreground if mode == "foreground" else background
                features[mode].append(
                    _model_features(model, masked).detach().cpu().float().numpy()
                )
            kept = patch_mask.mean(dim=1).detach().cpu().numpy()
            for offset, value in enumerate(kept):
                mask_rows.append(
                    {
                        "image_id": str(rows.iloc[start + offset]["image_id"]),
                        "backbone": args.backbone,
                        "keep_quantile": float(args.keep_quantile),
                        "foreground_patch_fraction": float(value),
                    }
                )
            for image in images:
                image.close()

    written = []
    for mode, name in output_names:
        values = np.concatenate(features[mode], axis=0)
        paths_out = write_feature_cache(
            name, rows, values, cache_dir=args.cache_dir
        )
        written.append(
            {
                "mode": mode,
                "backbone": name,
                "feature_path": str(paths_out.feature_path),
                "metadata_path": str(paths_out.metadata_path),
                "shape": list(values.shape),
            }
        )
    mask_stats = resolve_path(
        args.mask_stats
        or f"results/reviewer_rsn/saliency_{args.backbone}_mask_stats.csv"
    )
    mask_stats.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(mask_rows).to_csv(mask_stats, index=False)
    print(json.dumps({"status": "complete", "outputs": written, "mask_stats": str(mask_stats)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
