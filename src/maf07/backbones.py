from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

from .config import load_yaml


@dataclass(frozen=True)
class BackboneSpec:
    name: str
    family: str
    model_name: str
    checkpoint_name: str | None = None
    image_size: int = 224
    feature_dim: int | None = None


class BackboneRunner:
    def __init__(self, spec: BackboneSpec, device: str = "cuda") -> None:
        self.spec = spec
        self.device = device
        self.model = None
        self.preprocess = None
        self._load()

    def _load(self) -> None:
        if self.spec.family == "timm":
            import timm
            import torch
            from timm.data import create_transform, resolve_data_config

            model = timm.create_model(self.spec.model_name, pretrained=True, num_classes=0)
            model.eval()
            if device_available(self.device):
                model = model.to(self.device)
            data_config = resolve_data_config({}, model=model)
            self.preprocess = create_transform(**data_config)
            self.model = model
            return
        if self.spec.family == "open_clip":
            import open_clip

            model, _, preprocess = open_clip.create_model_and_transforms(
                self.spec.model_name,
                pretrained=self.spec.checkpoint_name,
            )
            model.eval()
            if device_available(self.device):
                model = model.to(self.device)
            self.preprocess = preprocess
            self.model = model
            return
        raise ValueError(f"Unknown backbone family: {self.spec.family}")

    def encode_pil(self, images: Sequence[Image.Image], batch_size: int = 64) -> np.ndarray:
        import torch

        if self.model is None or self.preprocess is None:
            raise RuntimeError("Backbone is not loaded")
        feats: list[np.ndarray] = []
        dev = self.device if device_available(self.device) else "cpu"
        with torch.no_grad():
            for start in range(0, len(images), batch_size):
                batch_images = images[start : start + batch_size]
                batch = torch.stack([self.preprocess(img.convert("RGB")) for img in batch_images]).to(dev)
                if self.spec.family == "open_clip":
                    out = self.model.encode_image(batch)
                else:
                    out = self.model(batch)
                if isinstance(out, (tuple, list)):
                    out = out[0]
                if out.ndim == 4:
                    out = out.mean(dim=(-2, -1))
                feats.append(out.detach().cpu().float().numpy())
        return np.concatenate(feats, axis=0)


def device_available(device: str) -> bool:
    if device == "cpu":
        return True
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def load_backbone_specs(config_path: str | Path = "configs/backbones.yaml") -> dict[str, BackboneSpec]:
    cfg = load_yaml(config_path)
    specs: dict[str, BackboneSpec] = {}
    for tier in ("tier0", "tier1", "tier2"):
        for item in cfg.get(tier, []):
            spec = BackboneSpec(
                name=str(item["name"]),
                family=str(item["family"]),
                model_name=str(item["model_name"]),
                checkpoint_name=item.get("checkpoint_name"),
                image_size=int(item.get("image_size", 224)),
                feature_dim=item.get("feature_dim"),
            )
            specs[spec.name] = spec
    return specs


def extract_features(
    images: Sequence[Image.Image | str | Path],
    backbone_name: str,
    *,
    backbone_config: str | Path = "configs/backbones.yaml",
    device: str = "cuda",
    batch_size: int = 64,
) -> np.ndarray:
    specs = load_backbone_specs(backbone_config)
    if backbone_name not in specs:
        raise KeyError(f"Unknown backbone: {backbone_name}")
    pil_images = [Image.open(x).convert("RGB") if not isinstance(x, Image.Image) else x for x in images]
    runner = BackboneRunner(specs[backbone_name], device=device)
    return runner.encode_pil(pil_images, batch_size=batch_size)

