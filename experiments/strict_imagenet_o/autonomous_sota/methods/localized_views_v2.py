"""Deterministic local views from frozen ResNet activation maps."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from extract_saliency_views import mass_bbox


VIEW_NAMES = (
    "cam_mass50",
    "energy_mass50",
    "energy_mass80",
    "energy_peak3",
)


def _crop_and_resize(
    image: torch.Tensor,
    box: tuple[int, int, int, int],
    grid_shape: tuple[int, int],
) -> torch.Tensor:
    image_height, image_width = image.shape[-2:]
    grid_height, grid_width = grid_shape
    y0, y1, x0, x1 = box
    py0 = (y0 * image_height) // grid_height
    py1 = max(py0 + 1, (y1 * image_height + grid_height - 1) // grid_height)
    px0 = (x0 * image_width) // grid_width
    px1 = max(px0 + 1, (x1 * image_width + grid_width - 1) // grid_width)
    return F.interpolate(
        image[:, py0:py1, px0:px1].unsqueeze(0),
        size=(image_height, image_width),
        mode="bilinear",
        align_corners=False,
    )[0]


def peak_box(score: torch.Tensor, width: int = 3) -> tuple[int, int, int, int]:
    """Return a fixed square around the strongest activation cell."""
    if score.ndim != 2 or width < 1:
        raise ValueError("score must be a matrix and width must be positive")
    height, columns = score.shape
    width = min(width, height, columns)
    index = int(score.reshape(-1).argmax())
    center_y, center_x = divmod(index, columns)
    y0 = min(max(center_y - width // 2, 0), height - width)
    x0 = min(max(center_x - width // 2, 0), columns - width)
    return y0, y0 + width, x0, x0 + width


def localized_views(
    images: torch.Tensor,
    class_cam: torch.Tensor,
    activation_energy: torch.Tensor,
) -> torch.Tensor:
    """Create four local crops without using data from any other query image."""
    if images.ndim != 4 or class_cam.ndim != 3 or activation_energy.ndim != 3:
        raise ValueError("expected images BxCxHxW and score maps Bxhxw")
    if not (len(images) == len(class_cam) == len(activation_energy)):
        raise ValueError("batch sizes do not match")
    if class_cam.shape != activation_energy.shape:
        raise ValueError("score map shapes do not match")

    grid_shape = tuple(class_cam.shape[-2:])
    rows = []
    for image, cam, energy in zip(images, class_cam, activation_energy):
        boxes = (
            mass_bbox(cam, mass_fraction=0.5, padding=0),
            mass_bbox(energy, mass_fraction=0.5, padding=0),
            mass_bbox(energy, mass_fraction=0.8, padding=0),
            peak_box(energy, width=3),
        )
        rows.append(
            torch.stack([_crop_and_resize(image, box, grid_shape) for box in boxes])
        )
    return torch.stack(rows)
