import torch

from methods.localized_views_v2 import VIEW_NAMES, localized_views, peak_box


def test_peak_box_stays_inside_grid() -> None:
    score = torch.zeros(7, 7)
    score[0, 6] = 1
    assert peak_box(score, 3) == (0, 3, 4, 7)


def test_localized_views_have_stable_shape() -> None:
    images = torch.randn(2, 3, 28, 28)
    cam = torch.rand(2, 7, 7)
    energy = torch.rand(2, 7, 7)
    views = localized_views(images, cam, energy)
    assert views.shape == (2, len(VIEW_NAMES), 3, 28, 28)
    assert torch.isfinite(views).all()
