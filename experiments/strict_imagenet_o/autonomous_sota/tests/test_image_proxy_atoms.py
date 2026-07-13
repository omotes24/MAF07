import numpy as np
from PIL import Image

from extract_image_proxy_atoms import center_cutmix, patch_shuffle4, phase_mix


def solid(color: tuple[int, int, int]) -> Image.Image:
    return Image.new("RGB", (224, 224), color)


def test_center_cutmix_uses_partner_only_in_center() -> None:
    mixed = np.asarray(center_cutmix(solid((255, 0, 0)), solid((0, 0, 255))))
    np.testing.assert_array_equal(mixed[0, 0], np.array([255, 0, 0]))
    np.testing.assert_array_equal(mixed[100, 100], np.array([0, 0, 255]))


def test_patch_shuffle_preserves_pixel_multiset() -> None:
    values = np.arange(224 * 224, dtype=np.uint16).reshape(224, 224)
    image = Image.fromarray((values % 256).astype(np.uint8), mode="L").convert("RGB")
    shuffled = np.asarray(patch_shuffle4(image))
    original = np.asarray(image)
    np.testing.assert_array_equal(np.sort(shuffled.reshape(-1)), np.sort(original.reshape(-1)))


def test_phase_mix_is_finite_rgb() -> None:
    mixed = np.asarray(phase_mix(solid((128, 64, 32)), solid((32, 128, 64))))
    assert mixed.shape == (224, 224, 3)
    assert np.isfinite(mixed).all()
