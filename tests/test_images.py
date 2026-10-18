import numpy as np
from PIL import Image

from app.image_processor import _trustworthy, cutout_quality


def _rgba(alpha: np.ndarray) -> Image.Image:
    img = Image.new("RGBA", (alpha.shape[1], alpha.shape[0]), (200, 50, 50, 255))
    img.putalpha(Image.fromarray(alpha.astype(np.uint8)))
    return img


def test_solid_cutout_is_trusted():
    a = np.zeros((100, 100)); a[20:80, 20:80] = 255           # crisp product, 36% of crop
    cov, half = cutout_quality(_rgba(a))
    assert cov > 0.3 and half == 0
    assert _trustworthy(_rgba(a))


def test_washed_out_cutout_is_not_trusted():
    a = np.zeros((100, 100)); a[20:80, 20:80] = 255
    a[40:80, 20:80] = 120                                       # lower half semi-transparent (chrome / white parts)
    assert not _trustworthy(_rgba(a))


def test_nearly_empty_cutout_is_not_trusted():
    a = np.zeros((100, 100)); a[45:55, 48:52] = 255            # white-on-white: almost nothing kept
    assert not _trustworthy(_rgba(a))
