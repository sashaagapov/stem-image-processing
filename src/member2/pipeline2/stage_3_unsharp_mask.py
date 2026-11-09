
from __future__ import annotations

from PIL import Image

from src.member2.image_ops import from_array, gaussian_blur, to_rgb_array


def apply(image: Image.Image, amount: float = 1.0, sigma: float = 1.0, radius: int | None = None) -> Image.Image:
    original = to_rgb_array(image)
    blurred = gaussian_blur(original, sigma, radius)
    return from_array(original + amount * (original - blurred))
