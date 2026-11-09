"""Стадія 2 пайплайну 1 (member1): Гістограмна корекція — вирівнювання гістограми.

Застосовується окремо до кожного з каналів R/G/B: будує гістограму яскравості
(256 рівнів), кумулятивну функцію розподілу (CDF) і перемаповує кожен піксель
через нормалізовану CDF — так, щоб вихідні значення рівномірніше покривали
діапазон 0..255.
"""
from __future__ import annotations

import numpy as np
from PIL import Image


def equalize_channel(channel: np.ndarray) -> np.ndarray:
    """Вирівнювання гістограми для одного каналу (значення 0..255)."""
    total_pixels = channel.size
    histogram = np.bincount(channel.flatten().astype(np.int64), minlength=256)
    cdf = np.cumsum(histogram)

    lookup_table = np.round(255 * cdf / total_pixels).astype(np.uint8)
    return lookup_table[channel.astype(np.int64)]


def apply(image: Image.Image) -> Image.Image:
    array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    equalized = np.stack(
        [equalize_channel(array[:, :, channel]) for channel in range(3)],
        axis=-1,
    )
    return Image.fromarray(equalized, "RGB")
