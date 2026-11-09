from __future__ import annotations

import numpy as np
from PIL import Image

from src.member2.image_ops import convolve2d, from_array, to_gray_array

KERNEL_4 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
KERNEL_8 = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float64)


def laplacian_edges(gray: np.ndarray, neighbors: int = 4) -> np.ndarray:
    """Мапа контурів: |Лаплас|, нормований так, що максимум = 255."""
    if neighbors not in (4, 8):
        raise ValueError("neighbors має бути 4 або 8")
    response = np.abs(convolve2d(gray, KERNEL_4 if neighbors == 4 else KERNEL_8))
    peak = response.max()
    if peak == 0:  # однотонне зображення — меж немає
        return np.zeros_like(response)
    return response / peak * 255.0


def apply(image: Image.Image, neighbors: int = 4) -> Image.Image:
    return from_array(laplacian_edges(to_gray_array(image), neighbors))
