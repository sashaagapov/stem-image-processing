"""Стадія 1 пайплайну 1 (member2): Фільтрація — Gaussian blur (не з ЛР5-6).

Згладжування гаусовим ядром прибирає дрібний шум знімка (зерно сенсора,
JPEG-артефакти), щоб наступні стадії (морфологія, Лаплас) реагували на межі
ділянок, а не на шум. Кожен канал RGB згортається окремо, кольори зберігаються.

Параметри (config/pipelines.yaml):
    sigma  — ширина ядра; більше sigma — сильніше розмиття.
    radius — половина розміру ядра; якщо не задано, ceil(3 * sigma).
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from src.member2.image_ops import convolve2d, from_array, gaussian_kernel, to_rgb_array


def gaussian_blur_array(array: np.ndarray, sigma: float, radius: int | None = None) -> np.ndarray:
    """Розмиття масиву H x W або H x W x 3 (кожен канал окремо)."""
    kernel = gaussian_kernel(sigma, radius)
    if array.ndim == 2:
        return convolve2d(array, kernel)
    return np.stack([convolve2d(array[..., c], kernel) for c in range(array.shape[2])], axis=2)


def apply(image: Image.Image, sigma: float = 2.0, radius: int | None = None) -> Image.Image:
    return from_array(gaussian_blur_array(to_rgb_array(image), sigma, radius))
