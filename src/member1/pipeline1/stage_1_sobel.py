"""Стадія 1 пайплайну 1 (member1): Контурний аналіз — Sobel.

Оператор Собеля: згортка з ядрами Gx/Gy (3x3) дає оцінку градієнта яскравості
в кожній точці; величина градієнта sqrt(Gx^2+Gy^2) висока на межах об'єктів і
низька на однорідних ділянках. Поріг threshold перетворює це на бінарну мапу
контурів: білий піксель, де градієнт достатньо сильний, інакше чорний.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

_GX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
_GY = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)


def _to_grayscale(array: np.ndarray) -> np.ndarray:
    # Стандартна формула яскравості (luminosity), а не просто середнє каналів.
    return 0.299 * array[:, :, 0] + 0.587 * array[:, :, 1] + 0.114 * array[:, :, 2]


def _convolve3x3(padded: np.ndarray, kernel: np.ndarray, height: int, width: int) -> np.ndarray:
    # Згортка 3x3 через суму 9 зсунутих зрізів — без scipy/cv2, лише numpy-слайсинг.
    result = np.zeros((height, width), dtype=np.float64)
    for dy in range(3):
        for dx in range(3):
            weight = kernel[dy, dx]
            if weight == 0:
                continue
            result += weight * padded[dy : dy + height, dx : dx + width]
    return result


def sobel_magnitude(image: Image.Image) -> np.ndarray:
    """Градієнт Собеля (без порогу) — винесено окремо, щоб можна було перевірити
    числа без порогу/бінаризації."""
    array = np.asarray(image.convert("RGB"), dtype=np.float64)
    gray = _to_grayscale(array)
    height, width = gray.shape
    padded = np.pad(gray, 1, mode="reflect")

    gx = _convolve3x3(padded, _GX, height, width)
    gy = _convolve3x3(padded, _GY, height, width)
    return np.sqrt(gx**2 + gy**2)


def apply(image: Image.Image, threshold: int = 100) -> Image.Image:
    magnitude = sobel_magnitude(image)
    edges = np.where(magnitude >= threshold, 255, 0).astype(np.uint8)
    rgb = np.stack([edges, edges, edges], axis=-1)
    return Image.fromarray(rgb, "RGB")
