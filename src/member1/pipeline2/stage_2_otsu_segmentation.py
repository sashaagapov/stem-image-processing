"""Стадія 2 пайплайну 2 (member1): Сегментація — метод Отсу.

Будує гістограму яскравості, перебирає всі пороги 0..255 і для кожного рахує
міжкласову дисперсію (between-class variance) між пікселями "нижче/вище
порогу". Обирається поріг із максимальною дисперсією — він найкраще розділяє
зображення на дві групи (напр. вода/суша). При рівних значеннях дисперсії
береться найменший поріг (перший знайдений максимум).
"""
from __future__ import annotations

import numpy as np
from PIL import Image


def _to_grayscale(array: np.ndarray) -> np.ndarray:
    return 0.299 * array[:, :, 0] + 0.587 * array[:, :, 1] + 0.114 * array[:, :, 2]


def find_otsu_threshold(gray: np.ndarray) -> int:
    histogram = np.bincount(gray.flatten().astype(np.int64), minlength=256).astype(np.float64)
    total_pixels = gray.size

    best_threshold = 0
    best_variance = -1.0

    weight_background = 0.0
    sum_background = 0.0
    total_sum = float(np.dot(histogram, np.arange(256)))

    for threshold in range(256):
        weight_background += histogram[threshold]
        if weight_background == 0:
            continue
        weight_foreground = total_pixels - weight_background
        if weight_foreground == 0:
            break

        sum_background += threshold * histogram[threshold]
        mean_background = sum_background / weight_background
        mean_foreground = (total_sum - sum_background) / weight_foreground

        between_class_variance = (
            weight_background * weight_foreground * (mean_background - mean_foreground) ** 2
        )

        if between_class_variance > best_variance:
            best_variance = between_class_variance
            best_threshold = threshold

    return best_threshold


def apply(image: Image.Image) -> Image.Image:
    array = np.asarray(image.convert("RGB"), dtype=np.float64)
    gray = _to_grayscale(array)

    threshold = find_otsu_threshold(gray)
    mask = np.where(gray > threshold, 255, 0).astype(np.uint8)

    rgb = np.stack([mask, mask, mask], axis=-1)
    return Image.fromarray(rgb, "RGB")
