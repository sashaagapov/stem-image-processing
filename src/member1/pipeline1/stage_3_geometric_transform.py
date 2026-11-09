"""Стадія 3 пайплайну 1 (member1): Геометричні перетворення — поворот/масштабування.

Для кожного пікселя результату обчислюється обернене відображення в джерело
(inverse mapping): куди в оригінальному зображенні "дивиться" ця точка після
повороту навколо центру на angle_degrees і масштабування в scale разів.
Значення береться методом найближчого сусіда (nearest-neighbor); точки поза
межами джерела заповнюються чорним.
"""
from __future__ import annotations

import numpy as np
from PIL import Image


def apply(image: Image.Image, angle_degrees: float = 15, scale: float = 1.0) -> Image.Image:
    array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    height, width = array.shape[:2]
    center_y, center_x = (height - 1) / 2, (width - 1) / 2

    theta = np.radians(angle_degrees)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    dest_y, dest_x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    dy = dest_y - center_y
    dx = dest_x - center_x

    # Обернене відображення: координати джерела для кожного пікселя результату.
    src_x = center_x + (dx * cos_t + dy * sin_t) / scale
    src_y = center_y + (-dx * sin_t + dy * cos_t) / scale

    src_x_rounded = np.round(src_x).astype(np.int64)
    src_y_rounded = np.round(src_y).astype(np.int64)

    valid = (src_x_rounded >= 0) & (src_x_rounded < width) & (src_y_rounded >= 0) & (src_y_rounded < height)

    result = np.zeros_like(array)
    result[valid] = array[src_y_rounded[valid], src_x_rounded[valid]]

    return Image.fromarray(result, "RGB")
