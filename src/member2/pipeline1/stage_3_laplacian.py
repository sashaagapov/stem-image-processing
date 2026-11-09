"""Стадія 3 пайплайну 1 (member2): Контурний аналіз — Лаплас.

Оператор Лапласа — друга похідна яскравості. На рівній ділянці він дає 0,
а на межі (дорога / поле, дах / тінь) — великий за модулем відгук.
Модуль відгуку нормується до 0..255: найсильніша межа знімка стає білою.

Ядра:
    4 сусіди: [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
    8 сусідів: [[1, 1, 1], [1, -8, 1], [1, 1, 1]] — реагує й на діагональні межі.

Параметри (config/pipelines.yaml):
    neighbors — 4 або 8.
"""
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
