"""Стадія 1 пайплайну 1 (member2): Фільтрація — білатеральний фільтр.

Звичайне згладжування (середнє, Гаусс — вони були в ЛР6) розмиває все
підряд, зокрема межі доріг і дахів, які потім шукає Лаплас. Білатеральний
фільтр (Tomasi, Manduchi, 1998) усереднює лише СХОЖІ за кольором сусідні
пікселі: шум на полі згладжується, а перепад "поле / дорога" лишається різким.

Вага сусіда q для пікселя p — добуток двох гаусіан:
    просторова:  exp(-(dx^2 + dy^2) / (2 sigma_spatial^2))  — близькі важать більше;
    кольорова:   exp(-|I(q) - I(p)|^2 / (2 sigma_range^2))   — схожі за кольором важать більше,
де |I(q) - I(p)|^2 — сума квадратів різниць по каналах R, G, B.
Новий колір p = сума(вага * I(q)) / сума(ваг).

Параметри (config/pipelines.yaml):
    sigma_spatial — наскільки далеко збираємо сусідів (у пікселях);
    sigma_range   — яка різниця кольору ще вважається "схожою" (0..255):
                    менша — сильніше зберігаються межі, більша — ближче до Гаусса;
    radius        — половина розміру вікна; якщо не задано, ceil(2 * sigma_spatial).
"""
from __future__ import annotations

import math

import numpy as np
from PIL import Image

from src.member2.image_ops import from_array, to_rgb_array


def bilateral_filter(rgb: np.ndarray, sigma_spatial: float = 2.0, sigma_range: float = 30.0,
                     radius: int | None = None) -> np.ndarray:
    if sigma_spatial <= 0 or sigma_range <= 0:
        raise ValueError("sigma_spatial і sigma_range мають бути більшими за 0")
    if radius is None:
        radius = math.ceil(2 * sigma_spatial)
    height, width = rgb.shape[:2]
    padded = np.pad(rgb, ((radius, radius), (radius, radius), (0, 0)), mode="edge")

    weighted_sum = np.zeros_like(rgb)
    weight_total = np.zeros((height, width))
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            # сусід на відстані (dy, dx) — для всіх пікселів одразу
            neighbour = padded[radius + dy : radius + dy + height, radius + dx : radius + dx + width]
            spatial = math.exp(-(dx * dx + dy * dy) / (2 * sigma_spatial**2))
            colour_distance = np.sum((neighbour - rgb) ** 2, axis=2)
            weight = spatial * np.exp(-colour_distance / (2 * sigma_range**2))
            weighted_sum += weight[..., None] * neighbour
            weight_total += weight
    return weighted_sum / weight_total[..., None]


def apply(image: Image.Image, sigma_spatial: float = 2.0, sigma_range: float = 30.0,
          radius: int | None = None) -> Image.Image:
    return from_array(bilateral_filter(to_rgb_array(image), sigma_spatial, sigma_range, radius))
