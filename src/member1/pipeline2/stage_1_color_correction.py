"""Стадія 1 пайплайну 2 (member1): Колірна корекція — RGB -> HSV, баланс білого.

Два кроки:
1. Баланс білого за принципом "сірого світу" (gray world): середнє кожного
   каналу приводиться до спільного сірого рівня — компенсує колірний відтінок
   від освітлення/сенсора знімка.
2. Ручна конвертація RGB -> HSV, підвищення насиченості (S) на SATURATION_BOOST
   і конвертація назад у RGB — робить кольори на знімку виразнішими.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

SATURATION_BOOST = 1.2


def gray_world_white_balance(array: np.ndarray) -> np.ndarray:
    """array: float, форма (..., 3). Повертає збалансований масив (без обрізання)."""
    channel_means = array.reshape(-1, 3).mean(axis=0)
    gray_mean = channel_means.mean()
    safe_means = np.where(channel_means > 0, channel_means, 1.0)
    scale = np.where(channel_means > 0, gray_mean / safe_means, 1.0)
    return array * scale


def rgb_to_hsv(array: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """array: float, форма (..., 3), канали 0..255. Повертає H (0..360), S (0..1), V (0..255)."""
    r, g, b = array[..., 0], array[..., 1], array[..., 2]
    max_c = np.maximum(np.maximum(r, g), b)
    min_c = np.minimum(np.minimum(r, g), b)
    delta = max_c - min_c

    value = max_c
    saturation = np.where(max_c > 0, delta / np.where(max_c > 0, max_c, 1.0), 0.0)

    hue = np.zeros_like(max_c)
    has_color = delta > 0
    safe_delta = np.where(has_color, delta, 1.0)  # уникаємо ділення на 0 там, де S=0

    is_r_max = has_color & (max_c == r)
    is_g_max = has_color & (max_c == g) & ~is_r_max
    is_b_max = has_color & ~is_r_max & ~is_g_max

    hue[is_r_max] = 60 * (((g - b) / safe_delta)[is_r_max] % 6)
    hue[is_g_max] = 60 * (((b - r) / safe_delta)[is_g_max] + 2)
    hue[is_b_max] = 60 * (((r - g) / safe_delta)[is_b_max] + 4)
    hue = hue % 360

    return hue, saturation, value


def hsv_to_rgb(hue: np.ndarray, saturation: np.ndarray, value: np.ndarray) -> np.ndarray:
    """Обернена конвертація HSV -> RGB. Повертає масив форми (..., 3), канали 0..255."""
    chroma = value * saturation
    h_sector = hue / 60.0
    x = chroma * (1 - np.abs(h_sector % 2 - 1))
    m = value - chroma

    r1 = np.zeros_like(hue)
    g1 = np.zeros_like(hue)
    b1 = np.zeros_like(hue)

    sectors = [
        ((h_sector >= 0) & (h_sector < 1), chroma, x, np.zeros_like(hue)),
        ((h_sector >= 1) & (h_sector < 2), x, chroma, np.zeros_like(hue)),
        ((h_sector >= 2) & (h_sector < 3), np.zeros_like(hue), chroma, x),
        ((h_sector >= 3) & (h_sector < 4), np.zeros_like(hue), x, chroma),
        ((h_sector >= 4) & (h_sector < 5), x, np.zeros_like(hue), chroma),
        ((h_sector >= 5) & (h_sector < 6), chroma, np.zeros_like(hue), x),
    ]
    for mask, r_val, g_val, b_val in sectors:
        r1[mask] = r_val[mask] if hasattr(r_val, "__getitem__") else r_val
        g1[mask] = g_val[mask] if hasattr(g_val, "__getitem__") else g_val
        b1[mask] = b_val[mask] if hasattr(b_val, "__getitem__") else b_val

    return np.stack([r1 + m, g1 + m, b1 + m], axis=-1)


def apply(image: Image.Image) -> Image.Image:
    array = np.asarray(image.convert("RGB"), dtype=np.float64)
    balanced = np.clip(gray_world_white_balance(array), 0, 255)

    hue, saturation, value = rgb_to_hsv(balanced)
    boosted_saturation = np.clip(saturation * SATURATION_BOOST, 0, 1)
    corrected = hsv_to_rgb(hue, boosted_saturation, value)

    result = np.clip(corrected, 0, 255).round().astype(np.uint8)
    return Image.fromarray(result, "RGB")
