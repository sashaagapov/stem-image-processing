"""Стадія 3 пайплайну 2 (member1): Статистичні характеристики — середнє/дисперсія/ентропія.

Рахує три характеристики гістограми яскравості: середнє (mean), дисперсію
(variance) і ентропію (Shannon entropy, біти) — показують середню яскравість,
розкид і "інформаційну складність" ділянки (ліс/текстурна поверхня матиме
вищу ентропію, ніж рівне поле чи вода). Значення малюються текстом на копії
зображення, щоб стадія лишалась у контракті apply(image) -> Image.Image.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw


def _to_grayscale(array: np.ndarray) -> np.ndarray:
    return 0.299 * array[:, :, 0] + 0.587 * array[:, :, 1] + 0.114 * array[:, :, 2]


def calculate_statistics(image: Image.Image) -> dict:
    array = np.asarray(image.convert("RGB"), dtype=np.float64)
    gray = _to_grayscale(array)

    mean = float(np.mean(gray))
    variance = float(np.mean((gray - mean) ** 2))

    histogram = np.bincount(gray.flatten().astype(np.int64), minlength=256).astype(np.float64)
    probabilities = histogram[histogram > 0] / gray.size
    entropy = float(-np.sum(probabilities * np.log2(probabilities)))

    return {"mean": mean, "variance": variance, "entropy": entropy}


def apply(image: Image.Image) -> Image.Image:
    rgb_image = image.convert("RGB")
    stats = calculate_statistics(rgb_image)

    result = rgb_image.copy()
    draw = ImageDraw.Draw(result)
    text = f"mean={stats['mean']:.1f} var={stats['variance']:.1f} entropy={stats['entropy']:.2f}"
    draw.text((2, 2), text, fill=(255, 0, 0))

    return result
