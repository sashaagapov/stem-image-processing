"""Стадія 2 пайплайну 1 (member2): Морфологічна обробка — бінарна ерозія/дилатація.

1. Бінаризація: піксель стає "об'єктом" (білим), якщо його яскравість більша
   за поріг. Поріг за замовчуванням — середня яскравість знімка (а не Оцу:
   Оцу вже реалізує member1, а тут бінаризація — лише підготовчий крок).
2. Бінарна морфологія зі структурним елементом-ДИСКОМ радіуса radius:
       ерозія:    піксель лишається білим, лише якщо ВЕСЬ диск навколо нього
                  лежить усередині об'єкта — прибирає дрібні світлі плями
                  й тонкі відростки, відокремлює ділянки, що ледь торкаються;
       дилатація: піксель стає білим, якщо диск зачіпає ХОЧ ОДИН білий
                  піксель — зшиває розірвані фрагменти доріг, заповнює щілини.
Це відрізняється від фільтрів мінімуму/максимуму з ЛР6 (там — сірі значення
і квадратне вікно) і від відкриття/закриття, які робить member3.
Результат — чорно-біла маска (0 / 255), з якої Лаплас (стадія 3) виділяє межі.

Параметри (config/pipelines.yaml):
    operation — erosion | dilation;
    radius    — радіус диска в пікселях (1 — хрестик 3x3, 2 — диск 5x5, ...);
    threshold — "mean" або число 0..255;
    invert    — true, якщо об'єкти темніші за фон (вода, тіні).
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from src.member2.image_ops import from_array, to_gray_array

OPERATIONS = ("erosion", "dilation")


def binarize(gray: np.ndarray, threshold="mean", invert: bool = False) -> np.ndarray:
    level = gray.mean() if threshold == "mean" else float(threshold)
    mask = gray > level
    return ~mask if invert else mask


def disk(radius: int) -> np.ndarray:
    """Структурний елемент: True там, де dx^2 + dy^2 <= radius^2."""
    offsets = np.arange(-radius, radius + 1)
    xx, yy = np.meshgrid(offsets, offsets)
    return xx**2 + yy**2 <= radius**2


def binary_morphology(mask: np.ndarray, operation: str = "erosion", radius: int = 1) -> np.ndarray:
    if operation not in OPERATIONS:
        raise ValueError(f"operation має бути одним із {OPERATIONS}, отримано {operation!r}")
    radius = max(0, int(radius))
    element = disk(radius)
    height, width = mask.shape
    padded = np.pad(mask, radius, mode="edge")  # за межею — те саме, що на краю
    erode = operation == "erosion"
    result = np.ones_like(mask) if erode else np.zeros_like(mask)
    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            if not element[dy, dx]:
                continue
            shifted = padded[dy : dy + height, dx : dx + width]
            result = (result & shifted) if erode else (result | shifted)
    return result


def apply(image: Image.Image, operation: str = "erosion", radius: int = 1,
          threshold="mean", invert: bool = False) -> Image.Image:
    mask = binarize(to_gray_array(image), threshold, invert)
    return from_array(binary_morphology(mask, operation, radius) * 255.0)
