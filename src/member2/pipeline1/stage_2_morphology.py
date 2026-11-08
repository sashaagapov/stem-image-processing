"""Стадія 2 пайплайну 1 (member2): Морфологічна обробка — Ерозія/дилатація.

Морфологія у відтінках сірого (без бінаризації):
    ерозія    = мінімум у вікні window_size x window_size,
    дилатація = максимум у вікні.
Комбінації:
    відкриття (opening) = ерозія, потім дилатація — прибирає світлі деталі,
        менші за вікно (окремі машини, відблиски), не зсуваючи великих меж;
    закриття (closing)  = дилатація, потім ерозія — заповнює темні дрібниці
        (тіні, щілини між дахами).
Результат — сіре зображення (у трьох однакових каналах).

Параметри (config/pipelines.yaml):
    window_size — розмір квадратного структурного елемента (непарний);
    operation   — erosion | dilation | opening | closing.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from src.member2.image_ops import from_array, to_gray_array

OPERATIONS = ("erosion", "dilation", "opening", "closing")


def _window_extreme(gray: np.ndarray, window_size: int, use_min: bool) -> np.ndarray:
    """Мінімум (або максимум) у вікні навколо кожного пікселя.
    Межі доповнюються крайніми пікселями, тому вони не "з'їдаються"."""
    radius = window_size // 2
    height, width = gray.shape
    padded = np.pad(gray, radius, mode="edge")
    result = padded[0:height, 0:width].copy()
    for dy in range(window_size):
        for dx in range(window_size):
            shifted = padded[dy : dy + height, dx : dx + width]
            result = np.minimum(result, shifted) if use_min else np.maximum(result, shifted)
    return result


def erosion(gray: np.ndarray, window_size: int) -> np.ndarray:
    return _window_extreme(gray, window_size, use_min=True)


def dilation(gray: np.ndarray, window_size: int) -> np.ndarray:
    return _window_extreme(gray, window_size, use_min=False)


def morphology(gray: np.ndarray, window_size: int = 3, operation: str = "opening") -> np.ndarray:
    if operation not in OPERATIONS:
        raise ValueError(f"operation має бути одним із {OPERATIONS}, отримано {operation!r}")
    window_size = max(1, int(window_size))
    if window_size % 2 == 0:
        window_size += 1  # вікно має мати центр
    if operation == "erosion":
        return erosion(gray, window_size)
    if operation == "dilation":
        return dilation(gray, window_size)
    if operation == "opening":
        return dilation(erosion(gray, window_size), window_size)
    return erosion(dilation(gray, window_size), window_size)


def apply(image: Image.Image, window_size: int = 3, operation: str = "opening") -> Image.Image:
    return from_array(morphology(to_gray_array(image), window_size, operation))
