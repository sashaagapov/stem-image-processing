"""Спільні допоміжні функції member2: перетворення PIL <-> numpy, переведення
у відтінки сірого, згортка та гаусове ядро. Усе написано вручну на numpy —
бібліотека використовується лише для масивів і базової математики.
"""
from __future__ import annotations

import math

import numpy as np
from PIL import Image


def to_rgb_array(image: Image.Image) -> np.ndarray:
    """PIL-зображення будь-якого режиму (L, RGBA, P...) -> масив H x W x 3 (float64)."""
    return np.asarray(image.convert("RGB"), dtype=np.float64)


def to_gray_array(image: Image.Image) -> np.ndarray:
    """Яскравість за формулою ITU-R BT.601: Y = 0.299 R + 0.587 G + 0.114 B."""
    rgb = to_rgb_array(image)
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def from_array(array: np.ndarray) -> Image.Image:
    """Масив (H x W або H x W x 3) -> RGB-зображення. Значення округлюються
    й обрізаються до 0..255. Сірий масив дублюється в три канали, бо спільна
    інфраструктура (common/) порівнює результати як RGB."""
    clipped = np.clip(np.round(array), 0, 255).astype(np.uint8)
    if clipped.ndim == 2:
        clipped = np.stack([clipped] * 3, axis=2)
    return Image.fromarray(clipped)


def convolve2d(channel: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Двовимірна згортка одного каналу з ядром.

    Межі доповнюються повторенням крайніх пікселів (режим "edge"), тому розмір
    результату дорівнює розміру входу. Для кожного зсуву (dy, dx) ядра додаємо
    зсунуту копію зображення, помножену на вагу ядра — це та сама сума
    sum(kernel[dy, dx] * image[y + dy, x + dx]), лише порахована для всіх
    пікселів одразу. Ядро перевертається, щоб це була саме згортка
    (для симетричних ядер — Гаусса, Лапласа — різниці немає).
    """
    kernel = np.asarray(kernel, dtype=np.float64)[::-1, ::-1]
    kernel_h, kernel_w = kernel.shape
    pad_y, pad_x = kernel_h // 2, kernel_w // 2
    height, width = channel.shape
    padded = np.pad(channel, ((pad_y, pad_y), (pad_x, pad_x)), mode="edge")

    result = np.zeros((height, width), dtype=np.float64)
    for dy in range(kernel_h):
        for dx in range(kernel_w):
            weight = kernel[dy, dx]
            if weight != 0:
                result += weight * padded[dy : dy + height, dx : dx + width]
    return result


def gaussian_kernel(sigma: float, radius: int | None = None) -> np.ndarray:
    """Гаусове ядро G(x, y) = exp(-(x^2 + y^2) / (2 sigma^2)), нормоване до суми 1.

    Радіус за замовчуванням ceil(3 sigma): за межами 3 sigma ваги менші за 1 %
    від центральної, тож їх можна відкинути.
    """
    if sigma <= 0:
        raise ValueError("sigma має бути більшою за 0")
    if radius is None:
        radius = math.ceil(3 * sigma)
    coords = np.arange(-radius, radius + 1, dtype=np.float64)
    xx, yy = np.meshgrid(coords, coords)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return kernel / kernel.sum()


def box_mean(array: np.ndarray, window: int) -> np.ndarray:
    """Середнє у вікні window x window навколо кожного пікселя через інтегральне
    зображення (summed-area table): сума у прямокутнику = 4 звертання до таблиці."""
    window = max(1, int(window))
    before = window // 2
    after = window - 1 - before
    padded = np.pad(array, ((before, after), (before, after)), mode="edge")
    integral = np.zeros((padded.shape[0] + 1, padded.shape[1] + 1))
    integral[1:, 1:] = np.cumsum(np.cumsum(padded, axis=0), axis=1)
    height, width = array.shape
    window_sum = (
        integral[window : window + height, window : window + width]
        - integral[0:height, window : window + width]
        - integral[window : window + height, 0:width]
        + integral[0:height, 0:width]
    )
    return window_sum / (window * window)
