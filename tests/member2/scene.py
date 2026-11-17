"""Синтетичний "аерознімок" з відомим вмістом для еталонних сценаріїв (критерій C1):
поле, ліс (зерниста текстура), річка, дорога і 6 будівель. Фіксований seed."""
import numpy as np
from PIL import Image


def make_scene(seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.zeros((128, 128, 3))
    img[:] = (170, 150, 110)                                  # поле
    img[:55, :60] = (40, 110, 45)                             # ліс
    img[:55, :60] += rng.normal(0, 20, (55, 60, 1))           # текстура лісу
    yy, xx = np.mgrid[0:128, 0:128]
    img[np.abs(yy - (0.6 * xx + 60)) < 5] = (40, 70, 140)     # річка
    img[:, 90:94] = (110, 110, 110)                           # дорога
    for k in range(6):                                        # будівлі 6x6
        y, x = 75 + (k // 3) * 20, 98 + (k % 3) * 10
        img[y : y + 6, x : x + 6] = (245, 245, 245)
    return np.clip(np.round(img), 0, 255).astype(np.uint8)


def add_noise(clean: np.ndarray, sigma: float = 25, seed: int = 42) -> np.ndarray:
    """Адитивний гаусів шум із фіксованим seed."""
    rng = np.random.default_rng(seed)
    noisy = clean.astype(np.float64) + rng.normal(0, sigma, clean.shape)
    return np.clip(np.round(noisy), 0, 255).astype(np.uint8)


def as_image(array: np.ndarray) -> Image.Image:
    return Image.fromarray(array)
