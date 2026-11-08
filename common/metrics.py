from __future__ import annotations

import numpy as np
from PIL import Image


def _rgb_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.float64)


def _grayscale_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("L"), dtype=np.float64)


def _check_same_size(a: Image.Image, b: Image.Image) -> None:
    if a.size != b.size:
        raise ValueError("Зображення мають бути однакового розміру для порівняння метрик.")


def calculate_mse(image_a: Image.Image, image_b: Image.Image) -> float:
    """Середньоквадратична похибка між пікселями двох RGB-зображень."""
    _check_same_size(image_a, image_b)
    a = _rgb_array(image_a)
    b = _rgb_array(image_b)
    return float(np.mean((a - b) ** 2))


def calculate_psnr(image_a: Image.Image, image_b: Image.Image, max_value: float = 255.0) -> float:
    """Піковий рівень сигнал/шум. inf, якщо зображення ідентичні."""
    mse = calculate_mse(image_a, image_b)
    if mse == 0:
        return float("inf")
    return float(10 * np.log10((max_value**2) / mse))


def _box_mean(array: np.ndarray, window: int) -> np.ndarray:
    """Середнє у вікні window x window для кожного пікселя — через інтегральне зображення
    (summed-area table), щоб не ховати формулу за готовою функцією згортки з бібліотеки."""
    half = window // 2
    padded = np.pad(array, half, mode="reflect")

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


def calculate_ssim(image_a: Image.Image, image_b: Image.Image, window: int = 7) -> float:
    """Спрощений SSIM (Wang et al., 2004): формула структурної подібності застосована
    вручну до локальних статистик (середнє/дисперсія/коваріація), порахованих
    прямокутним (box) вікном замість гаусового — простіше для реалізації й пояснення,
    принцип той самий."""
    _check_same_size(image_a, image_b)
    a = _grayscale_array(image_a)
    b = _grayscale_array(image_b)

    max_value = 255.0
    c1 = (0.01 * max_value) ** 2
    c2 = (0.03 * max_value) ** 2

    mean_a = _box_mean(a, window)
    mean_b = _box_mean(b, window)
    var_a = _box_mean(a * a, window) - mean_a**2
    var_b = _box_mean(b * b, window) - mean_b**2
    covariance = _box_mean(a * b, window) - mean_a * mean_b

    numerator = (2 * mean_a * mean_b + c1) * (2 * covariance + c2)
    denominator = (mean_a**2 + mean_b**2 + c1) * (var_a + var_b + c2)
    ssim_map = numerator / denominator

    return float(np.mean(ssim_map))
