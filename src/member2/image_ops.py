import math
import numpy as np
from PIL import Image


def to_rgb_array(image):
    return np.asarray(image.convert("RGB"), dtype=np.float64)


def to_gray_array(image):
    rgb = to_rgb_array(image)
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def from_array(array):
    clipped = np.clip(np.round(array), 0, 255).astype(np.uint8)
    if clipped.ndim == 2:
        clipped = np.stack([clipped] * 3, axis=2)
    return Image.fromarray(clipped)


def shift(array, dy, dx, pad):
    # копія зображення, зсунута на (dy, dx); за краєм повторюється крайній піксель
    padded = np.pad(array, [(pad, pad), (pad, pad)] + [(0, 0)] * (array.ndim - 2), mode="edge")
    height, width = array.shape[:2]
    return padded[pad + dy : pad + dy + height, pad + dx : pad + dx + width]


def convolve2d(channel, kernel):
    # ядра Лапласа й Гаусса симетричні, тому перевертати ядро не треба
    radius = kernel.shape[0] // 2
    result = np.zeros_like(channel)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            result += kernel[dy + radius, dx + radius] * shift(channel, dy, dx, radius)
    return result


def gaussian_kernel(sigma):
    radius = math.ceil(3 * sigma)
    coords = np.arange(-radius, radius + 1)
    xx, yy = np.meshgrid(coords, coords)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return kernel / kernel.sum()


def gaussian_blur(array, sigma):
    kernel = gaussian_kernel(sigma)
    if array.ndim == 2:
        return convolve2d(array, kernel)
    return np.stack([convolve2d(array[..., c], kernel) for c in range(3)], axis=2)


def box_mean(array, window):
    # середнє у вікні window x window (window непарне)
    radius = window // 2
    total = np.zeros_like(array)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            total += shift(array, dy, dx, radius)
    return total / (2 * radius + 1) ** 2
