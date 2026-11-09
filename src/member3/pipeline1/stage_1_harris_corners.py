"""Harris corner detector implemented from image gradients and a structure tensor."""
from __future__ import annotations

import numpy as np
from PIL import Image


def _gray_array(image: Image.Image) -> np.ndarray:
    """Return luminance values while accepting RGB and grayscale Pillow images."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.float64)
    return 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]


def _box_sum(values: np.ndarray, window_size: int) -> np.ndarray:
    """Sum a square window at every pixel, using zero padding beyond an edge."""
    radius = window_size // 2
    padded = np.pad(values, radius, mode="constant")
    result = np.zeros_like(values, dtype=np.float64)
    for row_offset in range(window_size):
        for column_offset in range(window_size):
            result += padded[
                row_offset : row_offset + values.shape[0],
                column_offset : column_offset + values.shape[1],
            ]
    return result


def apply(
    image: Image.Image,
    threshold: float = 0.01,
    harris_k: float = 0.04,
    window_size: int = 3,
) -> Image.Image:
    """Mark pixels whose Harris response exceeds ``threshold`` of the maximum."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    if not 0.0 < harris_k < 0.25:
        raise ValueError("harris_k must be between 0 and 0.25")
    if window_size < 1 or window_size % 2 == 0:
        raise ValueError("window_size must be a positive odd number")

    gray = _gray_array(image)
    padded = np.pad(gray, 1, mode="edge")
    gradient_x = padded[1:-1, 2:] - padded[1:-1, :-2]
    gradient_y = padded[2:, 1:-1] - padded[:-2, 1:-1]
    tensor_xx = _box_sum(gradient_x * gradient_x, window_size)
    tensor_yy = _box_sum(gradient_y * gradient_y, window_size)
    tensor_xy = _box_sum(gradient_x * gradient_y, window_size)
    response = tensor_xx * tensor_yy - tensor_xy**2 - harris_k * (tensor_xx + tensor_yy) ** 2

    largest_response = float(np.max(response)) if response.size else 0.0
    result = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    if largest_response > 0.0:
        result[response > threshold * largest_response] = (255, 0, 0)
    return Image.fromarray(result)
