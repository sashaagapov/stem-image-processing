"""Binary opening followed by closing, implemented with a square kernel."""
from __future__ import annotations

import numpy as np
from PIL import Image


def _erode(mask: np.ndarray, window_size: int) -> np.ndarray:
    radius = window_size // 2
    padded = np.pad(mask, radius, mode="constant", constant_values=False)
    result = np.ones_like(mask, dtype=bool)
    for row_offset in range(window_size):
        for column_offset in range(window_size):
            result &= padded[row_offset : row_offset + mask.shape[0], column_offset : column_offset + mask.shape[1]]
    return result


def _dilate(mask: np.ndarray, window_size: int) -> np.ndarray:
    radius = window_size // 2
    padded = np.pad(mask, radius, mode="constant", constant_values=False)
    result = np.zeros_like(mask, dtype=bool)
    for row_offset in range(window_size):
        for column_offset in range(window_size):
            result |= padded[row_offset : row_offset + mask.shape[0], column_offset : column_offset + mask.shape[1]]
    return result


def apply(image: Image.Image, window_size: int = 3, threshold: int = 128) -> Image.Image:
    """Remove small bright noise by opening, then fill small dark holes by closing."""
    if window_size < 1 or window_size % 2 == 0:
        raise ValueError("window_size must be a positive odd number")
    if not 0 <= threshold <= 255:
        raise ValueError("threshold must be between 0 and 255")
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    luminance = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    opened = _dilate(_erode(luminance >= threshold, window_size), window_size)
    closed = _erode(_dilate(opened, window_size), window_size)
    binary = np.where(closed, 255, 0).astype(np.uint8)
    return Image.fromarray(np.repeat(binary[:, :, None], 3, axis=2))
