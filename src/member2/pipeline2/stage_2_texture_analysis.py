
from __future__ import annotations

import numpy as np
from PIL import Image

from src.member2.image_ops import box_mean, from_array, to_gray_array

# зсув сусіда (dy, dx) для кожного кута; y росте вниз, тому 45° — це вгору-вправо
ANGLE_OFFSETS = {0: (0, 1), 45: (-1, 1), 90: (-1, 0), 135: (-1, -1)}
MAP_FEATURES = ("contrast", "dissimilarity", "homogeneity")


def quantize(gray: np.ndarray, levels: int) -> np.ndarray:
    """0..255 -> 0..levels-1 (рівні інтервали)."""
    return np.clip((gray * levels / 256.0).astype(np.int64), 0, levels - 1)


def _offset(angle: int, distance: int) -> tuple[int, int]:
    if angle not in ANGLE_OFFSETS:
        raise ValueError(f"angle має бути одним із {tuple(ANGLE_OFFSETS)}")
    dy, dx = ANGLE_OFFSETS[angle]
    return dy * distance, dx * distance


def glcm(quantized: np.ndarray, levels: int, distance: int = 1, angle: int = 0) -> np.ndarray:
    """Симетрична нормована GLCM для всього (квантованого) зображення."""
    dy, dx = _offset(angle, distance)
    height, width = quantized.shape
    y0, y1 = max(0, -dy), height - max(0, dy)
    x0, x1 = max(0, -dx), width - max(0, dx)
    matrix = np.zeros((levels, levels), dtype=np.float64)
    if y1 <= y0 or x1 <= x0:  # зображення менше за відстань — пар немає
        return matrix
    first = quantized[y0:y1, x0:x1]
    second = quantized[y0 + dy : y1 + dy, x0 + dx : x1 + dx]
    counts = np.bincount((first * levels + second).ravel(), minlength=levels * levels)
    matrix = counts.reshape(levels, levels).astype(np.float64)
    matrix = matrix + matrix.T  # пара (i, j) і (j, i) — те саме сусідство
    return matrix / matrix.sum()


def glcm_features(matrix: np.ndarray) -> dict[str, float]:
    levels = matrix.shape[0]
    i, j = np.mgrid[0:levels, 0:levels]
    nonzero = matrix[matrix > 0]
    return {
        "contrast": float(np.sum(matrix * (i - j) ** 2)),
        "dissimilarity": float(np.sum(matrix * np.abs(i - j))),
        "homogeneity": float(np.sum(matrix / (1.0 + np.abs(i - j)))),
        "energy": float(np.sqrt(np.sum(matrix**2))),
        "entropy": float(-np.sum(nonzero * np.log2(nonzero))),
    }


def texture_map(gray: np.ndarray, feature: str = "contrast", window: int = 9,
                levels: int = 16, distance: int = 1, angle: int = 0) -> np.ndarray:
    """Карта текстурної ознаки для кожного пікселя (у одиницях рівнів квантування)."""
    if feature not in MAP_FEATURES:
        raise ValueError(f"feature має бути одним із {MAP_FEATURES}")
    quantized = quantize(gray, levels)
    dy, dx = _offset(angle, distance)
    height, width = quantized.shape
    pad = distance
    padded = np.pad(quantized, pad, mode="edge")
    neighbour = padded[pad + dy : pad + dy + height, pad + dx : pad + dx + width]
    difference = np.abs(quantized - neighbour).astype(np.float64)
    if feature == "contrast":
        values = difference**2
    elif feature == "dissimilarity":
        values = difference
    else:
        values = 1.0 / (1.0 + difference)
    return box_mean(values, window)


def apply(image: Image.Image, feature: str = "contrast", window: int = 9, levels: int = 16,
          distance: int = 1, angle: int = 0) -> Image.Image:
    feature_map = texture_map(to_gray_array(image), feature, window, levels, distance, angle)
    peak = feature_map.max()
    scaled = feature_map / peak * 255.0 if peak > 0 else np.zeros_like(feature_map)
    return from_array(scaled)
