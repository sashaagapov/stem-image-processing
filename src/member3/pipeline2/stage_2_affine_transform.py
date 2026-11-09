"""Inverse-mapped affine transformation using nearest-neighbour sampling."""
from __future__ import annotations

import math

import numpy as np
from PIL import Image


def apply(image: Image.Image, angle_degrees: float = 2.0, scale_x: float = 1.0, scale_y: float = 1.0, translate_x: float = 0.0, translate_y: float = 0.0, fill_value: int = 0) -> Image.Image:
    """Rotate, scale and translate around the image centre with inverse mapping."""
    if scale_x == 0.0 or scale_y == 0.0:
        raise ValueError("scale_x and scale_y must be non-zero")
    if not 0 <= fill_value <= 255:
        raise ValueError("fill_value must be between 0 and 255")
    source = np.asarray(image.convert("RGB"), dtype=np.uint8)
    height, width = source.shape[:2]
    angle = math.radians(angle_degrees)
    cosine, sine = math.cos(angle), math.sin(angle)
    forward = np.array([[scale_x * cosine, -scale_y * sine], [scale_x * sine, scale_y * cosine]])
    inverse = np.linalg.inv(forward)
    result = np.full_like(source, fill_value)
    centre_x, centre_y = (width - 1) / 2.0, (height - 1) / 2.0
    for destination_y in range(height):
        for destination_x in range(width):
            destination = np.array([destination_x - centre_x - translate_x, destination_y - centre_y - translate_y])
            source_x, source_y = inverse @ destination + np.array([centre_x, centre_y])
            nearest_x, nearest_y = int(math.floor(source_x + 0.5)), int(math.floor(source_y + 0.5))
            if 0 <= nearest_x < width and 0 <= nearest_y < height:
                result[destination_y, destination_x] = source[nearest_y, nearest_x]
    return Image.fromarray(result)
