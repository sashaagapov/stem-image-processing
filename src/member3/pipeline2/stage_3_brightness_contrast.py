"""Per-channel brightness and contrast correction."""
from __future__ import annotations

import numpy as np
from PIL import Image


def apply(image: Image.Image, brightness: int = 0, contrast: float = 1.1) -> Image.Image:
    """Apply ``(value - 128) * contrast + 128 + brightness`` and clamp it."""
    if not -255 <= brightness <= 255:
        raise ValueError("brightness must be between -255 and 255")
    if contrast < 0.0:
        raise ValueError("contrast must be non-negative")
    source = np.asarray(image.convert("RGB"), dtype=np.float64)
    corrected = np.clip(np.rint((source - 128.0) * contrast + 128.0 + brightness), 0, 255)
    return Image.fromarray(corrected.astype(np.uint8))
