
from __future__ import annotations

import numpy as np
from PIL import Image

from src.member2.image_ops import from_array, to_rgb_array


def frequency_mask(shape: tuple[int, int], cutoff: float, mode: str = "lowpass", kind: str = "gaussian") -> np.ndarray:
    """Маска частот для зсунутого (fftshift) спектра розміру shape."""
    if cutoff <= 0:
        raise ValueError("cutoff має бути більшим за 0")
    if mode not in ("lowpass", "highpass") or kind not in ("gaussian", "ideal"):
        raise ValueError("mode: lowpass|highpass, kind: gaussian|ideal")
    height, width = shape
    v = np.arange(height) - height // 2  # після fftshift нульова частота стоїть тут
    u = np.arange(width) - width // 2
    uu, vv = np.meshgrid(u, v)
    distance = np.sqrt(uu**2 + vv**2)
    if kind == "ideal":
        low = (distance <= cutoff).astype(np.float64)
    else:
        low = np.exp(-(distance**2) / (2 * cutoff**2))
    return low if mode == "lowpass" else 1.0 - low


def frequency_filter(channel: np.ndarray, cutoff: float, mode: str = "lowpass", kind: str = "gaussian") -> np.ndarray:
    spectrum = np.fft.fftshift(np.fft.fft2(channel))
    filtered = spectrum * frequency_mask(channel.shape, cutoff, mode, kind)
    result = np.real(np.fft.ifft2(np.fft.ifftshift(filtered)))
    if mode == "highpass":
        result += 128.0
    return result


def apply(image: Image.Image, cutoff: float = 30, mode: str = "lowpass", kind: str = "gaussian") -> Image.Image:
    rgb = to_rgb_array(image)
    channels = [frequency_filter(rgb[..., c], cutoff, mode, kind) for c in range(3)]
    return from_array(np.stack(channels, axis=2))
