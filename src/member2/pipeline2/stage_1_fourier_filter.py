import numpy as np
from src.member2.image_ops import from_array, to_rgb_array


def lowpass_mask(height, width, cutoff):
    # відстань D від центру спектра; маска Гаусса exp(-D^2 / (2 cutoff^2))
    v = np.arange(height) - height // 2
    u = np.arange(width) - width // 2
    uu, vv = np.meshgrid(u, v)
    distance_sq = uu**2 + vv**2
    return np.exp(-distance_sq / (2 * cutoff**2))


def lowpass_filter(channel, cutoff):
    spectrum = np.fft.fftshift(np.fft.fft2(channel))
    filtered = spectrum * lowpass_mask(channel.shape[0], channel.shape[1], cutoff)
    return np.real(np.fft.ifft2(np.fft.ifftshift(filtered)))


def apply(image, cutoff=40):
    rgb = to_rgb_array(image)
    channels = [lowpass_filter(rgb[..., c], cutoff) for c in range(3)]
    return from_array(np.stack(channels, axis=2))
