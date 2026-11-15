import math
import numpy as np
from src.member2.image_ops import from_array, shift, to_rgb_array


def bilateral_filter(rgb, sigma_spatial, sigma_range):
    radius = math.ceil(2 * sigma_spatial)
    weighted_sum = np.zeros_like(rgb)
    weight_total = np.zeros(rgb.shape[:2])
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            neighbour = shift(rgb, dy, dx, radius)
            spatial = math.exp(-(dx * dx + dy * dy) / (2 * sigma_spatial**2))
            colour_distance = np.sum((neighbour - rgb) ** 2, axis=2)
            weight = spatial * np.exp(-colour_distance / (2 * sigma_range**2))
            weighted_sum += weight[..., None] * neighbour
            weight_total += weight
    return weighted_sum / weight_total[..., None]


def apply(image, sigma_spatial=2.0, sigma_range=50.0):
    return from_array(bilateral_filter(to_rgb_array(image), sigma_spatial, sigma_range))
