import numpy as np
from src.member2.image_ops import from_array, shift, to_gray_array


def disk(radius):
    offsets = np.arange(-radius, radius + 1)
    xx, yy = np.meshgrid(offsets, offsets)
    return xx**2 + yy**2 <= radius**2


def binary_morphology(mask, operation, radius):
    element = disk(radius)
    if operation == "erosion":
        result = np.ones_like(mask)
    elif operation == "dilation":
        result = np.zeros_like(mask)
    else:
        raise ValueError("operation має бути erosion або dilation")
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if element[dy + radius, dx + radius]:
                neighbour = shift(mask, dy, dx, radius)
                if operation == "erosion":
                    result = result & neighbour
                else:
                    result = result | neighbour
    return result


def apply(image, operation="erosion", radius=1):
    gray = to_gray_array(image)
    mask = gray > gray.mean()
    return from_array(binary_morphology(mask, operation, radius) * 255.0)
