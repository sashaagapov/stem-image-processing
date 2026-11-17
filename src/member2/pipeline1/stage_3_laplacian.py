import numpy as np
from src.member2.image_ops import convolve2d, from_array, to_gray_array

KERNEL_4 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
KERNEL_8 = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float64)


def laplacian_edges(gray, neighbors):
    if neighbors == 4:
        kernel = KERNEL_4
    elif neighbors == 8:
        kernel = KERNEL_8
    else:
        raise ValueError("neighbors має бути 4 або 8")
    response = np.abs(convolve2d(gray, kernel))
    peak = response.max()
    if peak == 0:
        return response
    return response / peak * 255.0


def apply(image, neighbors=8):
    return from_array(laplacian_edges(to_gray_array(image), neighbors))
