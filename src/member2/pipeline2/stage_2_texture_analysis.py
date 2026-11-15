import numpy as np
from src.member2.image_ops import box_mean, from_array, shift, to_gray_array


def quantize(gray, levels):
    return np.clip((gray * levels / 256).astype(int), 0, levels - 1)


def glcm(quantized, levels):
    # GLCM для пар "піксель - правий сусід"; пара (i, j) і (j, i) рахуються обидві
    matrix = np.zeros((levels, levels))
    height, width = quantized.shape
    for y in range(height):
        for x in range(width - 1):
            i, j = quantized[y, x], quantized[y, x + 1]
            matrix[i, j] += 1
            matrix[j, i] += 1
    return matrix / matrix.sum()


def glcm_contrast(matrix):
    levels = matrix.shape[0]
    i, j = np.mgrid[0:levels, 0:levels]
    return float(np.sum(matrix * (i - j) ** 2))


def texture_map(gray, window, levels):
    # контраст GLCM у вікні = середнє (i - j)^2 по парах сусідів у цьому вікні
    quantized = quantize(gray, levels)
    right = shift(quantized, 0, 1, 1)
    return box_mean((quantized - right).astype(float) ** 2, window)


def apply(image, window=9, levels=16):
    contrast = texture_map(to_gray_array(image), window, levels)
    peak = contrast.max()
    if peak > 0:
        contrast = contrast / peak * 255.0
    return from_array(contrast)
