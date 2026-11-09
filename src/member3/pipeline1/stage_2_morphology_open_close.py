"""Binary morphological thinning implemented with the Zhang-Suen algorithm."""
from __future__ import annotations

import numpy as np
from PIL import Image


def _transitions(neighbours: tuple[int, ...]) -> int:
    """Count 0-to-1 transitions in a circular sequence of eight neighbours."""
    return sum(neighbours[index] == 0 and neighbours[(index + 1) % 8] == 1 for index in range(8))


def _removable_pixels(mask: np.ndarray, first_subiteration: bool) -> list[tuple[int, int]]:
    """Return foreground pixels removable in one Zhang-Suen subiteration."""
    removable: list[tuple[int, int]] = []
    height, width = mask.shape
    for row in range(1, height - 1):
        for column in range(1, width - 1):
            if not mask[row, column]:
                continue
            # P2 .. P9, clockwise from north to north-west.
            neighbours = (
                int(mask[row - 1, column]),
                int(mask[row - 1, column + 1]),
                int(mask[row, column + 1]),
                int(mask[row + 1, column + 1]),
                int(mask[row + 1, column]),
                int(mask[row + 1, column - 1]),
                int(mask[row, column - 1]),
                int(mask[row - 1, column - 1]),
            )
            neighbour_count = sum(neighbours)
            if not 2 <= neighbour_count <= 6 or _transitions(neighbours) != 1:
                continue
            north, east, south, west = neighbours[0], neighbours[2], neighbours[4], neighbours[6]
            if first_subiteration:
                preserves_connectivity = north * east * south == 0 and east * south * west == 0
            else:
                preserves_connectivity = north * east * west == 0 and north * south * west == 0
            if preserves_connectivity:
                removable.append((row, column))
    return removable


def apply(image: Image.Image, threshold: int = 128, max_iterations: int = 100) -> Image.Image:
    """Reduce bright binary regions to one-pixel-wide connected skeletons."""
    if not 0 <= threshold <= 255:
        raise ValueError("threshold must be between 0 and 255")
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")

    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    luminance = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    mask = luminance >= threshold
    for _ in range(max_iterations):
        first = _removable_pixels(mask, first_subiteration=True)
        if first:
            for row, column in first:
                mask[row, column] = False
        second = _removable_pixels(mask, first_subiteration=False)
        if second:
            for row, column in second:
                mask[row, column] = False
        if not first and not second:
            break
    binary = np.where(mask, 255, 0).astype(np.uint8)
    return Image.fromarray(np.repeat(binary[:, :, None], 3, axis=2))
