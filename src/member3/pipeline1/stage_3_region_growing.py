"""Region-growing segmentation using a running mean of each region."""
from __future__ import annotations

import numpy as np
from PIL import Image


def apply(image: Image.Image, seed_threshold: int = 20, connectivity: int = 4) -> Image.Image:
    """Partition connected pixels with similar luminance and render each mean shade."""
    if not 0 <= seed_threshold <= 255:
        raise ValueError("seed_threshold must be between 0 and 255")
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    height, width = gray.shape
    assigned = np.zeros((height, width), dtype=bool)
    output = np.zeros((height, width), dtype=np.uint8)
    neighbours = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if connectivity == 8:
        neighbours += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    for start_row in range(height):
        for start_column in range(width):
            if assigned[start_row, start_column]:
                continue
            region = [(start_row, start_column)]
            assigned[start_row, start_column] = True
            value_sum = float(gray[start_row, start_column])
            cursor = 0
            while cursor < len(region):
                row, column = region[cursor]
                cursor += 1
                mean = value_sum / len(region)
                for row_delta, column_delta in neighbours:
                    next_row, next_column = row + row_delta, column + column_delta
                    if not (0 <= next_row < height and 0 <= next_column < width) or assigned[next_row, next_column]:
                        continue
                    if abs(float(gray[next_row, next_column]) - mean) <= seed_threshold:
                        assigned[next_row, next_column] = True
                        region.append((next_row, next_column))
                        value_sum += float(gray[next_row, next_column])
            region_value = np.uint8(np.rint(value_sum / len(region)))
            for row, column in region:
                output[row, column] = region_value
    return Image.fromarray(np.repeat(output[:, :, None], 3, axis=2))
