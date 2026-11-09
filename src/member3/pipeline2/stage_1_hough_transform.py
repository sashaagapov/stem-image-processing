"""Straight-line Hough transform with a manually built accumulator."""
from __future__ import annotations

import numpy as np
from PIL import Image


def apply(image: Image.Image, edge_threshold: int = 100, theta_steps: int = 180, vote_threshold: int = 30, max_lines: int = 12) -> Image.Image:
    """Overlay locally maximal accumulator peaks as red straight lines."""
    if not 0 <= edge_threshold <= 510:
        raise ValueError("edge_threshold must be between 0 and 510")
    if theta_steps < 2:
        raise ValueError("theta_steps must be at least 2")
    if vote_threshold < 1 or max_lines < 1:
        raise ValueError("vote_threshold and max_lines must be positive")
    source = np.asarray(image.convert("RGB"), dtype=np.uint8)
    gray = 0.299 * source[:, :, 0] + 0.587 * source[:, :, 1] + 0.114 * source[:, :, 2]
    padded = np.pad(gray, 1, mode="edge")
    gradient_x = padded[1:-1, 2:] - padded[1:-1, :-2]
    gradient_y = padded[2:, 1:-1] - padded[:-2, 1:-1]
    edge_rows, edge_columns = np.nonzero(np.abs(gradient_x) + np.abs(gradient_y) >= edge_threshold)
    if edge_rows.size == 0:
        return Image.fromarray(source)
    height, width = gray.shape
    rho_limit = int(np.ceil(np.hypot(height - 1, width - 1)))
    thetas = np.linspace(-np.pi / 2, np.pi / 2, theta_steps, endpoint=False)
    accumulator = np.zeros((2 * rho_limit + 1, theta_steps), dtype=np.int32)
    for theta_index, theta in enumerate(thetas):
        rhos = np.rint(edge_columns * np.cos(theta) + edge_rows * np.sin(theta)).astype(int)
        accumulator[:, theta_index] = np.bincount(rhos + rho_limit, minlength=2 * rho_limit + 1)
    candidates: list[tuple[int, int, int]] = []
    for rho_index in range(1, accumulator.shape[0] - 1):
        for theta_index in range(1, theta_steps - 1):
            votes = int(accumulator[rho_index, theta_index])
            neighbourhood = accumulator[rho_index - 1 : rho_index + 2, theta_index - 1 : theta_index + 2]
            if votes >= vote_threshold and votes == int(np.max(neighbourhood)):
                candidates.append((votes, rho_index, theta_index))
    candidates.sort(reverse=True)
    result = source.copy()
    rows, columns = np.indices((height, width))
    for _votes, rho_index, theta_index in candidates[:max_lines]:
        rho = rho_index - rho_limit
        theta = thetas[theta_index]
        line = np.abs(columns * np.cos(theta) + rows * np.sin(theta) - rho) <= 0.5
        result[line] = (255, 0, 0)
    return Image.fromarray(result)
