"""Стадія 3 пайплайну 2 (member2): Ретушування — Підвищення різкості (unsharp mask).

    result = original + amount * (original - blurred)

`original - blurred` — це лише дрібні деталі й межі (те, що розмиття
прибрало). Додаючи їх назад із підсиленням `amount`, робимо межі різкішими.
Після карти текстур (стадія 2) це чіткіше відділяє ділянки з різною текстурою
(ліс / поле / забудова). Розмиття — власний Gaussian blur із пайплайну 1.

Параметри (config/pipelines.yaml):
    amount — сила підсилення (0 — без змін);
    sigma  — ширина розмиття: яких розмірів деталі підсилюються.
"""
from __future__ import annotations

from PIL import Image

from src.member2.image_ops import from_array, to_rgb_array
from src.member2.pipeline1.stage_1_gaussian_filter import gaussian_blur_array


def apply(image: Image.Image, amount: float = 1.0, sigma: float = 1.0, radius: int | None = None) -> Image.Image:
    original = to_rgb_array(image)
    blurred = gaussian_blur_array(original, sigma, radius)
    return from_array(original + amount * (original - blurred))
