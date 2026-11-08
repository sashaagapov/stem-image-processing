"""Стадія 3 пайплайну 2 (member3): Колірна корекція — Яскравість/контраст.

TODO(member3): реалізувати вручну (без готових функцій бібліотек комп'ютерного
зору — Pillow/numpy можна використовувати лише для вводу/виводу та базової
математики, не для самого алгоритму). Підказка: new = clamp((old - 128) * contrast + 128 + brightness, 0, 255) для кожного каналу.

Поки що — заглушка: повертає зображення без змін, щоб pipeline_runner.py міг
виконати весь пайплайн end-to-end вже зараз (common/ тестується незалежно від
того, чи готові особисті стадії).
"""
from __future__ import annotations

from PIL import Image


def apply(image: Image.Image, brightness: int = 0, contrast: float = 1.1) -> Image.Image:
    # TODO(member3): замінити заглушку на реальну реалізацію Яскравість/контраст.
    return image.convert("RGB")
