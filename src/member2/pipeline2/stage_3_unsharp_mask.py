"""Стадія 3 пайплайну 2 (member2): Ретушування — Підвищення різкості (unsharp mask).

TODO(member2): реалізувати вручну (без готових функцій бібліотек комп'ютерного
зору — Pillow/numpy можна використовувати лише для вводу/виводу та базової
математики, не для самого алгоритму). Підказка: result = original + amount * (original - blurred); blurred можна взяти з власної gaussian-реалізації (стадія 1 цього ж пайплайну).

Поки що — заглушка: повертає зображення без змін, щоб pipeline_runner.py міг
виконати весь пайплайн end-to-end вже зараз (common/ тестується незалежно від
того, чи готові особисті стадії).
"""
from __future__ import annotations

from PIL import Image


def apply(image: Image.Image, amount: float = 1.0) -> Image.Image:
    # TODO(member2): замінити заглушку на реальну реалізацію Підвищення різкості (unsharp mask).
    return image.convert("RGB")
