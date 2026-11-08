"""Шаблон тесту для member3/pipeline2 (критерій C3: короткий тест на малому
зображенні з вручну порахованим очікуваним результатом).

ПОКИ ЩО тест перевіряє лише поведінку заглушки (apply() повертає зображення
без змін). Коли stage_1_hough_transform отримає реальну реалізацію — замініть
test_apply_matches_stub_behaviour нижче на справжню перевірку: невелике
зображення (3x3 чи 5x5), очікуваний результат, порахований вручну.
"""
from PIL import Image

from src.member3.pipeline2.stage_1_hough_transform import apply


def test_apply_matches_stub_behaviour() -> None:
    image = Image.new("RGB", (4, 4), (10, 20, 30))

    result = apply(image)

    assert result.size == image.size
    # TODO(member3): замінити на перевірку реального результату алгоритму,
    # коли stage_1_hough_transform.apply() перестане бути заглушкою.


def test_apply_accepts_small_image() -> None:
    tiny_image = Image.new("RGB", (3, 3), (5, 5, 5))

    result = apply(tiny_image)

    assert result.size == (3, 3)
