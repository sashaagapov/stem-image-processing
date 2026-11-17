"""Межові випадки завантаження зображень (критерій C2): відсутній,
пошкоджений і дуже малий/монохромний файл не повинні давати сирий
traceback — лише зрозумілу помилку ImageLoadError або коректний результат."""
from pathlib import Path

import pytest
from PIL import Image

from common.io import ImageLoadError, open_image, save_image


def test_open_image_missing_file_raises_clear_error(tmp_path: Path) -> None:
    missing_path = tmp_path / "does_not_exist.png"

    with pytest.raises(ImageLoadError, match="Файл не знайдено"):
        open_image(missing_path)


def test_open_image_corrupted_file_raises_clear_error(tmp_path: Path) -> None:
    corrupted_path = tmp_path / "corrupted.png"
    corrupted_path.write_bytes(b"this is not image data at all")

    with pytest.raises(ImageLoadError, match="пошкоджений файл"):
        open_image(corrupted_path)


def test_open_image_accepts_tiny_image(tmp_path: Path) -> None:
    tiny_path = tmp_path / "tiny.png"
    save_image(Image.new("RGB", (3, 3), (10, 20, 30)), tiny_path)

    result = open_image(tiny_path)

    assert result.size == (3, 3)


def test_open_image_accepts_grayscale_image(tmp_path: Path) -> None:
    grayscale_path = tmp_path / "grayscale.png"
    save_image(Image.new("L", (16, 16), 128), grayscale_path)

    result = open_image(grayscale_path)
    # Пайплайни очікують RGB на вході — підтверджуємо, що .convert("RGB")
    # коректно обробляє одноканальне зображення без втрати розміру.
    rgb_result = result.convert("RGB")

    assert result.size == (16, 16)
    assert rgb_result.getpixel((0, 0)) == (128, 128, 128)
