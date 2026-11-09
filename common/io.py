from __future__ import annotations

from pathlib import Path

from PIL import Image, UnidentifiedImageError

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


class ImageLoadError(Exception):
    """Зрозуміла помилка завантаження зображення (критерій C2) — замість
    сирого traceback Pillow на відсутньому/пошкодженому файлі."""


def open_image(path: Path) -> Image.Image:
    if not path.exists():
        raise ImageLoadError(f"Файл не знайдено: {path}")
    try:
        # Копія дозволяє закрити файл одразу після читання.
        with Image.open(path) as image:
            loaded = image.copy()
            loaded.format = image.format
            return loaded
    except UnidentifiedImageError as error:
        raise ImageLoadError(
            f"Не вдалося розпізнати зображення у файлі {path} "
            "(пошкоджений файл або непідтримуваний формат)"
        ) from error


def save_image(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def list_dataset_images(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
