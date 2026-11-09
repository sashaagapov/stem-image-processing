"""Консольне меню для запуску всіх 6 пайплайнів команди.

Запуск: кнопка «Пуск» у Visual Studio (Startup Item — common/menu.py) або
    .venv\\Scripts\\python.exe common\\menu.py
Список пайплайнів і назви стадій беруться з config/pipelines.yaml.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
VENV_PYTHON = ROOT_DIR / ".venv" / "Scripts" / "python.exe"


def _restart_in_venv() -> None:
    """Якщо меню запустили не тим Python, перезапускаємо його через .venv проєкту."""
    if VENV_PYTHON.exists() and Path(sys.prefix).resolve() != VENV_PYTHON.parents[1].resolve():
        raise SystemExit(subprocess.call([str(VENV_PYTHON), str(Path(__file__).resolve())]))


if __name__ == "__main__":
    _restart_in_venv()

sys.path.insert(0, str(ROOT_DIR))  # щоб працювали імпорти common.* і src.*

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from common.io import ImageLoadError, open_image  # noqa: E402
from common.pipeline_runner import RESULTS_DIR, load_config, run_pipeline  # noqa: E402

SAMPLES_DIR = ROOT_DIR / "data" / "samples"
DEFAULT_IMAGE = "river_02.jpg"
MEMBER_NAMES = {
    "member1": "Олександр Агапов",
    "member2": "Тимофій Стукало",
    "member3": "Микола Ковальський",
}


def list_pipelines(config: dict) -> list[dict]:
    """Усі пайплайни з конфігу по порядку: member1/pipeline1 -> №1, ..., member3/pipeline2 -> №6."""
    pipelines = []
    for member, member_pipelines in config["members"].items():
        for key, pipeline in member_pipelines.items():
            stages = [part.strip() for part in pipeline.get("description", "").split("->")]
            pipelines.append({
                "number": len(pipelines) + 1,
                "member": member,
                "key": key,
                "author": MEMBER_NAMES.get(member, member),
                "stages": [f"{index}. {name}" for index, name in enumerate(stages, start=1)],
            })
    return pipelines


def resolve_image_path(text: str, default: str = DEFAULT_IMAGE) -> Path | None:
    """Перетворює введений текст на шлях до файлу.

    Порожній рядок — знімок за замовчуванням; лапки навколо шляху прибираються;
    якщо файлу немає, шукаємо таке ім'я в data/samples (можна без .jpg).
    """
    text = text.strip().strip('"').strip("'") or default
    for candidate in (Path(text), SAMPLES_DIR / text, SAMPLES_DIR / f"{text}.jpg"):
        if candidate.is_file():
            return candidate
    return None


def build_comparison(original: Image.Image, stage_images: list[Image.Image],
                     captions: list[str], size: int = 256) -> Image.Image:
    """Складає оригінал і результати стадій в один ряд з підписами."""
    images = [original] + stage_images
    captions = ["Оригінал"] + captions
    gap, caption_height = 10, 30
    sheet = Image.new("RGB", (len(images) * (size + gap) + gap, size + caption_height + 2 * gap), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (image, caption) in enumerate(zip(images, captions)):
        x = gap + index * (size + gap)
        sheet.paste(image.convert("RGB").resize((size, size)), (x, gap))
        draw.text((x, gap + size + 6), caption, fill="black", font=_fitting_font(draw, caption, size))
    return sheet


def _fitting_font(draw: ImageDraw.ImageDraw, text: str, width: int):
    """Arial (має кирилицю) такого розміру, щоб підпис не виліз за ширину картинки."""
    for font_size in range(16, 7, -1):
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except OSError:
            return ImageFont.load_default()
        if draw.textlength(text, font=font) <= width:
            return font
    return font


def run_menu_pipeline(pipeline: dict) -> None:
    text = input(f"Шлях до знімка або ім'я з data\\samples (Enter — {DEFAULT_IMAGE}): ")
    image_path = resolve_image_path(text)
    if image_path is None:
        print("  Файл не знайдено. Перевірте шлях і спробуйте ще раз.\n")
        return

    print(f"\nПайплайн {pipeline['number']} ({pipeline['author']}) на {image_path.name}:")
    for stage in pipeline["stages"]:
        print(f"  {stage}")
    try:
        metrics = run_pipeline(pipeline["member"], pipeline["key"], image_path)
    except ImageLoadError as error:
        print(f"  Помилка: {error}\n")
        return

    print("\nМетрики відносно оригіналу:")
    print(f"  MSE  = {metrics['mse_vs_original']}")
    print(f"  PSNR = {metrics['psnr_vs_original']} дБ")
    print(f"  SSIM = {metrics['ssim_vs_original']}")

    output_dir = RESULTS_DIR / pipeline["member"] / pipeline["key"] / image_path.stem
    stage_images = [open_image(path) for path in sorted(output_dir.glob("stage_*.png"))]
    comparison = build_comparison(open_image(image_path), stage_images, pipeline["stages"])
    comparison_path = output_dir / "comparison.png"
    comparison.save(comparison_path)

    print(f"\nРезультати збережено в: {output_dir}")
    print("Відкриваю картинку «до / після»...\n")
    os.startfile(comparison_path)  # відкриває у стандартному переглядачі Windows


def run_tests() -> None:
    print("\nЗапуск усіх тестів...\n")
    subprocess.call([sys.executable, "-m", "pytest", str(ROOT_DIR / "tests"), "-q"], cwd=ROOT_DIR)
    print()


def main() -> None:
    os.chdir(ROOT_DIR)
    pipelines = list_pipelines(load_config())
    by_number = {str(p["number"]): p for p in pipelines}
    while True:
        print("=" * 72)
        print(" Аналіз супутникових знімків — STEM-проєкт")
        print("=" * 72)
        current_author = None
        for p in pipelines:
            if p["author"] != current_author:
                current_author = p["author"]
                print(f" {current_author}:")
            stage_names = " → ".join(stage.split(". ", 1)[1] for stage in p["stages"])
            print(f"   {p['number']}. {stage_names}")
        print(" T. Запустити тести")
        print(" 0. Вихід")
        answer = input("Ваш вибір: ").strip().upper()
        if answer in by_number:
            run_menu_pipeline(by_number[answer])
        elif answer in ("T", "Т"):  # латинська або кирилична
            run_tests()
        elif answer == "0":
            break
        else:
            print(f"  Введіть номер 1–{len(pipelines)}, T або 0.\n")


if __name__ == "__main__":
    main()
