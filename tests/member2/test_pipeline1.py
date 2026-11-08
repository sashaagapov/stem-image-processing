"""Тести member2/pipeline1: Gaussian -> морфологія -> Лаплас (критерій C3).

Очікувані значення пораховані вручну на зображеннях 3x3 і 5x5; розрахунок
наведено в коментарях. Межі доповнюються повторенням крайніх пікселів.
"""
import math
from pathlib import Path

import numpy as np
from PIL import Image

from common.pipeline_runner import run_pipeline
from src.member2.image_ops import gaussian_kernel
from src.member2.pipeline1 import stage_1_gaussian_filter, stage_2_morphology, stage_3_laplacian


def gray_image(values) -> Image.Image:
    array = np.array(values, dtype=np.uint8)
    return Image.fromarray(np.stack([array] * 3, axis=2))


def channel(image: Image.Image) -> np.ndarray:
    return np.asarray(image)[..., 0].astype(int)


# ---------- стадія 1: Gaussian ----------

def test_gaussian_kernel_sigma1_radius1_by_hand() -> None:
    # ваги до нормування: центр e^0 = 1, сторони e^-0.5, кути e^-1
    edge, corner = math.exp(-0.5), math.exp(-1.0)
    total = 1 + 4 * edge + 4 * corner  # = 4.89764
    kernel = gaussian_kernel(sigma=1.0, radius=1)
    assert kernel.shape == (3, 3)
    assert abs(kernel.sum() - 1) < 1e-12
    assert abs(kernel[1, 1] - 1 / total) < 1e-12
    assert abs(kernel[0, 1] - edge / total) < 1e-12
    assert abs(kernel[0, 0] - corner / total) < 1e-12


def test_gaussian_blur_of_impulse_by_hand() -> None:
    # Імпульс 100 у центрі 3x3. Пікселі, яких центр не досягає, лишаються 0,
    # бо доповнення краю — теж нулі. Кожен піксель отримує 100 * вага
    # відповідної позиції ядра: центр 100/4.898 = 20.42 -> 20,
    # сторона 100*0.6065/4.898 = 12.38 -> 12, кут 100*0.3679/4.898 = 7.51 -> 8.
    image = gray_image([[0, 0, 0], [0, 100, 0], [0, 0, 0]])
    result = channel(stage_1_gaussian_filter.apply(image, sigma=1.0, radius=1))
    assert result.tolist() == [[8, 12, 8], [12, 20, 12], [8, 12, 8]]


def test_gaussian_keeps_constant_image_and_colour() -> None:
    image = Image.new("RGB", (4, 4), (10, 20, 30))
    result = stage_1_gaussian_filter.apply(image, sigma=2.0)
    assert np.all(np.asarray(result) == [10, 20, 30])


# ---------- стадія 2: морфологія ----------

def test_erosion_shrinks_square_by_hand() -> None:
    # 5x5, світлий квадрат 3x3 у центрі. Мінімум у вікні 3x3 дорівнює 200
    # лише там, де все вікно лежить усередині квадрата, — тільки центр.
    values = np.zeros((5, 5), dtype=np.uint8)
    values[1:4, 1:4] = 200
    result = channel(stage_2_morphology.apply(gray_image(values), window_size=3, operation="erosion"))
    expected = np.zeros((5, 5), dtype=int)
    expected[2, 2] = 200
    assert result.tolist() == expected.tolist()


def test_opening_removes_single_bright_pixel() -> None:
    # Одиночний світлий піксель менший за вікно: ерозія прибирає його,
    # дилатації вже нічого відновлювати -> усе 0.
    values = np.zeros((5, 5), dtype=np.uint8)
    values[2, 2] = 255
    result = channel(stage_2_morphology.apply(gray_image(values), window_size=3, operation="opening"))
    assert result.max() == 0


def test_closing_fills_single_dark_pixel() -> None:
    values = np.full((5, 5), 180, dtype=np.uint8)
    values[2, 2] = 0
    result = channel(stage_2_morphology.apply(gray_image(values), window_size=3, operation="closing"))
    assert np.all(result == 180)


# ---------- стадія 3: Лаплас ----------

def test_laplacian_of_impulse_by_hand() -> None:
    # 3x3, центр 10. Ядро [[0,1,0],[1,-4,1],[0,1,0]]:
    #   центр:   0+0+0+0 - 4*10 = -40 -> |40|
    #   сторони: один сусід — центр (10), решта 0 -> 10
    #   кути:    центр не є 4-сусідом -> 0
    # Нормування на максимум 40: 40 -> 255, 10 -> 63.75 -> 64.
    image = gray_image([[0, 0, 0], [0, 10, 0], [0, 0, 0]])
    result = channel(stage_3_laplacian.apply(image, neighbors=4))
    assert result.tolist() == [[0, 64, 0], [64, 255, 64], [0, 64, 0]]


def test_laplacian_of_flat_image_is_zero() -> None:
    result = channel(stage_3_laplacian.apply(Image.new("RGB", (4, 4), (90, 90, 90))))
    assert result.max() == 0


# ---------- увесь пайплайн і межові випадки ----------

def test_full_pipeline_finds_vertical_border(tmp_path: Path) -> None:
    # Ліва половина темна (поле), права світла (дах): контур має бути
    # біля межі, а не на рівних ділянках.
    values = np.zeros((20, 20), dtype=np.uint8)
    values[:, 10:] = 200
    path = tmp_path / "border.png"
    gray_image(values).save(path)
    run_pipeline("member2", "pipeline1", path)
    from common.pipeline_runner import RESULTS_DIR
    final = channel(Image.open(RESULTS_DIR / "member2" / "pipeline1" / "border" / "final.png"))
    assert final[:, 8:12].max() > 100
    assert final[:, :3].max() == 0 and final[:, -3:].max() == 0


def test_stages_accept_tiny_and_grayscale_images() -> None:
    for image in (Image.new("L", (3, 3), 50), Image.new("RGB", (1, 1), (1, 2, 3)), Image.new("RGBA", (2, 5))):
        out = stage_3_laplacian.apply(stage_2_morphology.apply(stage_1_gaussian_filter.apply(image)))
        assert out.size == image.size and out.mode == "RGB"
