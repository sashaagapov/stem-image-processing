"""Тести member2/pipeline1: білатеральний фільтр -> бінарна морфологія -> Лаплас
(критерії C1, C2, C3).

Очікувані значення пораховані вручну на зображеннях 3x3 і 5x5; розрахунок
наведено в коментарях. Межі доповнюються повторенням крайніх пікселів.
"""
from pathlib import Path

import numpy as np
from PIL import Image

from common.metrics import calculate_psnr, calculate_ssim
from common.pipeline_runner import RESULTS_DIR, run_pipeline
from src.member2.image_ops import gaussian_blur
from src.member2.pipeline1 import stage_1_bilateral_filter, stage_2_morphology, stage_3_laplacian
from src.member2.pipeline1.stage_2_morphology import disk
from tests.member2.scene import add_noise, as_image, make_scene


def gray_image(values) -> Image.Image:
    array = np.array(values, dtype=np.uint8)
    return Image.fromarray(np.stack([array] * 3, axis=2))


def channel(image: Image.Image) -> np.ndarray:
    return np.asarray(image)[..., 0].astype(int)


IMPULSE = [[0, 0, 0], [0, 100, 0], [0, 0, 0]]


# ---------- стадія 1: білатеральний фільтр ----------

def test_bilateral_with_tiny_sigma_range_keeps_image() -> None:
    # sigma_range дуже мала: вага сусіда з іншим кольором exp(-30000 / 2e-6) = 0,
    # тож кожен піксель усереднюється лише з собі рівними -> зображення не змінюється.
    result = channel(stage_1_bilateral_filter.apply(gray_image(IMPULSE), sigma_spatial=1, sigma_range=0.001, radius=1))
    assert result.tolist() == IMPULSE


def test_bilateral_with_huge_sigma_range_equals_gaussian_by_hand() -> None:
    # sigma_range дуже велика: кольорова вага = 1, лишається лише просторова
    # e^-(dx^2+dy^2)/2 -> це гаусове ядро sigma = 1 з сумою 4.89764.
    # центр 100/4.898 = 20.42 -> 20, сторона 100*e^-0.5/4.898 = 12.38 -> 12,
    # кут 100*e^-1/4.898 = 7.51 -> 8.
    result = channel(stage_1_bilateral_filter.apply(gray_image(IMPULSE), sigma_spatial=1, sigma_range=1e9, radius=1))
    assert result.tolist() == [[8, 12, 8], [12, 20, 12], [8, 12, 8]]


def test_bilateral_keeps_edge_that_gaussian_blurs() -> None:
    # Межа 0 | 200 ("поле | дах"). Різниця кольору через межу 200 у кожному
    # каналі -> вага exp(-3 * 200^2 / (2 * 30^2)) ~ 1e-29, через межу нічого не
    # змішується: зображення не змінюється. Гаусс з тим самим радіусом межу розмиває.
    values = np.zeros((4, 6), dtype=np.uint8)
    values[:, 3:] = 200
    result = channel(stage_1_bilateral_filter.apply(gray_image(values), sigma_spatial=2, sigma_range=30))
    assert result.tolist() == values.tolist()
    blurred = np.round(gaussian_blur(values.astype(float), sigma=2))
    assert 0 < blurred[0, 2] < 200


def test_bilateral_scenario_c1_denoising() -> None:
    # Еталонний сценарій C1: синтетичний аерознімок + гаусів шум sigma = 25.
    # Зашумлений: PSNR ~ 20.3 дБ, SSIM ~ 0.46. Поріг ТЗ: PSNR >= 25 дБ, SSIM >= 0.7.
    clean = as_image(make_scene())
    noisy = as_image(add_noise(make_scene()))
    result = stage_1_bilateral_filter.apply(noisy, sigma_spatial=2, sigma_range=50)
    assert calculate_psnr(result, clean) >= 25
    assert calculate_ssim(result, clean) >= 0.7


# ---------- стадія 2: бінарна морфологія ----------

def test_disk_radius1_is_cross() -> None:
    assert disk(1).astype(int).tolist() == [[0, 1, 0], [1, 1, 1], [0, 1, 0]]


SQUARE = np.zeros((5, 5), dtype=np.uint8)
SQUARE[1:4, 1:4] = 200  # середнє = 9 * 200 / 25 = 72 -> маска = квадрат 3x3


def test_binary_erosion_by_hand() -> None:
    # Диск радіуса 1 = хрестик. Піксель лишається, лише якщо він і 4 сусіди
    # усі в квадраті 3x3 -> тільки центр (2, 2).
    result = channel(stage_2_morphology.apply(gray_image(SQUARE), operation="erosion", radius=1))
    expected = np.zeros((5, 5), dtype=int)
    expected[2, 2] = 255
    assert result.tolist() == expected.tolist()


def test_binary_dilation_by_hand() -> None:
    # Хрестик "дотягується" на 1 піксель уверх/вниз/вліво/вправо, але не по
    # діагоналі -> квадрат 3x3 стає 5x5 без чотирьох кутів.
    result = channel(stage_2_morphology.apply(gray_image(SQUARE), operation="dilation", radius=1))
    expected = np.full((5, 5), 255)
    expected[0, 0] = expected[0, 4] = expected[4, 0] = expected[4, 4] = 0
    assert result.tolist() == expected.tolist()


def test_binary_erosion_removes_single_bright_pixel() -> None:
    values = np.zeros((5, 5), dtype=np.uint8)
    values[2, 2] = 255
    assert channel(stage_2_morphology.apply(gray_image(values), operation="erosion")).max() == 0


def test_invert_makes_dark_objects_white() -> None:
    # Темна "водойма" 3x3 на світлому фоні: середнє = (16*200)/25 = 128,
    # з invert об'єктом стає те, що темніше -> маска = квадрат.
    values = np.full((5, 5), 200, dtype=np.uint8)
    values[1:4, 1:4] = 0
    result = channel(stage_2_morphology.apply(gray_image(values), operation="erosion", radius=1, invert=True))
    assert result[2, 2] == 255 and result.sum() == 255


# ---------- стадія 3: Лаплас ----------

def test_laplacian_of_impulse_by_hand() -> None:
    # 3x3, центр 10. Ядро [[0,1,0],[1,-4,1],[0,1,0]]:
    #   центр: -4 * 10 = -40 -> 40;  сторони: сусід-центр 10 -> 10;  кути: 0.
    # Нормування на максимум 40: 40 -> 255, 10 -> 63.75 -> 64.
    image = gray_image([[0, 0, 0], [0, 10, 0], [0, 0, 0]])
    result = channel(stage_3_laplacian.apply(image, neighbors=4))
    assert result.tolist() == [[0, 64, 0], [64, 255, 64], [0, 64, 0]]


def test_laplacian_of_flat_image_is_zero() -> None:
    assert channel(stage_3_laplacian.apply(Image.new("RGB", (4, 4), (90, 90, 90)))).max() == 0


# ---------- увесь пайплайн і межові випадки (C1, C2) ----------

def test_full_pipeline_finds_vertical_border(tmp_path: Path) -> None:
    # Еталон з відомою відповіддю: ліва половина темна (поле), права світла
    # (дах). Контур має бути біля межі (стовпці 9-11), а не на рівних ділянках.
    values = np.zeros((20, 20), dtype=np.uint8)
    values[:, 10:] = 200
    path = tmp_path / "border.png"
    gray_image(values).save(path)
    run_pipeline("member2", "pipeline1", path)
    final = channel(Image.open(RESULTS_DIR / "member2" / "pipeline1" / "border" / "final.png"))
    assert final[:, 8:13].max() == 255
    assert final[:, :5].max() == 0 and final[:, -5:].max() == 0


def test_stages_accept_tiny_and_grayscale_images() -> None:
    for image in (Image.new("L", (3, 3), 50), Image.new("RGB", (1, 1), (1, 2, 3)), Image.new("RGBA", (2, 5))):
        out = stage_3_laplacian.apply(stage_2_morphology.apply(stage_1_bilateral_filter.apply(image)))
        assert out.size == image.size and out.mode == "RGB"
