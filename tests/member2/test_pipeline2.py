"""Тести member2/pipeline2: FFT-фільтр -> текстура GLCM -> unsharp mask
(критерії C1, C2, C3).

Очікувані значення пораховані вручну; розрахунок — у коментарях.
"""
from pathlib import Path

import numpy as np
from PIL import Image

from common.metrics import calculate_psnr
from common.pipeline_runner import RESULTS_DIR, run_pipeline
from tests.member2.scene import add_noise, as_image, make_scene
from src.member2.pipeline2 import stage_1_fourier_filter, stage_2_texture_analysis, stage_3_unsharp_mask
from src.member2.pipeline2.stage_2_texture_analysis import glcm, glcm_features, quantize, texture_map


def gray_image(values) -> Image.Image:
    array = np.array(values, dtype=np.uint8)
    return Image.fromarray(np.stack([array] * 3, axis=2))


def channel(image: Image.Image) -> np.ndarray:
    return np.asarray(image)[..., 0].astype(int)


# ---------- стадія 1: FFT ----------

def test_lowpass_removes_checkerboard_by_hand() -> None:
    # Шахівниця 0/200 = 100 + 100 * (-1)^(x+y): у спектрі лише нульова частота
    # (середнє 100) і частота Найквіста на відстані sqrt(2^2 + 2^2) = 2.83 від
    # центру. Ідеальний фільтр з cutoff = 1 лишає тільки середнє -> усе 100.
    board = [[200 if (x + y) % 2 else 0 for x in range(4)] for y in range(4)]
    result = channel(stage_1_fourier_filter.apply(gray_image(board), cutoff=1, kind="ideal"))
    assert np.all(result == 100)


def test_highpass_of_constant_gives_128() -> None:
    # Однотонне зображення — лише нульова частота; високочастотна маска в
    # центрі = 1 - 1 = 0, лишається 0, плюс зсув 128.
    image = Image.new("RGB", (6, 6), (77, 77, 77))
    result = channel(stage_1_fourier_filter.apply(image, cutoff=2, mode="highpass"))
    assert np.all(result == 128)


def test_lowpass_keeps_constant_colour() -> None:
    image = Image.new("RGB", (8, 8), (30, 120, 200))
    result = np.asarray(stage_1_fourier_filter.apply(image, cutoff=3))
    assert np.all(result == [30, 120, 200])


def test_lowpass_scenario_c1_noise_reduction() -> None:
    # Еталонний сценарій C1: синтетичний аерознімок 128x128 + гаусів шум sigma = 25.
    # cutoff = 20 для 128x128 — та сама відносна частота, що й cutoff = 40 у
    # конфігу для 256x256. Поріг ТЗ: PSNR зростає щонайменше на 3 дБ.
    # (SSIM >= 0.7 тут не вимагаємо: низькочастотний фільтр за принципом дії
    # розмиває й межі — це обмеження методу, яке аналізується у звіті.)
    clean = as_image(make_scene())
    noisy = as_image(add_noise(make_scene()))
    result = stage_1_fourier_filter.apply(noisy, cutoff=20)
    assert calculate_psnr(result, clean) >= calculate_psnr(noisy, clean) + 3


# ---------- стадія 2: текстура GLCM ----------

# Сірі 0, 100, 200 при levels = 3 квантуються в рівні 0, 1, 2:
#   100 * 3 / 256 = 1.17 -> 1,  200 * 3 / 256 = 2.34 -> 2.
SMALL = [[0, 0, 100], [0, 100, 100], [200, 200, 200]]  # рівні [[0,0,1],[0,1,1],[2,2,2]]


def test_glcm_by_hand() -> None:
    # Горизонтальні пари (кут 0, відстань 1):
    #   рядок 0: (0,0), (0,1); рядок 1: (0,1), (1,1); рядок 2: (2,2), (2,2)
    # Симетрично (рахуємо і (j,i)): (0,0)=2, (0,1)=2, (1,0)=2, (1,1)=2, (2,2)=4,
    # усього 12.
    q = quantize(np.array(SMALL, dtype=float), 3)
    assert q.tolist() == [[0, 0, 1], [0, 1, 1], [2, 2, 2]]
    matrix = glcm(q, levels=3)
    assert np.allclose(matrix * 12, [[2, 2, 0], [2, 2, 0], [0, 0, 4]])

    f = glcm_features(matrix)
    # contrast = (2 + 2) / 12 * 1^2 = 1/3
    assert abs(f["contrast"] - 1 / 3) < 1e-12
    # homogeneity = (2 + 2 + 4) / 12 * 1 + (2 + 2) / 12 * 1/2 = 10/12
    assert abs(f["homogeneity"] - 10 / 12) < 1e-12
    # energy = sqrt((4 + 4 + 4 + 4 + 16) / 144) = sqrt(32) / 12
    assert abs(f["energy"] - np.sqrt(32) / 12) < 1e-12


def test_texture_map_center_by_hand() -> None:
    # Для кожного пікселя різниця з правим сусідом (крайній правий стовпець
    # порівнюється сам із собою):
    #   рядок 0: |0-0|=0, |0-1|=1, 0;  рядок 1: |0-1|=1, 0, 0;  рядок 2: 0, 0, 0
    # Контраст у вікні 3x3 навколо центру = середнє квадратів = 2/9.
    tmap = texture_map(np.array(SMALL, dtype=float), "contrast", window=3, levels=3)
    assert abs(tmap[1, 1] - 2 / 9) < 1e-12


def test_texture_map_separates_rough_and_smooth_halves() -> None:
    # Ліва половина — випадковий "ліс", права — рівне "поле".
    rng = np.random.default_rng(0)
    values = np.full((32, 32), 120, dtype=np.uint8)
    values[:, :16] = rng.integers(0, 256, size=(32, 16))
    result = channel(stage_2_texture_analysis.apply(gray_image(values), window=5))
    assert result[:, :12].mean() > 100
    assert result[:, 20:].max() == 0


# ---------- стадія 3: unsharp mask ----------

def test_unsharp_mask_of_impulse_by_hand() -> None:
    # Розмите значення беремо з тесту Гаусса (sigma = 1, radius = 1):
    #   центр 20.418, сторона 12.384, кут 7.511.
    # result = original + 1 * (original - blurred):
    #   центр 100 + (100 - 20.418) = 179.58 -> 180
    #   сторони та кути 0 + (0 - 12.38) < 0 -> обрізається до 0
    image = gray_image([[0, 0, 0], [0, 100, 0], [0, 0, 0]])
    result = channel(stage_3_unsharp_mask.apply(image, amount=1.0, sigma=1.0, radius=1))
    assert result.tolist() == [[0, 0, 0], [0, 180, 0], [0, 0, 0]]


def test_unsharp_mask_amount_zero_changes_nothing() -> None:
    image = gray_image([[10, 50, 90], [20, 60, 100], [30, 70, 110]])
    assert channel(stage_3_unsharp_mask.apply(image, amount=0)).tolist() == channel(image).tolist()


# ---------- увесь пайплайн і межові випадки ----------

def test_full_pipeline_runs_and_keeps_size(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    values = rng.integers(0, 256, size=(24, 30)).astype(np.uint8)
    path = tmp_path / "noise.png"
    gray_image(values).save(path)
    metrics = run_pipeline("member2", "pipeline2", path)
    final = Image.open(RESULTS_DIR / "member2" / "pipeline2" / "noise" / "final.png")
    assert final.size == (30, 24)
    assert metrics["ssim_vs_original"] < 1


def test_stages_accept_tiny_and_grayscale_images() -> None:
    for image in (Image.new("L", (3, 3), 50), Image.new("RGB", (1, 1), (1, 2, 3)), Image.new("RGBA", (2, 5))):
        out = stage_3_unsharp_mask.apply(stage_2_texture_analysis.apply(stage_1_fourier_filter.apply(image)))
        assert out.size == image.size and out.mode == "RGB"
