"""Тести пайплайну 2 (member1): колірна корекція (баланс білого + HSV) -> Otsu ->
статистики. Критерій C3: для кожної стадії — очікуваний результат, порахований
вручну на малому зображенні."""
import math

import numpy as np
import pytest
from PIL import Image

from src.member1.pipeline2.stage_1_color_correction import (
    apply as color_correction_apply,
    gray_world_white_balance,
    hsv_to_rgb,
    rgb_to_hsv,
)
from src.member1.pipeline2.stage_2_otsu_segmentation import apply as otsu_apply, find_otsu_threshold
from src.member1.pipeline2.stage_3_statistics import apply as statistics_apply, calculate_statistics


# --- stage 1: color correction (white balance + HSV) ------------------------


def test_gray_world_white_balance_single_pixel() -> None:
    # 1 піксель (90,60,30): сіре середнє = (90+60+30)/3 = 60.
    # Масштаб каналів: R*60/90=60, G*60/60=60, B*60/30=60 -> усі канали зрівнюються.
    array = np.array([[[90.0, 60.0, 30.0]]])

    result = gray_world_white_balance(array)

    assert result[0, 0].tolist() == pytest.approx([60.0, 60.0, 60.0])


def test_rgb_to_hsv_known_colors() -> None:
    array = np.array([[255.0, 0.0, 0.0], [0.0, 255.0, 0.0], [100.0, 100.0, 100.0]])

    hue, saturation, value = rgb_to_hsv(array)

    # Чистий червоний: H=0, S=1, V=255. Чистий зелений: H=120, S=1, V=255.
    # Сірий (R=G=B): насиченість 0, V=100 (відтінок при S=0 не має значення).
    assert hue[0] == pytest.approx(0.0)
    assert saturation[0] == pytest.approx(1.0)
    assert value[0] == pytest.approx(255.0)

    assert hue[1] == pytest.approx(120.0)
    assert saturation[1] == pytest.approx(1.0)

    assert saturation[2] == pytest.approx(0.0)
    assert value[2] == pytest.approx(100.0)


def test_hsv_to_rgb_known_colors() -> None:
    hue = np.array([0.0, 120.0])
    saturation = np.array([1.0, 1.0])
    value = np.array([255.0, 255.0])

    rgb = hsv_to_rgb(hue, saturation, value)

    assert rgb[0].tolist() == pytest.approx([255.0, 0.0, 0.0])
    assert rgb[1].tolist() == pytest.approx([0.0, 255.0, 0.0])


def test_color_correction_apply_whitens_single_pixel() -> None:
    # (90,60,30) -> баланс білого дає точно сірий (60,60,60) -> S=0,
    # тож підвищення насиченості (SATURATION_BOOST) нічого не змінює.
    image = Image.new("RGB", (1, 1), (90, 60, 30))

    result = color_correction_apply(image)

    assert result.getpixel((0, 0)) == (60, 60, 60)


def test_color_correction_accepts_small_image() -> None:
    tiny_image = Image.new("RGB", (3, 3), (5, 5, 5))

    result = color_correction_apply(tiny_image)

    assert result.size == (3, 3)


# --- stage 2: Otsu segmentation ----------------------------------------------


def test_find_otsu_threshold_separates_two_clusters() -> None:
    # Два чіткі кластери значень: 10 (2 пікселі) і 200 (2 пікселі).
    # Між 10 і 199 міжкласова дисперсія однакова (w0=w1=0.5, mu0=10, mu1=200) -
    # береться найменший поріг з максимумом -> 10.
    gray = np.array([[10, 10, 200, 200]], dtype=np.float64)

    threshold = find_otsu_threshold(gray)

    assert threshold == 10


def test_otsu_apply_segments_known_clusters() -> None:
    array = np.array([[[10, 10, 10], [10, 10, 10], [200, 200, 200], [200, 200, 200]]], dtype=np.uint8)
    image = Image.fromarray(array, "RGB")

    result = otsu_apply(image)

    assert result.getpixel((0, 0)) == (0, 0, 0)
    assert result.getpixel((1, 0)) == (0, 0, 0)
    assert result.getpixel((2, 0)) == (255, 255, 255)
    assert result.getpixel((3, 0)) == (255, 255, 255)


def test_otsu_accepts_small_image() -> None:
    tiny_image = Image.new("RGB", (3, 3), (42, 42, 42))

    result = otsu_apply(tiny_image)

    assert result.size == (3, 3)


# --- stage 3: statistics ------------------------------------------------------


def test_calculate_statistics_matches_hand_computed_values() -> None:
    # 1x3 зображення, значення (однакові по R/G/B) 50, 100, 150 -> N=3.
    array = np.array([[[50, 50, 50], [100, 100, 100], [150, 150, 150]]], dtype=np.uint8)
    image = Image.fromarray(array, "RGB")

    stats = calculate_statistics(image)

    # mean = (50+100+150)/3 = 100.
    # variance = ((50-100)^2+(100-100)^2+(150-100)^2)/3 = (2500+0+2500)/3 = 5000/3.
    # entropy = -3*(1/3 * log2(1/3)) = log2(3).
    assert stats["mean"] == pytest.approx(100.0)
    assert stats["variance"] == pytest.approx(5000 / 3)
    assert stats["entropy"] == pytest.approx(math.log2(3))


def test_statistics_apply_preserves_size() -> None:
    tiny_image = Image.new("RGB", (5, 5), (20, 20, 20))

    result = statistics_apply(tiny_image)

    assert result.size == (5, 5)
