"""Тести пайплайну 1 (member1): Sobel -> вирівнювання гістограми -> геометричне
перетворення. Критерій C3: для кожної стадії — очікуваний результат, порахований
вручну на малому зображенні."""
import numpy as np
from PIL import Image

from src.member1.pipeline1.stage_1_sobel import apply as sobel_apply
from src.member1.pipeline1.stage_2_histogram_equalization import apply as equalize_apply
from src.member1.pipeline1.stage_3_geometric_transform import apply as transform_apply


def _solid_rows_image(column_values: list[int], rows: int = 5) -> Image.Image:
    row = np.array(column_values, dtype=np.uint8)
    array = np.tile(row, (rows, 1))
    rgb = np.stack([array, array, array], axis=-1)
    return Image.fromarray(rgb, "RGB")


# --- stage 1: Sobel ---------------------------------------------------------


def test_sobel_detects_vertical_edge() -> None:
    # Вертикальна межа між колонками 2 і 3 (50 -> 200).
    image = _solid_rows_image([50, 50, 50, 200, 200])

    result = sobel_apply(image, threshold=100)

    # Вручну: Gx у точці (row=2,col=2) = -1*50+0*50+1*200 (x2 для середнього ряду,
    # x1 для крайніх) = 150+300+150 = 600; Gy=0 (немає горизонтальної межі).
    # magnitude=600 >= threshold(100) -> білий піксель.
    assert result.getpixel((2, 2)) == (255, 255, 255)


def test_sobel_flat_region_has_no_edge() -> None:
    image = _solid_rows_image([50, 50, 50, 200, 200])

    result = sobel_apply(image, threshold=100)

    # Точка (row=2,col=0) лежить у рівній ділянці (сусідні значення теж 50) -> градієнт 0.
    assert result.getpixel((0, 2)) == (0, 0, 0)


def test_sobel_accepts_small_image() -> None:
    tiny_image = Image.new("RGB", (3, 3), (10, 10, 10))

    result = sobel_apply(tiny_image)

    assert result.size == (3, 3)


# --- stage 2: histogram equalization ----------------------------------------


def test_histogram_equalization_matches_hand_computed_values() -> None:
    # 2x2 зображення, значення (однакові по R/G/B) 10,10 / 20,30. N=4.
    array = np.array(
        [[[10, 10, 10], [10, 10, 10]], [[20, 20, 20], [30, 30, 30]]],
        dtype=np.uint8,
    )
    image = Image.fromarray(array, "RGB")

    result = equalize_apply(image)

    # cdf(10)=2, cdf(20)=3, cdf(30)=4; new = round(255*cdf/4):
    # round(127.5)=128 (банківське округлення), round(191.25)=191, round(255)=255.
    assert result.getpixel((0, 0)) == (128, 128, 128)
    assert result.getpixel((1, 0)) == (128, 128, 128)
    assert result.getpixel((0, 1)) == (191, 191, 191)
    assert result.getpixel((1, 1)) == (255, 255, 255)


def test_histogram_equalization_accepts_small_image() -> None:
    tiny_image = Image.new("RGB", (3, 3), (5, 5, 5))

    result = equalize_apply(tiny_image)

    assert result.size == (3, 3)


# --- stage 3: geometric transform -------------------------------------------


def test_geometric_transform_180_degrees_flips_image() -> None:
    # 3x3 зображення з унікальними значеннями 1..9.
    array = np.array(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        dtype=np.uint8,
    )
    rgb = np.stack([array, array, array], axis=-1)
    image = Image.fromarray(rgb, "RGB")

    result = transform_apply(image, angle_degrees=180, scale=1.0)

    # Поворот на 180 навколо центру (1,1) непарного зображення == точне дзеркало.
    expected = np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]], dtype=np.uint8)
    result_gray = np.asarray(result)[:, :, 0]
    assert np.array_equal(result_gray, expected)


def test_geometric_transform_accepts_small_image() -> None:
    tiny_image = Image.new("RGB", (3, 3), (7, 7, 7))

    result = transform_apply(tiny_image, angle_degrees=15, scale=1.0)

    assert result.size == (3, 3)
