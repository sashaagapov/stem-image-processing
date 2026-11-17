"""Extra tests for member3: parameter validation and edge cases of all six stages."""
import numpy as np
import pytest
from PIL import Image

from src.member3.pipeline1.stage_1_harris_corners import apply as harris
from src.member3.pipeline1.stage_2_morphology_open_close import apply as thinning
from src.member3.pipeline1.stage_3_region_growing import apply as region_growing
from src.member3.pipeline2.stage_1_hough_transform import apply as hough
from src.member3.pipeline2.stage_2_affine_transform import apply as affine
from src.member3.pipeline2.stage_3_brightness_contrast import apply as brightness_contrast


def _gray_image(rows: list[list[int]]) -> Image.Image:
    return Image.fromarray(np.array(rows, dtype=np.uint8)).convert("RGB")


def _values(image: Image.Image) -> list[list[int]]:
    return np.asarray(image)[:, :, 0].tolist()


# ---------- Pipeline 5, stage 1: Harris ----------

@pytest.mark.parametrize(
    "kwargs",
    [{"threshold": 1.5}, {"harris_k": 0.3}, {"window_size": 2}],
)
def test_harris_rejects_invalid_parameters(kwargs) -> None:
    with pytest.raises(ValueError):
        harris(Image.new("RGB", (5, 5)), **kwargs)


def test_harris_marks_the_corners_of_a_bright_square_red() -> None:
    rows = np.zeros((9, 9), dtype=np.uint8)
    rows[3:6, 3:6] = 255
    image = Image.fromarray(rows).convert("RGB")

    result = np.asarray(harris(image, threshold=0.01, harris_k=0.04, window_size=3))

    red = (result[:, :, 0] == 255) & (result[:, :, 1] == 0) & (result[:, :, 2] == 0)
    assert red.any()
    # The image corner is far from the square, so nothing is marked there.
    assert not red[0, 0]
    assert result.shape == (9, 9, 3)


# ---------- Pipeline 5, stage 2: Zhang-Suen thinning ----------

def test_thinning_rejects_invalid_parameters() -> None:
    with pytest.raises(ValueError):
        thinning(Image.new("RGB", (5, 5)), threshold=300)
    with pytest.raises(ValueError):
        thinning(Image.new("RGB", (5, 5)), max_iterations=0)


def test_thinning_keeps_an_all_black_image_black() -> None:
    result = thinning(Image.new("RGB", (5, 5), (0, 0, 0)), threshold=128, max_iterations=10)

    assert _values(result) == [[0] * 5] * 5


def test_thinning_keeps_a_one_pixel_wide_line() -> None:
    rows = [[0] * 7 for _ in range(7)]
    rows[3] = [0, 255, 255, 255, 255, 255, 0]

    result = thinning(_gray_image(rows), threshold=128, max_iterations=10)

    # The line is already a skeleton: its end pixels have one neighbour and the middle ones have two
    # neighbours that are not connected to each other, so nothing may be removed.
    assert _values(result) == rows


# ---------- Pipeline 5, stage 3: region growing ----------

def test_region_growing_rejects_invalid_parameters() -> None:
    with pytest.raises(ValueError):
        region_growing(Image.new("RGB", (3, 3)), connectivity=6)
    with pytest.raises(ValueError):
        region_growing(Image.new("RGB", (3, 3)), seed_threshold=-1)


def test_region_growing_connectivity_four_does_not_cross_diagonals() -> None:
    image = _gray_image([[10, 200], [200, 14]])

    result = region_growing(image, seed_threshold=5, connectivity=4)

    # 10 and 14 touch only diagonally, so they stay separate regions with their own mean.
    assert _values(result) == [[10, 200], [200, 14]]


def test_region_growing_connectivity_eight_joins_diagonals() -> None:
    image = _gray_image([[10, 200], [200, 14]])

    result = region_growing(image, seed_threshold=5, connectivity=8)

    # Now 10 and 14 are one region, painted with the mean (10 + 14) / 2 = 12.
    assert _values(result) == [[12, 200], [200, 12]]


# ---------- Pipeline 6, stage 1: Hough ----------

def test_hough_rejects_invalid_parameters() -> None:
    with pytest.raises(ValueError):
        hough(Image.new("RGB", (5, 5)), theta_steps=1)
    with pytest.raises(ValueError):
        hough(Image.new("RGB", (5, 5)), max_lines=0)


def test_hough_returns_the_image_unchanged_when_there_are_no_edges() -> None:
    image = Image.new("RGB", (6, 6), (90, 90, 90))

    result = hough(image, edge_threshold=100, theta_steps=4, vote_threshold=1, max_lines=1)

    assert list(result.getdata()) == list(image.getdata())


# ---------- Pipeline 6, stage 2: affine ----------

def test_affine_rejects_invalid_parameters() -> None:
    with pytest.raises(ValueError):
        affine(Image.new("RGB", (3, 3)), scale_x=0.0)
    with pytest.raises(ValueError):
        affine(Image.new("RGB", (3, 3)), fill_value=300)


def test_affine_with_zero_angle_unit_scale_and_no_shift_is_identity() -> None:
    image = _gray_image([[10, 20, 30], [40, 50, 60], [70, 80, 90]])

    result = affine(image, angle_degrees=0.0, scale_x=1.0, scale_y=1.0, translate_x=0.0, translate_y=0.0)

    # This is exactly the configuration stored in config/pipelines.yaml.
    assert _values(result) == _values(image)


def test_affine_rotation_by_ninety_degrees_rotates_a_square_image() -> None:
    image = _gray_image([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    result = affine(image, angle_degrees=90.0)

    rotated = np.rot90(np.asarray(_values(image)), k=-1).tolist()
    assert _values(result) == rotated


# ---------- Pipeline 6, stage 3: brightness / contrast ----------

def test_brightness_contrast_rejects_invalid_parameters() -> None:
    with pytest.raises(ValueError):
        brightness_contrast(Image.new("RGB", (2, 2)), contrast=-0.1)
    with pytest.raises(ValueError):
        brightness_contrast(Image.new("RGB", (2, 2)), brightness=300)


def test_brightness_contrast_defaults_of_one_and_zero_change_nothing() -> None:
    image = _gray_image([[0, 100, 255]] * 3)

    result = brightness_contrast(image, brightness=0, contrast=1.0)

    assert _values(result) == _values(image)


def test_contrast_stretches_values_around_128_and_clamps() -> None:
    image = _gray_image([[100, 128, 200]] * 3)

    result = brightness_contrast(image, brightness=0, contrast=2.0)

    # (v - 128) * 2 + 128: 100 -> 72, 128 -> 128, 200 -> 272 which is clipped to 255.
    assert _values(result) == [[72, 128, 255]] * 3


def test_negative_brightness_clamps_at_zero() -> None:
    image = _gray_image([[5, 100, 255]] * 3)

    result = brightness_contrast(image, brightness=-20, contrast=1.0)

    assert _values(result) == [[0, 80, 235]] * 3
