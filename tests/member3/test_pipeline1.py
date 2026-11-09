"""Hand-computed C3 tests for every stage of member3/pipeline1."""
import numpy as np
from PIL import Image

from src.member3.pipeline1.stage_1_harris_corners import apply as harris
from src.member3.pipeline1.stage_2_morphology_open_close import apply as morphology
from src.member3.pipeline1.stage_3_region_growing import apply as region_growing


def _gray_image(rows: list[list[int]]) -> Image.Image:
    return Image.fromarray(np.array(rows, dtype=np.uint8)).convert("RGB")


def _values(image: Image.Image) -> list[list[int]]:
    return np.asarray(image)[:, :, 0].tolist()


def test_harris_keeps_a_uniform_three_by_three_image() -> None:
    image = Image.new("RGB", (3, 3), (10, 20, 30))

    result = harris(image, threshold=0.01, harris_k=0.04, window_size=3)

    # All horizontal and vertical differences are zero, so every R is zero.
    assert list(result.getdata()) == [(10, 20, 30)] * 9


def test_thinning_reduces_a_solid_three_by_three_square_to_its_centre() -> None:
    image = _gray_image(
        [
            [0, 0, 0, 0, 0],
            [0, 255, 255, 255, 0],
            [0, 255, 255, 255, 0],
            [0, 255, 255, 255, 0],
            [0, 0, 0, 0, 0],
        ]
    )

    result = morphology(image, threshold=128, max_iterations=20)

    # Zhang-Suen deletes only boundary pixels while preserving connectivity, leaving the centre.
    assert _values(result) == [
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 255, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ]


def test_region_growing_renders_two_constant_regions() -> None:
    image = _gray_image([[10, 10, 200], [10, 10, 200], [10, 10, 200]])

    result = region_growing(image, seed_threshold=5, connectivity=4)

    # Difference inside a group is 0, while 200 - 10 = 190 exceeds the threshold.
    assert _values(result) == [[10, 10, 200], [10, 10, 200], [10, 10, 200]]
