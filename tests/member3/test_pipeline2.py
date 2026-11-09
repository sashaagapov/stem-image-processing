"""Hand-computed C3 tests for every stage of member3/pipeline2."""
import numpy as np
from PIL import Image

from src.member3.pipeline2.stage_1_hough_transform import apply as hough
from src.member3.pipeline2.stage_2_affine_transform import apply as affine
from src.member3.pipeline2.stage_3_brightness_contrast import apply as brightness_contrast


def _gray_image(rows: list[list[int]]) -> Image.Image:
    return Image.fromarray(np.array(rows, dtype=np.uint8)).convert("RGB")


def _values(image: Image.Image) -> list[list[int]]:
    return np.asarray(image)[:, :, 0].tolist()


def test_hough_marks_both_edges_of_a_vertical_line() -> None:
    image = _gray_image([[0, 0, 255, 0, 0]] * 5)

    result = hough(image, edge_threshold=100, theta_steps=4, vote_threshold=4, max_lines=2)

    # The two five-pixel gradients vote for rho=1 and rho=3 at theta=0.
    assert list(result.getdata()) == [
        (0, 0, 0), (255, 0, 0), (255, 255, 255), (255, 0, 0), (0, 0, 0)
    ] * 5


def test_affine_translation_moves_a_three_by_three_image_right() -> None:
    image = _gray_image([[10, 20, 30], [40, 50, 60], [70, 80, 90]])

    result = affine(image, angle_degrees=0, translate_x=1, translate_y=0, fill_value=0)

    # Inverse mapping reads x-1, so the left column is fill and the other values shift right.
    assert _values(result) == [[0, 10, 20], [0, 40, 50], [0, 70, 80]]


def test_brightness_correction_clamps_each_channel() -> None:
    image = _gray_image([[0, 128, 250], [0, 128, 250], [0, 128, 250]])

    result = brightness_contrast(image, brightness=10, contrast=1.0)

    # The formula adds 10: 0->10, 128->138, while 250+10 is clipped to 255.
    assert _values(result) == [[10, 138, 255], [10, 138, 255], [10, 138, 255]]
