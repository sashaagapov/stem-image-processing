from PIL import Image

from common.metrics import calculate_mse, calculate_psnr, calculate_ssim


def test_mse_is_zero_for_identical_images() -> None:
    image = Image.new("RGB", (10, 10), (120, 60, 30))

    assert calculate_mse(image, image) == 0


def test_mse_is_positive_for_different_images() -> None:
    image_a = Image.new("RGB", (4, 4), (0, 0, 0))
    image_b = Image.new("RGB", (4, 4), (10, 10, 10))

    assert calculate_mse(image_a, image_b) > 0


def test_mse_matches_hand_computed_value_on_small_image() -> None:
    image_a = Image.new("RGB", (2, 1))
    image_a.putpixel((0, 0), (0, 0, 0))
    image_a.putpixel((1, 0), (10, 10, 10))

    image_b = Image.new("RGB", (2, 1))
    image_b.putpixel((0, 0), (10, 10, 10))
    image_b.putpixel((1, 0), (10, 10, 10))

    # Різниця лише в одному пікселі: (10-0)^2 = 100 на кожен з 3 каналів,
    # середнє по 2 пікселях x 3 канали = 100*3 / 6 = 50.
    assert calculate_mse(image_a, image_b) == 50.0


def test_psnr_is_infinite_for_identical_images() -> None:
    image = Image.new("RGB", (8, 8), (50, 100, 150))

    assert calculate_psnr(image, image) == float("inf")


def test_psnr_decreases_as_images_differ_more() -> None:
    reference = Image.new("RGB", (16, 16), (128, 128, 128))
    slightly_different = Image.new("RGB", (16, 16), (130, 128, 128))
    very_different = Image.new("RGB", (16, 16), (200, 128, 128))

    psnr_small_diff = calculate_psnr(reference, slightly_different)
    psnr_large_diff = calculate_psnr(reference, very_different)

    assert psnr_small_diff > psnr_large_diff


def test_ssim_is_one_for_identical_images() -> None:
    image = Image.new("RGB", (20, 20), (90, 90, 90))

    assert abs(calculate_ssim(image, image) - 1.0) < 1e-6


def test_ssim_is_lower_for_more_different_images() -> None:
    reference = Image.new("RGB", (24, 24), (100, 100, 100))
    slightly_different = Image.new("RGB", (24, 24), (110, 100, 100))
    very_different = Image.new("RGB", (24, 24), (220, 100, 100))

    ssim_small_diff = calculate_ssim(reference, slightly_different)
    ssim_large_diff = calculate_ssim(reference, very_different)

    assert ssim_small_diff > ssim_large_diff


def test_metrics_reject_images_of_different_size() -> None:
    small = Image.new("RGB", (4, 4))
    large = Image.new("RGB", (8, 8))

    for metric in (calculate_mse, calculate_psnr, calculate_ssim):
        try:
            metric(small, large)
        except ValueError:
            continue
        assert False, f"{metric.__name__} мав викликати ValueError на різних розмірах"
