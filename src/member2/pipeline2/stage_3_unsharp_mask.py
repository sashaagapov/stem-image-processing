from src.member2.image_ops import from_array, gaussian_blur, to_rgb_array


def apply(image, amount=1.0, sigma=2.0):
    original = to_rgb_array(image)
    blurred = gaussian_blur(original, sigma)
    return from_array(original + amount * (original - blurred))
