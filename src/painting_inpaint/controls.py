"""Control-image generation utilities."""

from __future__ import annotations

from PIL import Image, ImageFilter


def make_canny_control(
    image: Image.Image,
    low_threshold: int = 50,
    high_threshold: int = 150,
    blur_radius: float = 0.0,
) -> Image.Image:
    """Create a 3-channel Canny edge control image.

    OpenCV is an optional dependency. Install the ``controlnet`` extra before using this.
    """

    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise ImportError(
            "Canny control generation requires opencv-python. "
            'Install with: pip install -e ".[controlnet]"'
        ) from exc

    img = image.convert("RGB")
    if blur_radius > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    arr = np.asarray(img, dtype=np.uint8)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, threshold1=low_threshold, threshold2=high_threshold)
    rgb = np.stack([edges, edges, edges], axis=-1)
    return Image.fromarray(rgb.astype(np.uint8))
