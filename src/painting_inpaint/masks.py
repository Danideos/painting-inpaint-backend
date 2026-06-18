"""Mask loading, generation, morphology, and debug helpers."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

ImageInput = Image.Image | str | Path


def _load_image(image: ImageInput, mode: str) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert(mode)
    return Image.open(image).convert(mode)


@contextmanager
def allow_large_images() -> Iterator[None]:
    """Temporarily disable Pillow's large-image pixel guard.

    The Rosary Feast source scans are intentionally huge. Callers should still prefer
    preview/tiled workflows before full-resolution processing.
    """

    previous_limit = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = None
    try:
        yield
    finally:
        Image.MAX_IMAGE_PIXELS = previous_limit


def load_preview(
    path: str | Path,
    max_side: int,
    mode: str = "RGB",
) -> tuple[Image.Image, tuple[int, int]]:
    """Load a downscaled preview and return ``(preview, original_size)``."""

    if max_side <= 0:
        raise ValueError("max_side must be positive")
    with allow_large_images(), Image.open(path) as img:
        original_size = img.size
        img.draft(mode, (max_side, max_side))
        img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS, reducing_gap=3.0)
        return img.convert(mode).copy(), original_size


def binarize_mask(mask: ImageInput, threshold: int = 127, invert: bool = False) -> Image.Image:
    """Return a binary ``L`` mask where white means edit/fill."""

    arr = np.asarray(_load_image(mask, "L"), dtype=np.uint8)
    binary = arr > threshold
    if invert:
        binary = ~binary
    return Image.fromarray(binary.astype(np.uint8) * 255)


def load_binary_mask(path: str | Path, threshold: int = 127, invert: bool = False) -> Image.Image:
    """Load a mask from disk and binarize it."""

    return binarize_mask(path, threshold=threshold, invert=invert)


def dilate_mask(mask: ImageInput, pixels: int = 1) -> Image.Image:
    """Dilate a binary mask by approximately ``pixels`` pixels."""

    result = binarize_mask(mask)
    if pixels <= 0:
        return result
    return result.filter(ImageFilter.MaxFilter(2 * pixels + 1))


def erode_mask(mask: ImageInput, pixels: int = 1) -> Image.Image:
    """Erode a binary mask by approximately ``pixels`` pixels."""

    result = binarize_mask(mask)
    if pixels <= 0:
        return result
    return result.filter(ImageFilter.MinFilter(2 * pixels + 1))


def create_mask_from_white_regions(
    image: ImageInput,
    threshold: int = 240,
    dilate_pixels: int = 0,
    erode_pixels: int = 0,
) -> Image.Image:
    """Create an inpainting mask from near-white damaged regions.

    A pixel is considered damaged when all RGB channels are at least ``threshold``.
    The returned mask uses the common inpainting convention: white = edit.
    """

    rgb = np.asarray(_load_image(image, "RGB"), dtype=np.uint8)
    white = np.all(rgb >= threshold, axis=2)
    mask = Image.fromarray(white.astype(np.uint8) * 255)
    if dilate_pixels:
        mask = dilate_mask(mask, dilate_pixels)
    if erode_pixels:
        mask = erode_mask(mask, erode_pixels)
    return binarize_mask(mask)


def create_mask_from_white_regions_tiled(
    image_path: str | Path,
    threshold: int = 240,
    dilate_pixels: int = 0,
    erode_pixels: int = 0,
    tile_size: int = 2048,
) -> Image.Image:
    """Create a white-region mask from a large RGB image using tiled reads.

    This avoids materializing the full RGB source scan as a NumPy array. The returned
    mask is still a full-size ``L`` image because downstream warping needs the full
    mask canvas.
    """

    if not 0 <= threshold <= 255:
        raise ValueError("threshold must be in [0, 255]")
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")

    with allow_large_images(), Image.open(image_path) as img:
        width, height = img.size
        mask = Image.new("L", (width, height), 0)
        for top in range(0, height, tile_size):
            bottom = min(top + tile_size, height)
            for left in range(0, width, tile_size):
                right = min(left + tile_size, width)
                tile = img.crop((left, top, right, bottom)).convert("RGB")
                arr = np.asarray(tile, dtype=np.uint8)
                white = np.all(arr >= threshold, axis=2)
                mask_tile = Image.fromarray(white.astype(np.uint8) * 255)
                mask.paste(mask_tile, (left, top))

    if dilate_pixels:
        mask = dilate_mask(mask, dilate_pixels)
    if erode_pixels:
        mask = erode_mask(mask, erode_pixels)
    return binarize_mask(mask)


def mask_coverage(mask: ImageInput, threshold: int = 127) -> float:
    """Return the fraction of pixels marked as editable."""

    arr = np.asarray(_load_image(mask, "L"), dtype=np.uint8)
    if arr.size == 0:
        return 0.0
    return float(np.mean(arr > threshold))


def ensure_same_size(images: Iterable[Image.Image]) -> tuple[int, int]:
    """Validate that all provided images share one size and return it."""

    sizes = [img.size for img in images if img is not None]
    if not sizes:
        raise ValueError("At least one image is required.")
    first = sizes[0]
    if any(size != first for size in sizes):
        raise ValueError(f"Image sizes do not match: {sizes}")
    return first


def make_mask_overlay(
    image: ImageInput,
    mask: ImageInput,
    color: tuple[int, int, int] = (255, 0, 0),
    alpha: float = 0.45,
    threshold: int = 127,
) -> Image.Image:
    """Return an RGB preview with masked pixels tinted."""

    base = _load_image(image, "RGB")
    mask_img = binarize_mask(mask, threshold=threshold)
    ensure_same_size([base, mask_img])

    base_arr = np.asarray(base, dtype=np.float32)
    mask_arr = np.asarray(mask_img, dtype=np.uint8) > threshold
    color_arr = np.asarray(color, dtype=np.float32)
    out = base_arr.copy()
    out[mask_arr] = (1.0 - alpha) * out[mask_arr] + alpha * color_arr
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def save_mask_debug_preview(
    image: ImageInput,
    mask: ImageInput,
    output_path: str | Path,
    color: tuple[int, int, int] = (255, 0, 0),
    alpha: float = 0.45,
) -> Path:
    """Save a mask overlay preview and return its path."""

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    make_mask_overlay(image, mask, color=color, alpha=alpha).save(out)
    return out
