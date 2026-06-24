"""Small image helpers kept local to the RunPod worker."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
from PIL import Image

ImageInput = Image.Image | str | Path


def _load_image(image: ImageInput, mode: str) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert(mode)
    return Image.open(image).convert(mode)


def hard_composite(
    original: ImageInput,
    generated: ImageInput,
    mask: ImageInput,
    threshold: int = 127,
) -> Image.Image:
    """Copy generated pixels inside the mask and original pixels outside it exactly."""

    original_img = _load_image(original, "RGB")
    generated_img = _load_image(generated, "RGB")
    mask_img = _load_image(mask, "L")
    ensure_same_size([original_img, generated_img, mask_img])

    original_arr = np.asarray(original_img, dtype=np.uint8)
    generated_arr = np.asarray(generated_img, dtype=np.uint8)
    mask_arr = np.asarray(mask_img, dtype=np.uint8) > threshold

    out = original_arr.copy()
    out[mask_arr] = generated_arr[mask_arr]
    return Image.fromarray(out)


def outside_mask_changed(
    original: ImageInput,
    composite: ImageInput,
    mask: ImageInput,
    threshold: int = 127,
) -> bool:
    """Return whether any unmasked pixel differs between original and composite."""

    original_img = _load_image(original, "RGB")
    composite_img = _load_image(composite, "RGB")
    mask_img = _load_image(mask, "L")
    ensure_same_size([original_img, composite_img, mask_img])

    original_arr = np.asarray(original_img, dtype=np.uint8)
    composite_arr = np.asarray(composite_img, dtype=np.uint8)
    keep = np.asarray(mask_img, dtype=np.uint8) <= threshold
    return bool(np.any(original_arr[keep] != composite_arr[keep]))


def binarize_mask(mask: ImageInput, threshold: int = 127, invert: bool = False) -> Image.Image:
    """Return a binary ``L`` mask where white means edit/fill."""

    arr = np.asarray(_load_image(mask, "L"), dtype=np.uint8)
    binary = arr > threshold
    if invert:
        binary = ~binary
    return Image.fromarray(binary.astype(np.uint8) * 255)


def ensure_same_size(images: Iterable[Image.Image]) -> tuple[int, int]:
    """Validate that all provided images share one size and return it."""

    sizes = [img.size for img in images if img is not None]
    if not sizes:
        raise ValueError("At least one image is required.")
    first = sizes[0]
    if any(size != first for size in sizes):
        raise ValueError(f"Image sizes do not match: {sizes}")
    return first


def mask_coverage(mask: ImageInput, threshold: int = 127) -> float:
    """Return the fraction of pixels marked as editable."""

    arr = np.asarray(_load_image(mask, "L"), dtype=np.uint8)
    if arr.size == 0:
        return 0.0
    return float(np.mean(arr > threshold))
