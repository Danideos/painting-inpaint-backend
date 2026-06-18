"""Compositing helpers for preserving undamaged painting regions."""

from __future__ import annotations

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

    if original_img.size != generated_img.size or original_img.size != mask_img.size:
        raise ValueError(
            "original, generated, and mask must have the same size: "
            f"{original_img.size}, {generated_img.size}, {mask_img.size}"
        )

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
    if original_img.size != composite_img.size or original_img.size != mask_img.size:
        raise ValueError("original, composite, and mask must have the same size")

    original_arr = np.asarray(original_img, dtype=np.uint8)
    composite_arr = np.asarray(composite_img, dtype=np.uint8)
    keep = np.asarray(mask_img, dtype=np.uint8) <= threshold
    return bool(np.any(original_arr[keep] != composite_arr[keep]))
