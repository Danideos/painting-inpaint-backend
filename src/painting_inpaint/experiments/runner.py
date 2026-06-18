"""Small orchestration helpers shared by experiment entry points."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from painting_inpaint.masks import load_binary_mask
from painting_inpaint.paths import resolve_data_path


def load_case_images(case: dict) -> tuple[Image.Image, Image.Image]:
    """Load the image and mask declared by a case config."""

    image_path = resolve_data_path(case.get("image"))
    mask_path = resolve_data_path(case.get("mask"))
    if image_path is None or mask_path is None:
        raise ValueError("Case config must define image and mask paths.")
    if not Path(image_path).exists():
        raise FileNotFoundError(image_path)
    if not Path(mask_path).exists():
        raise FileNotFoundError(mask_path)
    image = Image.open(image_path).convert("RGB")
    mask = load_binary_mask(mask_path, threshold=int(case.get("mask_threshold", 127)))
    if image.size != mask.size:
        raise ValueError(f"Image and mask sizes differ: {image.size} vs {mask.size}")
    return image, mask
