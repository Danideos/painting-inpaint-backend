"""Helpers for assembling optional conditioning inputs."""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image


@dataclass(frozen=True)
class ConditioningInputs:
    """Synchronized optional inputs for conditioned inpainting."""

    reference_image: Image.Image | None = None
    control_image: Image.Image | None = None


def validate_conditioning_inputs(
    image: Image.Image,
    mask: Image.Image,
    reference_image: Image.Image | None = None,
    control_image: Image.Image | None = None,
) -> ConditioningInputs:
    """Validate that optional conditioning inputs match image/mask size."""

    expected = image.size
    items = {
        "mask": mask,
        "reference_image": reference_image,
        "control_image": control_image,
    }
    for name, item in items.items():
        if item is not None and item.size != expected:
            raise ValueError(f"{name} size {item.size} does not match image size {expected}")
    return ConditioningInputs(reference_image=reference_image, control_image=control_image)
