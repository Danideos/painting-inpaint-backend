"""Stable Diffusion 1.5 inpainting baseline service."""

from __future__ import annotations

from typing import Any

from .methods import SD15_INPAINT_METHOD
from .sd_inpaint_base import SDInpaintServiceBase

SD15_MODEL_ID = "runwayml/stable-diffusion-inpainting"


class SD15InferenceService(SDInpaintServiceBase):
    """Lazy-loading SD1.5 inpainting service."""

    method = SD15_INPAINT_METHOD
    model_id = SD15_MODEL_ID
    torch_dtype_name = "float16"
    pipeline_display_name = "SD1.5 inpainting"

    def pipeline_class(self) -> Any:
        from diffusers import StableDiffusionInpaintPipeline

        return StableDiffusionInpaintPipeline

    def from_pretrained_kwargs(self, torch_dtype: Any) -> dict[str, Any]:
        return {
            "variant": "fp16",
            "safety_checker": None,
            "requires_safety_checker": False,
        }


__all__ = ["SD15InferenceService"]

