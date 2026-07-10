"""Stable Diffusion XL inpainting baseline service."""

from __future__ import annotations

from typing import Any

from .methods import SDXL_INPAINT_METHOD
from .sd_inpaint_base import SDInpaintServiceBase

SDXL_MODEL_ID = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"


class SDXLInferenceService(SDInpaintServiceBase):
    """Lazy-loading SDXL inpainting service."""

    method = SDXL_INPAINT_METHOD
    model_id = SDXL_MODEL_ID
    torch_dtype_name = "bfloat16"
    pipeline_display_name = "SDXL inpainting"

    def pipeline_class(self) -> Any:
        from diffusers import StableDiffusionXLInpaintPipeline

        return StableDiffusionXLInpaintPipeline

    def from_pretrained_kwargs(self, torch_dtype: Any) -> dict[str, Any]:
        del torch_dtype
        return {"add_watermarker": False}


__all__ = ["SDXLInferenceService"]

