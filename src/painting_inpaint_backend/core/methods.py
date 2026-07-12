"""Stable restoration method identifiers shared by provider adapters."""

from __future__ import annotations

from typing import Any, Literal

FLUX_FILL_METHOD = "flux_fill"
FLUX_CANNY_LANPAINT_METHOD = "flux_canny_lanpaint"
FLUX_CANNY_LANPAINT_NATIVE_METHOD = "flux_canny_lanpaint_native"
FLUX_CANNY_FILL_METHOD = "flux_canny_fill"
FLUX_FILL_CANNY_NATIVE_METHOD = "flux_fill_canny_native"
FLUX_FILL_CANNY_FILL_METHOD = "flux_fill_canny_fill"
SD15_INPAINT_METHOD = "sd15_inpaint"
SDXL_INPAINT_METHOD = "sdxl_inpaint"
QWEN_EDIT_METHOD = "qwen_edit"
QWEN_IMAGE_METHOD = "qwen_image"
QWEN_IMAGE_INPAINT_METHOD = "qwen_image_inpaint"
QWEN_IMAGE_LANPAINT_METHOD = "qwen_image_lanpaint"
SD35_INPAINT_METHOD = "sd35_inpaint"
SDXL_BRUSHNET_METHOD = "sdxl_brushnet"
DEFAULT_METHOD = FLUX_FILL_METHOD

RestorationMethod = Literal[
    "flux_fill",
    "flux_canny_lanpaint",
    "flux_canny_lanpaint_native",
    "flux_canny_fill",
    "flux_fill_canny_native",
    "flux_fill_canny_fill",
    "sd15_inpaint",
    "sdxl_inpaint",
    "qwen_edit",
    "qwen_image",
    "qwen_image_inpaint",
    "qwen_image_lanpaint",
    "sd35_inpaint",
    "sdxl_brushnet",
]


def normalize_method(value: Any = None) -> RestorationMethod:
    """Return a supported method, defaulting omitted requests to FLUX Fill."""

    normalized = DEFAULT_METHOD if value is None else str(value).strip().lower()
    if normalized == FLUX_FILL_METHOD:
        return FLUX_FILL_METHOD
    if normalized == FLUX_CANNY_LANPAINT_METHOD:
        return FLUX_CANNY_LANPAINT_METHOD
    if normalized == FLUX_CANNY_LANPAINT_NATIVE_METHOD:
        return FLUX_CANNY_LANPAINT_NATIVE_METHOD
    if normalized == FLUX_CANNY_FILL_METHOD:
        return FLUX_CANNY_FILL_METHOD
    if normalized == FLUX_FILL_CANNY_NATIVE_METHOD:
        return FLUX_FILL_CANNY_NATIVE_METHOD
    if normalized == FLUX_FILL_CANNY_FILL_METHOD:
        return FLUX_FILL_CANNY_FILL_METHOD
    if normalized == SD15_INPAINT_METHOD:
        return SD15_INPAINT_METHOD
    if normalized == SDXL_INPAINT_METHOD:
        return SDXL_INPAINT_METHOD
    if normalized == QWEN_EDIT_METHOD:
        return QWEN_EDIT_METHOD
    if normalized == QWEN_IMAGE_METHOD:
        return QWEN_IMAGE_METHOD
    if normalized == QWEN_IMAGE_INPAINT_METHOD:
        return QWEN_IMAGE_INPAINT_METHOD
    if normalized == QWEN_IMAGE_LANPAINT_METHOD:
        return QWEN_IMAGE_LANPAINT_METHOD
    if normalized == SD35_INPAINT_METHOD:
        return SD35_INPAINT_METHOD
    if normalized == SDXL_BRUSHNET_METHOD:
        return SDXL_BRUSHNET_METHOD
    raise ValueError(
        "method must be one of: flux_fill, flux_canny_lanpaint, "
        "flux_canny_lanpaint_native, flux_canny_fill, flux_fill_canny_native, "
        "flux_fill_canny_fill, sd15_inpaint, sdxl_inpaint, qwen_edit, "
        "qwen_image, qwen_image_inpaint, qwen_image_lanpaint, "
        "sd35_inpaint, sdxl_brushnet; "
        f"got {value!r}."
    )


__all__ = [
    "DEFAULT_METHOD",
    "FLUX_CANNY_FILL_METHOD",
    "FLUX_CANNY_LANPAINT_METHOD",
    "FLUX_CANNY_LANPAINT_NATIVE_METHOD",
    "FLUX_FILL_CANNY_FILL_METHOD",
    "FLUX_FILL_CANNY_NATIVE_METHOD",
    "FLUX_FILL_METHOD",
    "QWEN_EDIT_METHOD",
    "QWEN_IMAGE_INPAINT_METHOD",
    "QWEN_IMAGE_METHOD",
    "QWEN_IMAGE_LANPAINT_METHOD",
    "RestorationMethod",
    "SD35_INPAINT_METHOD",
    "SD15_INPAINT_METHOD",
    "SDXL_BRUSHNET_METHOD",
    "SDXL_INPAINT_METHOD",
    "normalize_method",
]
