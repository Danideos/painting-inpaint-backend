"""Stable restoration method identifiers shared by provider adapters."""

from __future__ import annotations

from typing import Any, Literal

FLUX_FILL_METHOD = "flux_fill"
FLUX_CANNY_LANPAINT_METHOD = "flux_canny_lanpaint"
DEFAULT_METHOD = FLUX_FILL_METHOD

RestorationMethod = Literal["flux_fill", "flux_canny_lanpaint"]


def normalize_method(value: Any = None) -> RestorationMethod:
    """Return a supported method, defaulting omitted requests to FLUX Fill."""

    normalized = DEFAULT_METHOD if value is None else str(value).strip().lower()
    if normalized == FLUX_FILL_METHOD:
        return FLUX_FILL_METHOD
    if normalized == FLUX_CANNY_LANPAINT_METHOD:
        return FLUX_CANNY_LANPAINT_METHOD
    raise ValueError(
        "method must be one of: flux_fill, flux_canny_lanpaint; "
        f"got {value!r}."
    )


__all__ = [
    "DEFAULT_METHOD",
    "FLUX_CANNY_LANPAINT_METHOD",
    "FLUX_FILL_METHOD",
    "RestorationMethod",
    "normalize_method",
]
