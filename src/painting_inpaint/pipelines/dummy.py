"""No-model inpainting backend for infrastructure smoke tests."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from PIL import Image

from painting_inpaint.compositing import hard_composite
from painting_inpaint.pipelines.base import InpaintingBackend, InpaintingRequest, InpaintingResult


class DummyInpaintBackend(InpaintingBackend):
    """Fake backend that visibly fills masked pixels without loading any model."""

    name = "dummy_inpaint"

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}

    def run(self, request: InpaintingRequest) -> InpaintingResult:
        cfg = {**self.config, **dict(request.config)}
        fill_color = tuple(int(v) for v in cfg.get("fill_color", [255, 0, 255]))
        if len(fill_color) != 3:
            raise ValueError("dummy fill_color must contain exactly three RGB values")

        started = time.perf_counter()
        image = request.image.convert("RGB")
        mask = request.mask.convert("L")

        image_arr = np.asarray(image, dtype=np.uint8)
        mask_arr = np.asarray(mask, dtype=np.uint8) > int(cfg.get("mask_threshold", 127))

        raw_arr = image_arr.copy()
        raw_arr[mask_arr] = np.asarray(fill_color, dtype=np.uint8)
        raw = Image.fromarray(raw_arr)
        composite = hard_composite(image, raw, mask)
        runtime_seconds = time.perf_counter() - started

        return InpaintingResult(
            raw_image=raw,
            composite_image=composite,
            metadata={
                "backend": self.name,
                "model_id": cfg.get("model_id", "dummy/no-model"),
                "seed": request.seed if request.seed is not None else cfg.get("seed"),
                "runtime_seconds": runtime_seconds,
                "fill_color": list(fill_color),
                "model_loaded": False,
            },
        )
