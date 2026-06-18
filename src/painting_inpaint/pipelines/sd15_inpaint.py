"""Stable Diffusion 1.5 inpainting backend."""

from __future__ import annotations

import time
from typing import Any

from painting_inpaint.compositing import hard_composite
from painting_inpaint.pipelines.base import InpaintingBackend, InpaintingRequest, InpaintingResult


class SD15InpaintBackend(InpaintingBackend):
    """Plain SD1.5 inpainting through diffusers."""

    name = "sd15_inpaint"

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self._pipe = None

    def _load_pipeline(self):
        if self._pipe is not None:
            return self._pipe

        try:
            import torch
            from diffusers import AutoPipelineForInpainting
        except ImportError as exc:
            raise ImportError(
                "SD1.5 inpainting requires optional dependencies. "
                'Install with: pip install -e ".[sd15]"'
            ) from exc

        model_id = self.config.get("model_id", "runwayml/stable-diffusion-inpainting")
        device = self.config.get("device", "auto")
        dtype_name = self.config.get("torch_dtype", "float16")
        dtype = torch.float16 if dtype_name == "float16" else torch.float32

        pipe = AutoPipelineForInpainting.from_pretrained(
            model_id,
            torch_dtype=dtype,
            variant=self.config.get("variant", "fp16" if dtype == torch.float16 else None),
            safety_checker=None if self.config.get("disable_safety_checker", True) else None,
        )

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if device == "cuda":
            if self.config.get("enable_attention_slicing", True):
                pipe.enable_attention_slicing()
            if self.config.get("enable_vae_slicing", True):
                pipe.enable_vae_slicing()
            if self.config.get("enable_model_cpu_offload", True):
                pipe.enable_model_cpu_offload()
            else:
                pipe.to("cuda")
        else:
            pipe.to(device)

        self._pipe = pipe
        return pipe

    def run(self, request: InpaintingRequest) -> InpaintingResult:
        pipe = self._load_pipeline()

        try:
            import torch
        except ImportError as exc:
            raise ImportError("torch is required for SD1.5 inpainting") from exc

        cfg = {**self.config, **dict(request.config)}
        seed = request.seed if request.seed is not None else cfg.get("seed")
        generator = None
        if seed is not None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            generator = torch.Generator(device=device).manual_seed(int(seed))

        started = time.perf_counter()
        raw = pipe(
            prompt=cfg.get("prompt", ""),
            negative_prompt=cfg.get("negative_prompt", ""),
            image=request.image.convert("RGB"),
            mask_image=request.mask.convert("L"),
            guidance_scale=float(cfg.get("guidance_scale", 7.5)),
            num_inference_steps=int(cfg.get("num_inference_steps", 30)),
            strength=float(cfg.get("strength", 1.0)),
            generator=generator,
        ).images[0]
        runtime_seconds = time.perf_counter() - started

        composite = hard_composite(request.image, raw, request.mask)
        return InpaintingResult(
            raw_image=raw,
            composite_image=composite,
            metadata={
                "backend": self.name,
                "model_id": cfg.get("model_id", "runwayml/stable-diffusion-inpainting"),
                "seed": seed,
                "runtime_seconds": runtime_seconds,
            },
        )
