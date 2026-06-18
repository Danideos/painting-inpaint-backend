"""Flux Fill inpainting backend."""

from __future__ import annotations

import inspect
import os
import time
from typing import Any

from PIL import Image

from painting_inpaint.compositing import hard_composite
from painting_inpaint.pipelines.base import (
    InpaintingBackend,
    InpaintingRequest,
    InpaintingResult,
)

DEFAULT_FLUX_FILL_MODEL_ID = "black-forest-labs/FLUX.1-Fill-dev"


def _disable_transformers_sklearn_imports() -> None:
    """Avoid optional sklearn import paths that are fragile in Colab runtimes."""

    try:
        import transformers.utils.import_utils as transformers_import_utils

        transformers_import_utils._sklearn_available = False
    except Exception:
        pass


def _resolve_torch_dtype(torch: Any, dtype_config: Any) -> Any:
    if dtype_config in (None, "auto"):
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if dtype_config in ("bfloat16", "bf16"):
        return torch.bfloat16
    if dtype_config in ("float16", "fp16"):
        return torch.float16
    if dtype_config in ("float32", "fp32"):
        return torch.float32
    return dtype_config


class FluxFillBackend(InpaintingBackend):
    """Plain Flux 1 Fill inpainting through diffusers."""

    name = "flux_fill"

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self._pipe = None
        self._torch = None
        self._torch_dtype = None
        self._supports_negative_prompt = False

    def _load_pipeline(self):
        if self._pipe is not None:
            return self._pipe

        _disable_transformers_sklearn_imports()

        try:
            import torch
            from diffusers import FluxFillPipeline
        except ImportError as exc:
            raise ImportError(
                "Flux Fill requires optional model dependencies: "
                "diffusers, transformers, accelerate, safetensors, and torch."
            ) from exc

        model_id = self.config.get("model_id", DEFAULT_FLUX_FILL_MODEL_ID)
        torch_dtype = _resolve_torch_dtype(torch, self.config.get("torch_dtype", "auto"))
        token = self.config.get("token") or os.environ.get("HF_TOKEN")

        kwargs: dict[str, Any] = {"torch_dtype": torch_dtype}
        if token:
            kwargs["token"] = token

        pipe = FluxFillPipeline.from_pretrained(model_id, **kwargs)

        if self.config.get("enable_model_cpu_offload", True):
            pipe.enable_model_cpu_offload()

        if getattr(pipe, "vae", None) is not None:
            if self.config.get("enable_vae_tiling", True):
                pipe.vae.enable_tiling()
            if self.config.get("enable_vae_slicing", True):
                pipe.vae.enable_slicing()

        self._pipe = pipe
        self._torch = torch
        self._torch_dtype = torch_dtype
        self._supports_negative_prompt = "negative_prompt" in inspect.signature(
            pipe.__call__
        ).parameters
        return pipe

    @property
    def supports_negative_prompt(self) -> bool:
        self._load_pipeline()
        return self._supports_negative_prompt

    @property
    def torch_dtype(self) -> Any:
        self._load_pipeline()
        return self._torch_dtype

    def run(self, request: InpaintingRequest) -> InpaintingResult:
        pipe = self._load_pipeline()
        torch = self._torch
        if torch is None:
            raise RuntimeError("Flux pipeline loaded without torch")

        cfg = {**self.config, **dict(request.config)}
        seed = request.seed if request.seed is not None else cfg.get("seed")

        generator = None
        if seed is not None:
            generator = torch.Generator(device=cfg.get("generator_device", "cpu")).manual_seed(
                int(seed)
            )

        width = int(cfg.get("width", request.image.width))
        height = int(cfg.get("height", request.image.height))

        call_kwargs: dict[str, Any] = {
            "prompt": cfg.get("prompt", ""),
            "image": request.image.convert("RGB"),
            "mask_image": request.mask.convert("L"),
            "height": height,
            "width": width,
            "strength": float(cfg.get("strength", 1.0)),
            "num_inference_steps": int(cfg.get("num_inference_steps", 50)),
            "guidance_scale": float(cfg.get("guidance_scale", 30.0)),
            "max_sequence_length": int(cfg.get("max_sequence_length", 512)),
        }
        if generator is not None:
            call_kwargs["generator"] = generator
        if self._supports_negative_prompt:
            call_kwargs["negative_prompt"] = cfg.get("negative_prompt", "")

        started = time.perf_counter()
        with torch.inference_mode():
            raw = pipe(**call_kwargs).images[0].convert("RGB")
        runtime_seconds = time.perf_counter() - started

        if raw.size != request.image.size:
            raw = raw.resize(request.image.size, Image.Resampling.LANCZOS)

        composite = hard_composite(request.image, raw, request.mask)
        return InpaintingResult(
            raw_image=raw,
            composite_image=composite,
            metadata={
                "backend": self.name,
                "model_id": cfg.get("model_id", DEFAULT_FLUX_FILL_MODEL_ID),
                "seed": seed,
                "runtime_seconds": runtime_seconds,
                "torch_dtype": str(self._torch_dtype),
                "supports_negative_prompt": self._supports_negative_prompt,
                "inference_settings": {
                    "prompt": cfg.get("prompt", ""),
                    "negative_prompt": cfg.get("negative_prompt", ""),
                    "negative_prompt_passed_to_pipeline": self._supports_negative_prompt,
                    "strength": call_kwargs["strength"],
                    "num_inference_steps": call_kwargs["num_inference_steps"],
                    "guidance_scale": call_kwargs["guidance_scale"],
                    "height": height,
                    "width": width,
                    "max_sequence_length": call_kwargs["max_sequence_length"],
                },
            },
        )
