"""Qwen-Image-Edit masked inpainting service."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from PIL import Image

from .image_helpers import binarize_mask, hard_composite, mask_coverage, outside_mask_changed
from .image_io import image_to_base64, normalize_output_format
from .inference import (
    WorkerInputError,
    _add_inference_progress_callback,
    _cuda_memory_stats,
    _effective_step_total,
    _optional_float,
    _optional_int,
    _payload_flag,
    _reset_cuda_peak_memory,
    load_request_images,
)
from .methods import QWEN_EDIT_METHOD, RestorationMethod, normalize_method
from .model_loading import env_flag, resolve_model_load_target
from .progress import ProgressReporter
from .sd_inpaint_base import LoadedSDPipeline

LOGGER = logging.getLogger(__name__)

QWEN_EDIT_MODEL_ID = "Qwen/Qwen-Image-Edit"
QWEN_MASKED_REGION_FILL_RGB = (255, 255, 255)


def make_qwen_source_image(image: Image.Image, mask: Image.Image) -> Image.Image:
    """Hide masked pixels before Qwen's image-conditioning path sees the source."""

    source = image.convert("RGB").copy()
    source.paste(QWEN_MASKED_REGION_FILL_RGB, mask=binarize_mask(mask))
    return source


@dataclass(frozen=True)
class QwenEditSettings:
    """Validated scalar settings for Qwen masked editing."""

    method: RestorationMethod
    prompt: str
    negative_prompt: str
    guidance_scale: float
    num_inference_steps: int
    strength: float
    seed: int | None
    output_format: str
    max_sequence_length: int
    padding_mask_crop: int | None


def parse_qwen_edit_settings(payload: dict[str, Any]) -> QwenEditSettings:
    """Validate request scalar parameters for Qwen masked editing."""

    try:
        method = normalize_method(payload.get("method"))
    except ValueError as exc:
        raise WorkerInputError(str(exc)) from exc
    if method != QWEN_EDIT_METHOD:
        raise WorkerInputError(f"The service only accepts method={QWEN_EDIT_METHOD!r}.")

    prompt = payload.get("prompt", "")
    if not isinstance(prompt, str):
        raise WorkerInputError("prompt must be a string.")

    guidance_scale = _optional_float(payload, "guidance_scale", 4.0)
    if guidance_scale < 0:
        raise WorkerInputError("guidance_scale must be non-negative.")

    num_inference_steps = _optional_int(payload, "num_inference_steps", 50)
    if num_inference_steps <= 0:
        raise WorkerInputError("num_inference_steps must be positive.")

    strength = _optional_float(payload, "strength", 1.0)
    if not 0 <= strength <= 1:
        raise WorkerInputError("strength must be between 0 and 1.")

    max_sequence_length = _optional_int(payload, "max_sequence_length", 512)
    if max_sequence_length <= 0:
        raise WorkerInputError("max_sequence_length must be positive.")

    padding_mask_crop = None
    if payload.get("padding_mask_crop") is not None:
        padding_mask_crop = _optional_int(payload, "padding_mask_crop", 0)
        if padding_mask_crop < 0:
            raise WorkerInputError("padding_mask_crop must be non-negative.")

    seed_value = payload.get("seed")
    seed = None
    if seed_value is not None:
        try:
            seed = int(seed_value)
        except (TypeError, ValueError) as exc:
            raise WorkerInputError("seed must be an integer when provided.") from exc

    try:
        output_format = normalize_output_format(payload.get("output_format", "png"))
    except ValueError as exc:
        raise WorkerInputError(str(exc)) from exc

    return QwenEditSettings(
        method=method,
        prompt=prompt,
        negative_prompt=str(payload.get("negative_prompt", " ")),
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        strength=strength,
        seed=seed,
        output_format=output_format,
        max_sequence_length=max_sequence_length,
        padding_mask_crop=padding_mask_crop,
    )


class QwenEditInferenceService:
    """Lazy-loading Qwen-Image-Edit inpainting service."""

    method = QWEN_EDIT_METHOD
    model_id = QWEN_EDIT_MODEL_ID
    pipeline_display_name = "Qwen-Image-Edit inpainting"

    def __init__(self) -> None:
        self._loaded: LoadedSDPipeline | None = None

    @property
    def loaded(self) -> LoadedSDPipeline:
        return self.get_loaded()

    def get_loaded(self, *, reporter: ProgressReporter | None = None) -> LoadedSDPipeline:
        if self._loaded is None:
            started = time.perf_counter()
            self._loaded = self._load_pipeline(reporter=reporter)
            LOGGER.info(
                "%s pipeline ready in %.3fs",
                self.pipeline_display_name,
                time.perf_counter() - started,
            )
        elif reporter is not None:
            reporter.emit(
                "model_load_start",
                stage="model",
                message=f"Reusing cached {self.pipeline_display_name} pipeline.",
                metadata={"cached": True, "method": self.method},
            )
            reporter.emit(
                "model_load_done",
                stage="model",
                message=f"Cached {self.pipeline_display_name} pipeline ready.",
                metadata={"cached": True, "model": self._loaded.model},
            )
        return self._loaded

    def _load_pipeline(self, *, reporter: ProgressReporter | None = None) -> LoadedSDPipeline:
        import torch
        from diffusers import QwenImageEditInpaintPipeline

        if reporter is not None:
            reporter.emit(
                "model_load_start",
                stage="model",
                message=f"Loading {self.pipeline_display_name} pipeline.",
                metadata={"method": self.method, "model_id": self.model_id},
            )
        timings: dict[str, float] = {}
        resolve_started = time.perf_counter()
        resolution = resolve_model_load_target(self.model_id)
        timings["model_path_discovery_seconds"] = time.perf_counter() - resolve_started

        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        kwargs: dict[str, Any] = {
            "torch_dtype": torch.bfloat16,
            "local_files_only": resolution.local_files_only,
        }
        if token:
            kwargs["token"] = token

        load_started = time.perf_counter()
        pipe = QwenImageEditInpaintPipeline.from_pretrained(resolution.load_target, **kwargs)
        timings["from_pretrained_seconds"] = time.perf_counter() - load_started

        device_started = time.perf_counter()
        if env_flag("ENABLE_MODEL_CPU_OFFLOAD", default=True) and hasattr(
            pipe,
            "enable_model_cpu_offload",
        ):
            pipe.enable_model_cpu_offload()
            device_mode = "model_cpu_offload"
        elif torch.cuda.is_available():
            pipe.to("cuda")
            device_mode = "cuda"
        else:
            pipe.to("cpu")
            device_mode = "cpu"

        if getattr(pipe, "vae", None) is not None:
            if env_flag("ENABLE_VAE_TILING", default=True) and hasattr(pipe.vae, "enable_tiling"):
                pipe.vae.enable_tiling()
            if env_flag("ENABLE_VAE_SLICING", default=True) and hasattr(
                pipe.vae,
                "enable_slicing",
            ):
                pipe.vae.enable_slicing()
        timings["device_setup_seconds"] = time.perf_counter() - device_started

        model = {
            "method": self.method,
            "model_id": self.model_id,
            "load_target": resolution.load_target,
            "source": resolution.source,
            "local_files_only": resolution.local_files_only,
            "cache_root": resolution.cache_root,
            "snapshot_path": resolution.snapshot_path,
            "torch_dtype": str(torch.bfloat16),
            "device_mode": device_mode,
            "pipeline_class": type(pipe).__name__,
            "mask_route": "native_diffusers_mask_image",
            "source_image_route": "masked_region_filled_before_qwen_image_conditioning",
        }
        if reporter is not None:
            reporter.emit(
                "model_load_done",
                stage="model",
                message=f"{self.pipeline_display_name} pipeline loaded.",
                metadata={**model, "timings": timings},
            )

        return LoadedSDPipeline(
            pipe=pipe,
            torch=torch,
            torch_dtype=torch.bfloat16,
            model=model,
            supports_negative_prompt=True,
            timings=timings,
        )

    def run(
        self,
        payload: dict[str, Any],
        *,
        reporter: ProgressReporter | None = None,
    ) -> dict[str, Any]:
        include_progress_history = _payload_flag(
            payload,
            "include_progress_history",
            default=False,
        )
        if reporter is None:
            reporter = ProgressReporter(
                enabled=include_progress_history,
                keep_history=include_progress_history,
            )
        settings = parse_qwen_edit_settings(payload)
        reporter.emit(
            "input_decode_start",
            stage="input",
            message="Decoding request image and mask.",
        )
        image, mask = load_request_images(payload)
        qwen_source_image = make_qwen_source_image(image, mask)
        mask_fraction = mask_coverage(mask)
        reporter.emit(
            "input_decode_done",
            stage="input",
            message="Request image and mask decoded.",
            metadata={
                "image_width": image.width,
                "image_height": image.height,
                "mask_width": mask.width,
                "mask_height": mask.height,
                "mask_coverage": mask_fraction,
                "qwen_masked_source_fill_rgb": QWEN_MASKED_REGION_FILL_RGB,
            },
        )

        loaded = self.get_loaded(reporter=reporter)
        pipe = loaded.pipe
        torch = loaded.torch

        generator = None
        if settings.seed is not None:
            generator = torch.Generator(device="cpu").manual_seed(settings.seed)

        call_kwargs: dict[str, Any] = {
            "prompt": settings.prompt,
            "negative_prompt": settings.negative_prompt,
            "image": qwen_source_image,
            "mask_image": mask,
            "height": image.height,
            "width": image.width,
            "strength": settings.strength,
            "num_inference_steps": settings.num_inference_steps,
            "true_cfg_scale": settings.guidance_scale,
            "max_sequence_length": settings.max_sequence_length,
        }
        if settings.padding_mask_crop is not None:
            call_kwargs["padding_mask_crop"] = settings.padding_mask_crop
        if generator is not None:
            call_kwargs["generator"] = generator
        _add_inference_progress_callback(
            call_kwargs=call_kwargs,
            pipe=pipe,
            settings=settings,  # type: ignore[arg-type]
            reporter=reporter,
        )

        with TemporaryDirectory(prefix=f"{self.method}_worker_") as temp_dir:
            temp_path = Path(temp_dir)
            memory_before_inference = _cuda_memory_stats(torch, prefix="pre_inference_")
            _reset_cuda_peak_memory(torch)
            reporter.emit(
                "inference_start",
                stage="inference",
                message=f"Starting {self.pipeline_display_name} inference.",
                progress={"current": 0, "total": settings.num_inference_steps},
                metadata={
                    "method": settings.method,
                    "num_inference_steps": settings.num_inference_steps,
                    "strength": settings.strength,
                    "guidance_scale": settings.guidance_scale,
                    "seed": settings.seed,
                },
            )
            inference_started = time.perf_counter()
            with torch.inference_mode():
                raw = pipe(**call_kwargs).images[0].convert("RGB")
            inference_seconds = time.perf_counter() - inference_started
            inference_memory = {
                **memory_before_inference,
                **_cuda_memory_stats(torch, prefix="inference_"),
            }
            reporter.emit(
                "inference_done",
                stage="inference",
                message=f"{self.pipeline_display_name} inference completed.",
                progress={
                    "current": _effective_step_total(pipe, settings),  # type: ignore[arg-type]
                    "total": _effective_step_total(pipe, settings),  # type: ignore[arg-type]
                },
                metadata={
                    "inference_seconds": inference_seconds,
                    "gpu_memory": inference_memory,
                },
            )

            if raw.size != image.size:
                raw = raw.resize(image.size, Image.Resampling.LANCZOS)
            reporter.emit(
                "hard_composite_start",
                stage="output",
                message="Hard-compositing output with preserved pixels.",
            )
            composite = hard_composite(image, raw, mask)
            changed_outside_mask = outside_mask_changed(image, composite, mask)
            reporter.emit(
                "hard_composite_done",
                stage="output",
                message="Hard composite completed.",
                metadata={"outside_mask_changed": changed_outside_mask},
            )
            if changed_outside_mask:
                raise RuntimeError("Hard composite changed pixels outside the edit mask.")

            output_path = temp_path / f"composite.{settings.output_format}"
            reporter.emit(
                "output_encode_start",
                stage="output",
                message="Encoding output image.",
                metadata={"output_format": settings.output_format},
            )
            composite.save(output_path)
            encoded = image_to_base64(composite, output_format=settings.output_format)
            reporter.emit(
                "output_encode_done",
                stage="output",
                message="Output image encoded.",
                metadata={
                    "output_format": settings.output_format,
                    "width": composite.width,
                    "height": composite.height,
                },
            )

        output = {
            "image_base64": encoded,
            "output_format": settings.output_format,
            "width": composite.width,
            "height": composite.height,
            "mask_convention": "white = inpaint/edit, black = preserve",
            "timings": {
                **loaded.timings,
                "inference_seconds": inference_seconds,
            },
            "gpu_memory": inference_memory,
            "model": loaded.model,
            "inference_settings": {
                "method": settings.method,
                "prompt": settings.prompt,
                "negative_prompt": settings.negative_prompt,
                "strength": settings.strength,
                "guidance_scale": settings.guidance_scale,
                "guidance_scale_passed_as": "true_cfg_scale",
                "num_inference_steps": settings.num_inference_steps,
                "max_sequence_length": settings.max_sequence_length,
                "padding_mask_crop": settings.padding_mask_crop,
                "seed": settings.seed,
                "mask_coverage": mask_fraction,
                "qwen_masked_source_fill_rgb": QWEN_MASKED_REGION_FILL_RGB,
            },
            "outside_mask_changed_after_hard_composite": changed_outside_mask,
        }
        if include_progress_history:
            output["run_report"] = {"progress_events": reporter.history}
        return output


__all__ = ["QWEN_EDIT_MODEL_ID", "QwenEditInferenceService", "parse_qwen_edit_settings"]
