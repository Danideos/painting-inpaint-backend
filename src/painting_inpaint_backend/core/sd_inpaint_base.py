"""Shared Stable Diffusion inpainting service implementation."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from PIL import Image

from .image_helpers import hard_composite, mask_coverage, outside_mask_changed
from .image_io import image_to_base64, normalize_output_format
from .inference import (
    WorkerInputError,
    _add_inference_progress_callback,
    _cuda_memory_stats,
    _effective_step_total,
    _optional_float,
    _optional_int,
    _payload_flag,
    _pipeline_accepts_argument,
    _reset_cuda_peak_memory,
    load_request_images,
)
from .methods import RestorationMethod, normalize_method
from .model_loading import env_flag, resolve_model_load_target
from .progress import ProgressReporter

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SDInpaintSettings:
    """Validated scalar settings for SD inpainting baselines."""

    method: RestorationMethod
    prompt: str
    negative_prompt: str
    guidance_scale: float
    num_inference_steps: int
    strength: float
    seed: int | None
    output_format: str


@dataclass(frozen=True)
class LoadedSDPipeline:
    """Loaded SD pipeline plus JSON-safe metadata."""

    pipe: Any
    torch: Any
    torch_dtype: Any
    model: dict[str, Any]
    supports_negative_prompt: bool
    timings: dict[str, float]


def parse_sd_inpaint_settings(
    payload: dict[str, Any],
    *,
    expected_method: RestorationMethod,
) -> SDInpaintSettings:
    """Validate request scalar parameters for SD inpainting baselines."""

    try:
        method = normalize_method(payload.get("method"))
    except ValueError as exc:
        raise WorkerInputError(str(exc)) from exc
    if method != expected_method:
        raise WorkerInputError(f"The service only accepts method={expected_method!r}.")

    if "prompt" not in payload:
        raise WorkerInputError("prompt is required.")
    prompt = payload["prompt"]
    if not isinstance(prompt, str):
        raise WorkerInputError("prompt must be a string.")

    guidance_scale = _optional_float(payload, "guidance_scale", 7.5)
    if guidance_scale < 0:
        raise WorkerInputError("guidance_scale must be non-negative.")

    num_inference_steps = _optional_int(payload, "num_inference_steps", 30)
    if num_inference_steps <= 0:
        raise WorkerInputError("num_inference_steps must be positive.")

    strength = _optional_float(payload, "strength", 1.0)
    if not 0 <= strength <= 1:
        raise WorkerInputError("strength must be between 0 and 1.")

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

    return SDInpaintSettings(
        method=method,
        prompt=prompt,
        negative_prompt=str(payload.get("negative_prompt", "")),
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        strength=strength,
        seed=seed,
        output_format=output_format,
    )


class SDInpaintServiceBase:
    """Lazy-loading Stable Diffusion inpainting baseline service."""

    method: RestorationMethod
    model_id: str
    torch_dtype_name: str
    pipeline_display_name: str

    def __init__(self) -> None:
        self._loaded: LoadedSDPipeline | None = None

    @property
    def loaded(self) -> LoadedSDPipeline:
        return self.get_loaded()

    def pipeline_class(self) -> Any:
        raise NotImplementedError

    def from_pretrained_kwargs(self, torch_dtype: Any) -> dict[str, Any]:
        del torch_dtype
        return {}

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

    def _torch_dtype(self, torch: Any) -> Any:
        if self.torch_dtype_name == "bfloat16":
            return torch.bfloat16
        if self.torch_dtype_name == "float16":
            return torch.float16
        raise RuntimeError(f"Unsupported torch dtype: {self.torch_dtype_name}")

    def _load_pipeline(self, *, reporter: ProgressReporter | None = None) -> LoadedSDPipeline:
        import torch

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

        torch_dtype = self._torch_dtype(torch)
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        kwargs: dict[str, Any] = {
            "torch_dtype": torch_dtype,
            "local_files_only": resolution.local_files_only,
            **self.from_pretrained_kwargs(torch_dtype),
        }
        if token:
            kwargs["token"] = token

        load_started = time.perf_counter()
        pipe = self.pipeline_class().from_pretrained(resolution.load_target, **kwargs)
        timings["from_pretrained_seconds"] = time.perf_counter() - load_started

        device_started = time.perf_counter()
        if env_flag("ENABLE_MODEL_CPU_OFFLOAD", default=True) and hasattr(
            pipe, "enable_model_cpu_offload"
        ):
            pipe.enable_model_cpu_offload()
            device_mode = "model_cpu_offload"
        elif torch.cuda.is_available():
            pipe.to("cuda")
            device_mode = "cuda"
        else:
            pipe.to("cpu")
            device_mode = "cpu"

        if env_flag("ENABLE_ATTENTION_SLICING", default=True) and hasattr(
            pipe,
            "enable_attention_slicing",
        ):
            pipe.enable_attention_slicing()
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
            "torch_dtype": str(torch_dtype),
            "device_mode": device_mode,
            "pipeline_class": type(pipe).__name__,
            "baseline": True,
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
            torch_dtype=torch_dtype,
            model=model,
            supports_negative_prompt=_pipeline_accepts_argument(pipe, "negative_prompt"),
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
        settings = parse_sd_inpaint_settings(payload, expected_method=self.method)
        reporter.emit(
            "input_decode_start",
            stage="input",
            message="Decoding request image and mask.",
        )
        image, mask = load_request_images(payload)
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
            },
        )

        loaded = self.get_loaded(reporter=reporter)
        pipe = loaded.pipe
        torch = loaded.torch

        generator = None
        if settings.seed is not None:
            generator_device = "cuda" if torch.cuda.is_available() else "cpu"
            generator = torch.Generator(device=generator_device).manual_seed(settings.seed)

        call_kwargs: dict[str, Any] = {
            "prompt": settings.prompt,
            "image": image,
            "mask_image": mask,
            "height": image.height,
            "width": image.width,
            "strength": settings.strength,
            "num_inference_steps": settings.num_inference_steps,
            "guidance_scale": settings.guidance_scale,
        }
        if generator is not None:
            call_kwargs["generator"] = generator
        if loaded.supports_negative_prompt:
            call_kwargs["negative_prompt"] = settings.negative_prompt
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
                "negative_prompt_passed_to_pipeline": loaded.supports_negative_prompt,
                "strength": settings.strength,
                "guidance_scale": settings.guidance_scale,
                "num_inference_steps": settings.num_inference_steps,
                "seed": settings.seed,
                "mask_coverage": mask_fraction,
            },
            "outside_mask_changed_after_hard_composite": changed_outside_mask,
        }
        if include_progress_history:
            output["run_report"] = {"progress_events": reporter.history}
        return output
