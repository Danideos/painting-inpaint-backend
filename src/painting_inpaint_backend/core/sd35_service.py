"""SD3-family inpainting ControlNet service."""

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
    _reset_cuda_peak_memory,
    load_request_images,
)
from .methods import SD35_INPAINT_METHOD, RestorationMethod, normalize_method
from .model_loading import env_flag, resolve_model_load_target
from .progress import ProgressReporter
from .sd_inpaint_base import LoadedSDPipeline

LOGGER = logging.getLogger(__name__)

SD35_BASE_MODEL_ID = "stabilityai/stable-diffusion-3-medium-diffusers"
SD35_CONTROLNET_MODEL_ID = "alimama-creative/SD3-Controlnet-Inpainting"


@dataclass(frozen=True)
class SD35InpaintSettings:
    """Validated scalar settings for the SD3 inpainting ControlNet route."""

    method: RestorationMethod
    prompt: str
    negative_prompt: str
    guidance_scale: float
    controlnet_conditioning_scale: float
    num_inference_steps: int
    seed: int | None
    output_format: str
    max_sequence_length: int


def parse_sd35_inpaint_settings(payload: dict[str, Any]) -> SD35InpaintSettings:
    """Validate request scalar parameters for the SD3 inpainting ControlNet route."""

    try:
        method = normalize_method(payload.get("method"))
    except ValueError as exc:
        raise WorkerInputError(str(exc)) from exc
    if method != SD35_INPAINT_METHOD:
        raise WorkerInputError(f"The service only accepts method={SD35_INPAINT_METHOD!r}.")

    if "prompt" not in payload:
        raise WorkerInputError("prompt is required.")
    prompt = payload["prompt"]
    if not isinstance(prompt, str):
        raise WorkerInputError("prompt must be a string.")

    guidance_scale = _optional_float(payload, "guidance_scale", 7.0)
    if guidance_scale < 0:
        raise WorkerInputError("guidance_scale must be non-negative.")

    controlnet_conditioning_scale = _optional_float(
        payload,
        "controlnet_conditioning_scale",
        0.95,
    )
    if controlnet_conditioning_scale < 0:
        raise WorkerInputError("controlnet_conditioning_scale must be non-negative.")

    num_inference_steps = _optional_int(payload, "num_inference_steps", 28)
    if num_inference_steps <= 0:
        raise WorkerInputError("num_inference_steps must be positive.")

    max_sequence_length = _optional_int(payload, "max_sequence_length", 256)
    if max_sequence_length <= 0:
        raise WorkerInputError("max_sequence_length must be positive.")

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

    return SD35InpaintSettings(
        method=method,
        prompt=prompt,
        negative_prompt=str(payload.get("negative_prompt", "")),
        guidance_scale=guidance_scale,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        num_inference_steps=num_inference_steps,
        seed=seed,
        output_format=output_format,
        max_sequence_length=max_sequence_length,
    )


class SD35InferenceService:
    """Lazy-loading SD3 inpainting ControlNet service.

    The public method name is ``sd35_inpaint`` because it is the requested comparison
    slot. The model metadata reports the actual upstream route: Alimama's inpainting
    ControlNet is published for SD3-medium, not SD3.5-Large.
    """

    method = SD35_INPAINT_METHOD
    base_model_id = SD35_BASE_MODEL_ID
    controlnet_model_id = SD35_CONTROLNET_MODEL_ID
    pipeline_display_name = "SD3 inpainting ControlNet"

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
        from diffusers.models.controlnets.controlnet_sd3 import SD3ControlNetModel
        from diffusers.pipelines import StableDiffusion3ControlNetInpaintingPipeline

        if reporter is not None:
            reporter.emit(
                "model_load_start",
                stage="model",
                message=f"Loading {self.pipeline_display_name} pipeline.",
                metadata={
                    "method": self.method,
                    "base_model_id": self.base_model_id,
                    "controlnet_model_id": self.controlnet_model_id,
                },
            )
        timings: dict[str, float] = {}
        resolve_started = time.perf_counter()
        base_resolution = resolve_model_load_target(self.base_model_id, env_name="MODEL_PATH")
        controlnet_resolution = resolve_model_load_target(
            self.controlnet_model_id,
            env_name="CONTROLNET_PATH",
        )
        timings["model_path_discovery_seconds"] = time.perf_counter() - resolve_started

        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        common_kwargs: dict[str, Any] = {"torch_dtype": torch.float16}
        if token:
            common_kwargs["token"] = token

        controlnet_kwargs = {
            **common_kwargs,
            "local_files_only": controlnet_resolution.local_files_only,
            "use_safetensors": True,
            "extra_conditioning_channels": 1,
        }
        load_started = time.perf_counter()
        controlnet = SD3ControlNetModel.from_pretrained(
            controlnet_resolution.load_target,
            **controlnet_kwargs,
        )
        timings["controlnet_from_pretrained_seconds"] = time.perf_counter() - load_started

        pipe_kwargs = {
            **common_kwargs,
            "local_files_only": base_resolution.local_files_only,
            "controlnet": controlnet,
        }
        load_started = time.perf_counter()
        pipe = StableDiffusion3ControlNetInpaintingPipeline.from_pretrained(
            base_resolution.load_target,
            **pipe_kwargs,
        )
        timings["base_from_pretrained_seconds"] = time.perf_counter() - load_started

        for name in ("text_encoder", "text_encoder_2", "text_encoder_3", "controlnet"):
            component = getattr(pipe, name, None)
            if component is not None and hasattr(component, "to"):
                component.to(torch.float16)

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
            "base_model_id": self.base_model_id,
            "base_load_target": base_resolution.load_target,
            "base_source": base_resolution.source,
            "base_local_files_only": base_resolution.local_files_only,
            "base_snapshot_path": base_resolution.snapshot_path,
            "controlnet_model_id": self.controlnet_model_id,
            "controlnet_load_target": controlnet_resolution.load_target,
            "controlnet_source": controlnet_resolution.source,
            "controlnet_local_files_only": controlnet_resolution.local_files_only,
            "controlnet_snapshot_path": controlnet_resolution.snapshot_path,
            "torch_dtype": str(torch.float16),
            "device_mode": device_mode,
            "pipeline_class": type(pipe).__name__,
            "controlnet_class": type(controlnet).__name__,
            "mask_route": "native_control_image_and_control_mask",
            "compatibility_note": (
                "Alimama publishes SD3-Controlnet-Inpainting for SD3-medium; "
                "no Alimama SD3.5-Large inpainting ControlNet was found."
            ),
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
            torch_dtype=torch.float16,
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
        settings = parse_sd35_inpaint_settings(payload)
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
            "negative_prompt": settings.negative_prompt,
            "control_image": image,
            "control_mask": mask,
            "height": image.height,
            "width": image.width,
            "num_inference_steps": settings.num_inference_steps,
            "guidance_scale": settings.guidance_scale,
            "controlnet_conditioning_scale": settings.controlnet_conditioning_scale,
            "max_sequence_length": settings.max_sequence_length,
        }
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
                    "guidance_scale": settings.guidance_scale,
                    "controlnet_conditioning_scale": settings.controlnet_conditioning_scale,
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
                "guidance_scale": settings.guidance_scale,
                "controlnet_conditioning_scale": settings.controlnet_conditioning_scale,
                "num_inference_steps": settings.num_inference_steps,
                "max_sequence_length": settings.max_sequence_length,
                "seed": settings.seed,
                "mask_coverage": mask_fraction,
            },
            "outside_mask_changed_after_hard_composite": changed_outside_mask,
        }
        if include_progress_history:
            output["run_report"] = {"progress_events": reporter.history}
        return output


__all__ = [
    "SD35_BASE_MODEL_ID",
    "SD35_CONTROLNET_MODEL_ID",
    "SD35InferenceService",
    "parse_sd35_inpaint_settings",
]
