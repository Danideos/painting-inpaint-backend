"""Provider-neutral FLUX Fill inference orchestration."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from PIL import Image

from .image_helpers import (
    binarize_mask,
    ensure_same_size,
    hard_composite,
    mask_coverage,
    outside_mask_changed,
)
from .image_io import image_to_base64, load_request_image, normalize_output_format
from .methods import FLUX_FILL_METHOD, RestorationMethod, normalize_method
from .model_loading import LoadedPipeline, load_pipeline, set_lora_scale
from .partial_noise import normalize_partial_noise, set_partial_noise
from .progress import ProgressReporter

LOGGER = logging.getLogger(__name__)

DIFFUSERS_STRENGTH_FOR_CALL = 1.0


class WorkerInputError(ValueError):
    """Raised when a request payload is invalid."""


def _payload_flag(payload: dict[str, Any], key: str, *, default: bool = False) -> bool:
    value = payload.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _pipeline_accepts_argument(pipe: Any, parameter_name: str) -> bool:
    import inspect

    try:
        signature = inspect.signature(pipe.__call__)
    except Exception:
        return False
    if parameter_name in signature.parameters:
        return True
    return any(
        param.kind is inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()
    )


def _json_timestep(timestep: Any) -> Any:
    if timestep is None or isinstance(timestep, str | int | float | bool):
        return timestep
    try:
        if hasattr(timestep, "detach"):
            timestep = timestep.detach()
        if hasattr(timestep, "cpu"):
            timestep = timestep.cpu()
        if hasattr(timestep, "item"):
            return timestep.item()
    except Exception:
        pass
    return str(timestep)


def _cuda_memory_stats(torch: Any, *, prefix: str = "") -> dict[str, Any]:
    cuda = getattr(torch, "cuda", None)
    if cuda is None:
        return {f"{prefix}cuda_available": False}
    try:
        cuda_available = bool(cuda.is_available())
    except Exception:
        cuda_available = False
    stats: dict[str, Any] = {f"{prefix}cuda_available": cuda_available}
    if not cuda_available:
        return stats

    def _bytes_to_mb(value: Any) -> float | None:
        try:
            return round(float(value) / (1024 * 1024), 2)
        except (TypeError, ValueError):
            return None

    try:
        device_index = int(cuda.current_device())
        stats[f"{prefix}cuda_device_index"] = device_index
        if hasattr(cuda, "get_device_name"):
            stats[f"{prefix}cuda_device_name"] = cuda.get_device_name(device_index)
    except Exception:
        device_index = None

    metric_names = (
        ("memory_allocated", "allocated_mb"),
        ("memory_reserved", "reserved_mb"),
        ("max_memory_allocated", "peak_allocated_mb"),
        ("max_memory_reserved", "peak_reserved_mb"),
    )
    for method_name, output_name in metric_names:
        method = getattr(cuda, method_name, None)
        if method is None:
            continue
        try:
            value = method(device_index) if device_index is not None else method()
        except TypeError:
            value = method()
        except Exception:
            continue
        stats[f"{prefix}{output_name}"] = _bytes_to_mb(value)
    return stats


def _reset_cuda_peak_memory(torch: Any) -> None:
    cuda = getattr(torch, "cuda", None)
    if cuda is None:
        return
    try:
        if cuda.is_available() and hasattr(cuda, "reset_peak_memory_stats"):
            cuda.reset_peak_memory_stats()
    except Exception:
        return


def _effective_step_total(pipe: Any, settings: RequestSettings) -> int:
    schedule_debug = getattr(pipe, "_last_schedule_debug", {}) or {}
    total = (
        schedule_debug.get("effective_num_steps")
        or schedule_debug.get("num_inference_steps")
        or settings.num_inference_steps
    )
    try:
        parsed = int(total)
    except (TypeError, ValueError):
        parsed = settings.num_inference_steps
    return max(parsed, 1)


def _add_inference_progress_callback(
    *,
    call_kwargs: dict[str, Any],
    pipe: Any,
    settings: RequestSettings,
    reporter: ProgressReporter,
) -> None:
    if not _pipeline_accepts_argument(pipe, "callback_on_step_end"):
        reporter.emit(
            "inference_step_logging_unavailable",
            stage="inference",
            message="Pipeline does not support per-step progress callbacks.",
            metadata={"pipeline_class": type(pipe).__name__},
        )
        return

    def _callback(
        callback_pipe: Any,
        step_index: int,
        timestep: Any,
        callback_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        total = _effective_step_total(callback_pipe or pipe, settings)
        current = min(max(int(step_index) + 1, 1), total)
        reporter.emit(
            "inference_step",
            stage="inference",
            message=f"Inference step {current}/{total}.",
            progress={"current": current, "total": total},
            metadata={"timestep": _json_timestep(timestep)},
        )
        return callback_kwargs

    call_kwargs["callback_on_step_end"] = _callback
    if _pipeline_accepts_argument(pipe, "callback_on_step_end_tensor_inputs"):
        call_kwargs["callback_on_step_end_tensor_inputs"] = ["latents"]


@dataclass(frozen=True)
class RequestSettings:
    """Validated scalar inference settings."""

    method: RestorationMethod
    prompt: str
    negative_prompt: str
    partial_noise: float
    guidance_scale: float
    num_inference_steps: int
    seed: int | None
    lora_scale: float
    output_format: str
    max_sequence_length: int


def _optional_float(payload: dict[str, Any], key: str, default: float) -> float:
    value = payload.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise WorkerInputError(f"{key} must be a number.") from exc


def _optional_int(payload: dict[str, Any], key: str, default: int) -> int:
    value = payload.get(key, default)
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise WorkerInputError(f"{key} must be an integer.") from exc
    return parsed


def parse_request_settings(payload: dict[str, Any]) -> RequestSettings:
    """Validate request scalar parameters."""

    try:
        method = normalize_method(payload.get("method"))
    except ValueError as exc:
        raise WorkerInputError(str(exc)) from exc

    if "prompt" not in payload:
        raise WorkerInputError("prompt is required.")
    prompt = payload["prompt"]
    if not isinstance(prompt, str):
        raise WorkerInputError("prompt must be a string.")

    try:
        partial_noise = normalize_partial_noise(payload.get("partial_noise"))
    except ValueError as exc:
        raise WorkerInputError(str(exc)) from exc
    default_guidance = 1.5 if method == "flux_canny_lanpaint" else 30.0
    guidance_scale = _optional_float(payload, "guidance_scale", default_guidance)
    if guidance_scale < 0:
        raise WorkerInputError("guidance_scale must be non-negative.")

    num_inference_steps = _optional_int(payload, "num_inference_steps", 28)
    if num_inference_steps <= 0:
        raise WorkerInputError("num_inference_steps must be positive.")

    max_sequence_length = _optional_int(payload, "max_sequence_length", 512)
    if max_sequence_length <= 0:
        raise WorkerInputError("max_sequence_length must be positive.")

    seed_value = payload.get("seed")
    seed = None
    if seed_value is not None:
        try:
            seed = int(seed_value)
        except (TypeError, ValueError) as exc:
            raise WorkerInputError("seed must be an integer when provided.") from exc

    lora_scale = _optional_float(payload, "lora_scale", 1.0)
    if lora_scale < 0:
        raise WorkerInputError("lora_scale must be non-negative.")

    try:
        output_format = normalize_output_format(payload.get("output_format", "png"))
    except ValueError as exc:
        raise WorkerInputError(str(exc)) from exc

    return RequestSettings(
        method=method,
        prompt=prompt,
        negative_prompt=str(payload.get("negative_prompt", "")),
        partial_noise=partial_noise,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        seed=seed,
        lora_scale=lora_scale,
        output_format=output_format,
        max_sequence_length=max_sequence_length,
    )


def load_request_images(payload: dict[str, Any]) -> tuple[Image.Image, Image.Image]:
    """Load and validate the request image and edit mask."""

    try:
        image = load_request_image(
            payload,
            label="image",
            url_key="image_url",
            base64_key="image_base64",
            mode="RGB",
        )
        mask = load_request_image(
            payload,
            label="mask",
            url_key="mask_url",
            base64_key="mask_base64",
            mode="L",
        )
    except ValueError as exc:
        raise WorkerInputError(str(exc)) from exc
    try:
        ensure_same_size([image, mask])
    except ValueError as exc:
        raise WorkerInputError(f"image and mask must have identical dimensions. {exc}") from exc
    return image.convert("RGB"), binarize_mask(mask)


class InferenceService:
    """Lazy-loading FLUX Fill service reused across provider requests."""

    def __init__(self) -> None:
        self._loaded: LoadedPipeline | None = None

    @property
    def loaded(self) -> LoadedPipeline:
        return self.get_loaded()

    def get_loaded(self, *, reporter: ProgressReporter | None = None) -> LoadedPipeline:
        if self._loaded is None:
            started = time.perf_counter()
            self._loaded = load_pipeline(reporter=reporter)
            LOGGER.info("Pipeline ready in %.3fs", time.perf_counter() - started)
        elif reporter is not None:
            reporter.emit(
                "model_load_start",
                stage="model",
                message="Reusing cached FLUX Fill pipeline.",
                metadata={"cached": True},
            )
            reporter.emit(
                "model_load_done",
                stage="model",
                message="Cached FLUX Fill pipeline ready.",
                metadata={
                    "cached": True,
                    "model": self._loaded.model,
                    "lora_loaded": bool(self._loaded.lora.get("loaded")),
                },
            )
        return self._loaded

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
        settings = parse_request_settings(payload)
        if settings.method != FLUX_FILL_METHOD:
            raise WorkerInputError(
                "The FLUX Fill service only accepts method='flux_fill'. "
                "FLUX-Canny/LanPaint is available through the Modal Canny service."
            )
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

        set_partial_noise(pipe, settings.partial_noise)
        reporter.emit(
            "lora_scale_apply_start",
            stage="lora",
            message="Applying request LoRA scale.",
            metadata={
                "requested_scale": settings.lora_scale,
                "adapter_loaded": bool(loaded.lora.get("loaded")),
                "adapter_name": loaded.lora.get("adapter_name"),
            },
        )
        lora_scale_result = set_lora_scale(
            pipe,
            adapter_name=(loaded.lora.get("adapter_name") if loaded.lora.get("loaded") else None),
            lora_scale=settings.lora_scale,
        )
        reporter.emit(
            "lora_scale_apply_done",
            stage="lora",
            message="Request LoRA scale applied.",
            metadata={
                "requested_scale": settings.lora_scale,
                "effective_scale": lora_scale_result["effective_scale"],
                "mode": lora_scale_result["mode"],
            },
        )

        generator = None
        if settings.seed is not None:
            generator = torch.Generator(device="cpu").manual_seed(settings.seed)

        call_kwargs: dict[str, Any] = {
            "prompt": settings.prompt,
            "image": image,
            "mask_image": mask,
            "height": image.height,
            "width": image.width,
            "strength": DIFFUSERS_STRENGTH_FOR_CALL,
            "num_inference_steps": settings.num_inference_steps,
            "guidance_scale": settings.guidance_scale,
            "max_sequence_length": settings.max_sequence_length,
        }
        if generator is not None:
            call_kwargs["generator"] = generator
        if loaded.supports_negative_prompt:
            call_kwargs["negative_prompt"] = settings.negative_prompt
        _add_inference_progress_callback(
            call_kwargs=call_kwargs,
            pipe=pipe,
            settings=settings,
            reporter=reporter,
        )

        with TemporaryDirectory(prefix="flux_fill_worker_") as temp_dir:
            temp_path = Path(temp_dir)
            memory_before_inference = _cuda_memory_stats(
                torch,
                prefix="pre_inference_",
            )
            _reset_cuda_peak_memory(torch)
            reporter.emit(
                "inference_start",
                stage="inference",
                message="Starting FLUX Fill inference.",
                progress={"current": 0, "total": settings.num_inference_steps},
                metadata={
                    "num_inference_steps": settings.num_inference_steps,
                    "partial_noise": settings.partial_noise,
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
            schedule_debug = getattr(pipe, "_last_schedule_debug", {}) or {}
            reporter.emit(
                "inference_done",
                stage="inference",
                message="FLUX Fill inference completed.",
                progress={
                    "current": _effective_step_total(pipe, settings),
                    "total": _effective_step_total(pipe, settings),
                },
                metadata={
                    "inference_seconds": inference_seconds,
                    "schedule_debug": schedule_debug,
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

        latent_debug = getattr(pipe, "_last_latent_init_debug", {}) or {}
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
            "lora": {
                **loaded.lora,
                "requested_scale": settings.lora_scale,
                "effective_scale": lora_scale_result["effective_scale"],
                "request_strength_mode": lora_scale_result["mode"],
            },
            "inference_settings": {
                "prompt": settings.prompt,
                "negative_prompt": settings.negative_prompt,
                "negative_prompt_passed_to_pipeline": loaded.supports_negative_prompt,
                "partial_noise": settings.partial_noise,
                "diffusers_strength_for_call": DIFFUSERS_STRENGTH_FOR_CALL,
                "guidance_scale": settings.guidance_scale,
                "num_inference_steps": settings.num_inference_steps,
                "max_sequence_length": settings.max_sequence_length,
                "seed": settings.seed,
                "mask_coverage": mask_coverage(mask),
            },
            "schedule_debug": schedule_debug,
            "latent_init_debug": latent_debug,
            "outside_mask_changed_after_hard_composite": changed_outside_mask,
        }
        if include_progress_history:
            output["run_report"] = {"progress_events": reporter.history}
        return output
