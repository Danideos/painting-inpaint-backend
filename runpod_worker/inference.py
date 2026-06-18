"""Inference orchestration for the FLUX Fill RunPod worker."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from PIL import Image

from painting_inpaint.compositing import hard_composite, outside_mask_changed
from painting_inpaint.masks import binarize_mask, ensure_same_size, mask_coverage

from .image_io import image_to_base64, load_request_image, normalize_output_format
from .model_loading import LoadedPipeline, load_pipeline, set_lora_scale
from .partial_noise import normalize_partial_noise, set_partial_noise

LOGGER = logging.getLogger(__name__)

DIFFUSERS_STRENGTH_FOR_CALL = 1.0


class WorkerInputError(ValueError):
    """Raised when a request payload is invalid."""


@dataclass(frozen=True)
class RequestSettings:
    """Validated scalar inference settings."""

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

    if "prompt" not in payload:
        raise WorkerInputError("prompt is required.")
    prompt = payload["prompt"]
    if not isinstance(prompt, str):
        raise WorkerInputError("prompt must be a string.")

    try:
        partial_noise = normalize_partial_noise(payload.get("partial_noise"))
    except ValueError as exc:
        raise WorkerInputError(str(exc)) from exc
    guidance_scale = _optional_float(payload, "guidance_scale", 30.0)
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
        raise WorkerInputError(
            f"image and mask must have identical dimensions. {exc}"
        ) from exc
    return image.convert("RGB"), binarize_mask(mask)


class FluxFillWorker:
    """Lazy-loading worker service reused across warm RunPod jobs."""

    def __init__(self) -> None:
        self._loaded: LoadedPipeline | None = None

    @property
    def loaded(self) -> LoadedPipeline:
        if self._loaded is None:
            started = time.perf_counter()
            self._loaded = load_pipeline()
            LOGGER.info("Pipeline ready in %.3fs", time.perf_counter() - started)
        return self._loaded

    def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        settings = parse_request_settings(payload)
        image, mask = load_request_images(payload)
        loaded = self.loaded
        pipe = loaded.pipe
        torch = loaded.torch

        set_partial_noise(pipe, settings.partial_noise)
        lora_strength_mode = set_lora_scale(
            pipe,
            adapter_name=loaded.lora.get("adapter_name"),
            lora_scale=settings.lora_scale,
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

        with TemporaryDirectory(prefix="flux_fill_worker_") as temp_dir:
            temp_path = Path(temp_dir)
            inference_started = time.perf_counter()
            with torch.inference_mode():
                raw = pipe(**call_kwargs).images[0].convert("RGB")
            inference_seconds = time.perf_counter() - inference_started

            if raw.size != image.size:
                raw = raw.resize(image.size, Image.Resampling.LANCZOS)
            composite = hard_composite(image, raw, mask)
            changed_outside_mask = outside_mask_changed(image, composite, mask)
            if changed_outside_mask:
                raise RuntimeError("Hard composite changed pixels outside the edit mask.")

            output_path = temp_path / f"composite.{settings.output_format}"
            composite.save(output_path)
            encoded = image_to_base64(composite, output_format=settings.output_format)

        schedule_debug = getattr(pipe, "_last_schedule_debug", {}) or {}
        latent_debug = getattr(pipe, "_last_latent_init_debug", {}) or {}
        return {
            "image_base64": encoded,
            "output_format": settings.output_format,
            "width": composite.width,
            "height": composite.height,
            "mask_convention": "white = inpaint/edit, black = preserve",
            "timings": {
                **loaded.timings,
                "inference_seconds": inference_seconds,
            },
            "model": loaded.model,
            "lora": {
                **loaded.lora,
                "requested_scale": settings.lora_scale,
                "request_strength_mode": lora_strength_mode,
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


_WORKER = FluxFillWorker()


def run_job_input(payload: dict[str, Any]) -> dict[str, Any]:
    """Run one validated RunPod input payload."""

    if not isinstance(payload, dict):
        raise WorkerInputError("RunPod job input must be a JSON object.")
    return _WORKER.run(payload)
