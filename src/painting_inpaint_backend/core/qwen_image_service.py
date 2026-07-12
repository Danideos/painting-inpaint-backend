"""Qwen-Image text-to-image experiment service."""

from __future__ import annotations

import logging
import os
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
from PIL import Image

from .image_helpers import hard_composite, mask_coverage, outside_mask_changed
from .image_io import image_to_base64, max_image_pixels, normalize_output_format
from .inference import (
    WorkerInputError,
    _add_inference_progress_callback,
    _cuda_memory_stats,
    _effective_step_total,
    _json_timestep,
    _optional_float,
    _optional_int,
    _payload_flag,
    _reset_cuda_peak_memory,
    load_request_images,
)
from .methods import (
    QWEN_IMAGE_LANPAINT_METHOD,
    QWEN_IMAGE_METHOD,
    RestorationMethod,
    normalize_method,
)
from .model_loading import env_flag, resolve_model_load_target
from .progress import ProgressReporter
from .qwen_edit_service import (
    QWEN_LANPAINT_BETA,
    QWEN_LANPAINT_FINAL_OUTER_STEPS_WITHOUT_INNER,
    QWEN_LANPAINT_FRICTION,
    QWEN_LANPAINT_INNER_STEPS,
    QWEN_LANPAINT_LAMBDA,
    QWEN_LANPAINT_STEP_SIZE,
    _free_qwen_offloaded_modules,
    _make_qwen_current_times,
    _qwen_noise_scaling,
    make_qwen_lanpaint_source_by_strategy,
    parse_qwen_source_strategy,
)
from .sd_inpaint_base import LoadedSDPipeline

LOGGER = logging.getLogger(__name__)

QWEN_IMAGE_MODEL_ID = "Qwen/Qwen-Image"
QWEN_IMAGE_BACKEND_REVISION = "qwen-image-text-to-image-v1"
QWEN_IMAGE_LANPAINT_BACKEND_REVISION = "qwen-image-lanpaint-v1"
QWEN_IMAGE_DEFAULT_SIZE = 1024
QWEN_IMAGE_DIMENSION_MULTIPLE = 16


@dataclass(frozen=True)
class QwenImageSettings:
    """Validated scalar settings for Qwen-Image text-to-image generation."""

    method: RestorationMethod
    prompt: str
    negative_prompt: str
    true_cfg_scale: float
    num_inference_steps: int
    width: int
    height: int
    seed: int | None
    output_format: str
    max_sequence_length: int


@dataclass(frozen=True)
class QwenImageLanPaintSettings:
    """Validated scalar settings for Qwen-Image LanPaint masked generation."""

    method: RestorationMethod
    prompt: str
    negative_prompt: str
    true_cfg_scale: float
    num_inference_steps: int
    seed: int | None
    output_format: str
    max_sequence_length: int
    qwen_source_strategy: str
    lanpaint_inner_steps: int
    lanpaint_friction: float
    lanpaint_lambda: float
    lanpaint_beta: float
    lanpaint_step_size: float
    lanpaint_final_outer_steps_without_inner: int


def _parse_dimension(payload: dict[str, Any], key: str) -> int:
    value = _optional_int(payload, key, QWEN_IMAGE_DEFAULT_SIZE)
    if value <= 0:
        raise WorkerInputError(f"{key} must be positive.")
    if value % QWEN_IMAGE_DIMENSION_MULTIPLE != 0:
        raise WorkerInputError(
            f"{key} must be divisible by {QWEN_IMAGE_DIMENSION_MULTIPLE}."
        )
    return value


def _parse_prompt(payload: dict[str, Any]) -> str:
    if "prompt" not in payload:
        raise WorkerInputError("prompt is required.")
    prompt = payload["prompt"]
    if not isinstance(prompt, str) or not prompt.strip():
        raise WorkerInputError("prompt must be a non-empty string.")
    return prompt


def _parse_true_cfg_scale(payload: dict[str, Any]) -> float:
    true_cfg_default = payload.get("guidance_scale", 4.0)
    if true_cfg_default is None:
        true_cfg_default = 4.0
    true_cfg_scale = _optional_float(payload, "true_cfg_scale", true_cfg_default)
    if true_cfg_scale < 0:
        raise WorkerInputError("true_cfg_scale must be non-negative.")
    return true_cfg_scale


def _parse_seed(payload: dict[str, Any]) -> int | None:
    seed_value = payload.get("seed")
    if seed_value is None:
        return None
    try:
        return int(seed_value)
    except (TypeError, ValueError) as exc:
        raise WorkerInputError("seed must be an integer when provided.") from exc


def parse_qwen_image_settings(payload: dict[str, Any]) -> QwenImageSettings:
    """Validate request scalar parameters for Qwen-Image generation."""

    try:
        method = normalize_method(payload.get("method"))
    except ValueError as exc:
        raise WorkerInputError(str(exc)) from exc
    if method != QWEN_IMAGE_METHOD:
        raise WorkerInputError(f"The service only accepts method={QWEN_IMAGE_METHOD!r}.")

    prompt = _parse_prompt(payload)
    true_cfg_scale = _parse_true_cfg_scale(payload)

    num_inference_steps = _optional_int(payload, "num_inference_steps", 50)
    if num_inference_steps <= 0:
        raise WorkerInputError("num_inference_steps must be positive.")

    width = _parse_dimension(payload, "width")
    height = _parse_dimension(payload, "height")
    pixels = width * height
    limit = max_image_pixels()
    if pixels > limit:
        raise WorkerInputError(
            f"Requested output dimensions are too large: {width}x{height} "
            f"({pixels} pixels) exceeds MAX_IMAGE_PIXELS={limit}."
        )

    max_sequence_length = _optional_int(payload, "max_sequence_length", 512)
    if max_sequence_length <= 0:
        raise WorkerInputError("max_sequence_length must be positive.")

    seed = _parse_seed(payload)

    try:
        output_format = normalize_output_format(payload.get("output_format", "png"))
    except ValueError as exc:
        raise WorkerInputError(str(exc)) from exc

    return QwenImageSettings(
        method=method,
        prompt=prompt,
        negative_prompt=str(payload.get("negative_prompt", " ")),
        true_cfg_scale=true_cfg_scale,
        num_inference_steps=num_inference_steps,
        width=width,
        height=height,
        seed=seed,
        output_format=output_format,
        max_sequence_length=max_sequence_length,
    )


def parse_qwen_image_lanpaint_settings(payload: dict[str, Any]) -> QwenImageLanPaintSettings:
    """Validate request scalar parameters for Qwen-Image LanPaint inpainting."""

    try:
        method = normalize_method(payload.get("method"))
    except ValueError as exc:
        raise WorkerInputError(str(exc)) from exc
    if method != QWEN_IMAGE_LANPAINT_METHOD:
        raise WorkerInputError(
            f"The service only accepts method={QWEN_IMAGE_LANPAINT_METHOD!r}."
        )

    prompt = _parse_prompt(payload)
    true_cfg_scale = _parse_true_cfg_scale(payload)

    num_inference_steps = _optional_int(payload, "num_inference_steps", 50)
    if num_inference_steps <= 0:
        raise WorkerInputError("num_inference_steps must be positive.")

    max_sequence_length = _optional_int(payload, "max_sequence_length", 512)
    if max_sequence_length <= 0:
        raise WorkerInputError("max_sequence_length must be positive.")

    lanpaint_inner_steps = _optional_int(
        payload,
        "lanpaint_inner_steps",
        QWEN_LANPAINT_INNER_STEPS,
    )
    if lanpaint_inner_steps < 0:
        raise WorkerInputError("lanpaint_inner_steps must be non-negative.")

    lanpaint_final_without_inner = _optional_int(
        payload,
        "lanpaint_final_outer_steps_without_inner",
        QWEN_LANPAINT_FINAL_OUTER_STEPS_WITHOUT_INNER,
    )
    if lanpaint_final_without_inner < 0:
        raise WorkerInputError(
            "lanpaint_final_outer_steps_without_inner must be non-negative."
        )

    lanpaint_friction = _optional_float(payload, "lanpaint_friction", QWEN_LANPAINT_FRICTION)
    if lanpaint_friction < 0:
        raise WorkerInputError("lanpaint_friction must be non-negative.")

    lanpaint_lambda = _optional_float(payload, "lanpaint_lambda", QWEN_LANPAINT_LAMBDA)
    if lanpaint_lambda < 0:
        raise WorkerInputError("lanpaint_lambda must be non-negative.")

    lanpaint_beta = _optional_float(payload, "lanpaint_beta", QWEN_LANPAINT_BETA)
    if lanpaint_beta < 0:
        raise WorkerInputError("lanpaint_beta must be non-negative.")

    lanpaint_step_size = _optional_float(payload, "lanpaint_step_size", QWEN_LANPAINT_STEP_SIZE)
    if lanpaint_step_size <= 0:
        raise WorkerInputError("lanpaint_step_size must be greater than 0.")

    try:
        output_format = normalize_output_format(payload.get("output_format", "png"))
    except ValueError as exc:
        raise WorkerInputError(str(exc)) from exc

    return QwenImageLanPaintSettings(
        method=method,
        prompt=prompt,
        negative_prompt=str(payload.get("negative_prompt", " ")),
        true_cfg_scale=true_cfg_scale,
        num_inference_steps=num_inference_steps,
        seed=_parse_seed(payload),
        output_format=output_format,
        max_sequence_length=max_sequence_length,
        qwen_source_strategy=parse_qwen_source_strategy(payload),
        lanpaint_inner_steps=lanpaint_inner_steps,
        lanpaint_friction=lanpaint_friction,
        lanpaint_lambda=lanpaint_lambda,
        lanpaint_beta=lanpaint_beta,
        lanpaint_step_size=lanpaint_step_size,
        lanpaint_final_outer_steps_without_inner=lanpaint_final_without_inner,
    )


class QwenImageLanPaintAdapter:
    """Expose Qwen-Image text-to-image flow components to LanPaint."""

    def __init__(self, pipe: Any) -> None:
        self.pipe = pipe
        self.height: int | None = None
        self.width: int | None = None
        self.prompt_embeds = None
        self.prompt_embeds_mask = None
        self.negative_prompt_embeds = None
        self.negative_prompt_embeds_mask = None
        self.img_shapes: list[list[tuple[int, int, int]]] | None = None
        self.guidance = None
        self.do_true_cfg = False
        self.true_cfg_scale = 1.0
        self.attention_kwargs: dict[str, Any] = {}
        self.schedule_debug: dict[str, Any] = {}

    @property
    def device(self) -> Any:
        return self.pipe._execution_device

    @property
    def dtype(self) -> Any:
        return self.pipe.transformer.dtype

    def prepare_prompt(
        self,
        *,
        prompt: str,
        negative_prompt: str,
        true_cfg_scale: float,
        max_sequence_length: int,
    ) -> None:
        self.do_true_cfg = true_cfg_scale > 1 and negative_prompt is not None
        self.true_cfg_scale = true_cfg_scale
        self.prompt_embeds, self.prompt_embeds_mask = self.pipe.encode_prompt(
            prompt=prompt,
            device=self.device,
            num_images_per_prompt=1,
            max_sequence_length=max_sequence_length,
        )
        if self.do_true_cfg:
            self.negative_prompt_embeds, self.negative_prompt_embeds_mask = (
                self.pipe.encode_prompt(
                    prompt=negative_prompt,
                    device=self.device,
                    num_images_per_prompt=1,
                    max_sequence_length=max_sequence_length,
                )
            )
        if self.pipe.transformer.config.guidance_embeds:
            raise WorkerInputError(
                "guidance-distilled Qwen Image models are not supported by this LanPaint route."
            )

    def prepare_source_latents(
        self,
        *,
        source_image: Image.Image,
        generator: Any,
        num_inference_steps: int,
    ) -> tuple[Any, Any, Any, Any]:
        import torch
        from diffusers.pipelines.qwenimage.pipeline_qwenimage import (
            calculate_shift,
            retrieve_timesteps,
        )
        from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_inpaint import (
            retrieve_latents,
        )
        from diffusers.utils.torch_utils import randn_tensor

        if self.prompt_embeds is None:
            raise RuntimeError("Prompt embeds must be prepared before source latents.")

        self.width = int(source_image.width) // QWEN_IMAGE_DIMENSION_MULTIPLE
        self.width *= QWEN_IMAGE_DIMENSION_MULTIPLE
        self.height = int(source_image.height) // QWEN_IMAGE_DIMENSION_MULTIPLE
        self.height *= QWEN_IMAGE_DIMENSION_MULTIPLE
        if self.width <= 0 or self.height <= 0:
            raise WorkerInputError("image dimensions are too small for Qwen Image LanPaint.")

        resized = source_image.resize((self.width, self.height), Image.Resampling.LANCZOS)
        image_array = np.asarray(resized.convert("RGB"), dtype=np.float32) / 127.5 - 1.0
        image_tensor = (
            torch.from_numpy(image_array)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .unsqueeze(2)
            .to(device=self.device, dtype=self.prompt_embeds.dtype)
        )

        image_latents = retrieve_latents(self.pipe.vae.encode(image_tensor), generator=generator)
        latents_mean = (
            torch.tensor(self.pipe.vae.config.latents_mean)
            .view(1, self.pipe.vae.config.z_dim, 1, 1, 1)
            .to(image_latents.device, image_latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.pipe.vae.config.latents_std).view(
            1,
            self.pipe.vae.config.z_dim,
            1,
            1,
            1,
        ).to(image_latents.device, image_latents.dtype)
        image_latents = (image_latents - latents_mean) * latents_std
        image_latents = image_latents.transpose(1, 2)

        num_channels_latents = self.pipe.transformer.config.in_channels // 4
        latent_height = 2 * (self.height // (self.pipe.vae_scale_factor * 2))
        latent_width = 2 * (self.width // (self.pipe.vae_scale_factor * 2))
        source_latent = self.pipe._pack_latents(
            image_latents,
            1,
            num_channels_latents,
            latent_height,
            latent_width,
        )
        noise = randn_tensor(
            (1, 1, num_channels_latents, latent_height, latent_width),
            generator=generator,
            device=self.device,
            dtype=self.prompt_embeds.dtype,
        )
        noise = self.pipe._pack_latents(noise, 1, num_channels_latents, latent_height, latent_width)

        sigmas = np.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps)
        image_seq_len = source_latent.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.pipe.scheduler.config.get("base_image_seq_len", 256),
            self.pipe.scheduler.config.get("max_image_seq_len", 4096),
            self.pipe.scheduler.config.get("base_shift", 0.5),
            self.pipe.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, resolved_steps = retrieve_timesteps(
            self.pipe.scheduler,
            num_inference_steps,
            self.device,
            sigmas=sigmas,
            mu=mu,
        )
        self.pipe.scheduler.set_begin_index(0)
        self.img_shapes = [
            [
                (
                    1,
                    self.height // self.pipe.vae_scale_factor // 2,
                    self.width // self.pipe.vae_scale_factor // 2,
                )
            ]
        ]
        first_flow = (timesteps[:1] / 1000).reshape(1).to(
            device=self.device,
            dtype=source_latent.dtype,
        )
        latents = _qwen_noise_scaling(first_flow, noise, source_latent).to(source_latent.dtype)
        self.schedule_debug = {
            "scheduler_class": type(self.pipe.scheduler).__name__,
            "requested_num_inference_steps": int(num_inference_steps),
            "effective_num_inference_steps": int(resolved_steps),
            "mu": float(mu),
        }
        return latents, noise, source_latent, timesteps

    def mask_edit_to_latents(self, mask_edit: Image.Image) -> tuple[Any, Any]:
        import torch

        if self.height is None or self.width is None or self.prompt_embeds is None:
            raise RuntimeError("Source latents must be prepared before the mask.")

        num_channels_latents = self.pipe.transformer.config.in_channels // 4
        latent_height = 2 * (self.height // (self.pipe.vae_scale_factor * 2))
        latent_width = 2 * (self.width // (self.pipe.vae_scale_factor * 2))
        resized_mask = mask_edit.resize((self.width, self.height), Image.Resampling.NEAREST)
        mask_array = (np.asarray(resized_mask.convert("L"), dtype=np.float32) > 127).astype(
            np.float32
        )
        mask = torch.from_numpy(mask_array).unsqueeze(0).unsqueeze(0)
        mask = torch.nn.functional.interpolate(mask, size=(latent_height, latent_width))
        mask = mask.to(device=self.device, dtype=self.prompt_embeds.dtype)
        edit = self.pipe._pack_latents(
            mask.repeat(1, num_channels_latents, 1, 1),
            1,
            num_channels_latents,
            latent_height,
            latent_width,
        )
        keep = 1.0 - edit
        return keep, edit

    def predict_x0(self, latents: Any, flow_t: float) -> Any:
        import torch

        if self.img_shapes is None:
            raise RuntimeError("Qwen Image source latents must be prepared before prediction.")

        timestep = torch.full(
            (latents.shape[0],),
            flow_t,
            device=latents.device,
            dtype=latents.dtype,
        )
        cache_context = getattr(self.pipe.transformer, "cache_context", None)
        context = cache_context("cond") if cache_context is not None else nullcontext()
        with context:
            noise_pred = self.pipe.transformer(
                hidden_states=latents.to(self.dtype),
                timestep=timestep.to(self.dtype),
                guidance=self.guidance,
                encoder_hidden_states_mask=self.prompt_embeds_mask,
                encoder_hidden_states=self.prompt_embeds,
                img_shapes=self.img_shapes,
                attention_kwargs=self.attention_kwargs,
                return_dict=False,
            )[0]

        if self.do_true_cfg:
            context = cache_context("uncond") if cache_context is not None else nullcontext()
            with context:
                neg_noise_pred = self.pipe.transformer(
                    hidden_states=latents.to(self.dtype),
                    timestep=timestep.to(self.dtype),
                    guidance=self.guidance,
                    encoder_hidden_states_mask=self.negative_prompt_embeds_mask,
                    encoder_hidden_states=self.negative_prompt_embeds,
                    img_shapes=self.img_shapes,
                    attention_kwargs=self.attention_kwargs,
                    return_dict=False,
                )[0]
            combined = neg_noise_pred + self.true_cfg_scale * (noise_pred - neg_noise_pred)
            cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
            noise_norm = torch.norm(combined, dim=-1, keepdim=True)
            noise_pred = combined * (cond_norm / noise_norm)

        flow = torch.as_tensor(flow_t, device=latents.device, dtype=noise_pred.dtype)
        return (latents.to(noise_pred.dtype) - flow * noise_pred).to(latents.dtype)

    def decode_latents(self, latents: Any) -> Image.Image:
        import torch

        if self.height is None or self.width is None:
            raise RuntimeError("Qwen Image dimensions must be prepared before decoding.")

        with torch.no_grad():
            unpacked = self.pipe._unpack_latents(
                latents.detach(),
                self.height,
                self.width,
                self.pipe.vae_scale_factor,
            )
            unpacked = unpacked.to(self.pipe.vae.dtype)
            latents_mean = (
                torch.tensor(self.pipe.vae.config.latents_mean)
                .view(1, self.pipe.vae.config.z_dim, 1, 1, 1)
                .to(unpacked.device, unpacked.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.pipe.vae.config.latents_std).view(
                1,
                self.pipe.vae.config.z_dim,
                1,
                1,
                1,
            ).to(unpacked.device, unpacked.dtype)
            decoded_latents = unpacked / latents_std + latents_mean
            image = self.pipe.vae.decode(decoded_latents, return_dict=False)[0][:, :, 0]
        return self.pipe.image_processor.postprocess(image.detach(), output_type="pil")[
            0
        ].convert("RGB")


class QwenImageLanPaintModelWrapper:
    """Minimal model interface consumed by official LanPaint for Qwen Image flow latents."""

    def __init__(self, adapter: QwenImageLanPaintAdapter) -> None:
        self.adapter = adapter
        self.inner_model = self
        self.model_sampling = self

    def noise_scaling(self, sigma: Any, noise: Any, latent_image: Any) -> Any:
        return _qwen_noise_scaling(sigma, noise, latent_image)

    def __call__(self, x: Any, t: Any, **_kwargs: Any) -> tuple[Any, Any]:
        flow_t = float(t.flatten()[0])
        x0 = self.adapter.predict_x0(x, flow_t)
        return x0, x0


class QwenImageInferenceService:
    """Lazy-loading Qwen-Image text-to-image service for experiments."""

    method = QWEN_IMAGE_METHOD
    model_id = QWEN_IMAGE_MODEL_ID
    pipeline_display_name = "Qwen-Image text-to-image"

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
        from diffusers import QwenImagePipeline

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
        pipe = QwenImagePipeline.from_pretrained(resolution.load_target, **kwargs)
        timings["from_pretrained_seconds"] = time.perf_counter() - load_started

        device_started = time.perf_counter()
        if env_flag("ENABLE_QWEN_SEQUENTIAL_CPU_OFFLOAD", default=True) and hasattr(
            pipe,
            "enable_sequential_cpu_offload",
        ):
            pipe.enable_sequential_cpu_offload()
            device_mode = "sequential_cpu_offload"
        elif env_flag("ENABLE_MODEL_CPU_OFFLOAD", default=True) and hasattr(
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
            "generation_mode": "text_to_image",
            "backend_revision": QWEN_IMAGE_BACKEND_REVISION,
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

    def _run_lanpaint(
        self,
        payload: dict[str, Any],
        *,
        reporter: ProgressReporter,
        include_progress_history: bool,
    ) -> dict[str, Any]:
        settings = parse_qwen_image_lanpaint_settings(payload)
        reporter.emit(
            "input_decode_start",
            stage="input",
            message="Decoding request image and mask.",
        )
        image, mask = load_request_images(payload)
        mask_fraction = mask_coverage(mask)
        source_image = make_qwen_lanpaint_source_by_strategy(
            image,
            mask,
            strategy=settings.qwen_source_strategy,
        )
        reporter.emit(
            "input_decode_done",
            stage="input",
            message="Request image, mask, and Qwen Image source decoded.",
            metadata={
                "image_width": image.width,
                "image_height": image.height,
                "mask_width": mask.width,
                "mask_height": mask.height,
                "mask_coverage": mask_fraction,
                "backend_revision": QWEN_IMAGE_LANPAINT_BACKEND_REVISION,
                "qwen_source_strategy": settings.qwen_source_strategy,
            },
        )

        loaded = self.get_loaded(reporter=reporter)
        pipe = loaded.pipe
        torch = loaded.torch
        generator = torch.Generator(device="cpu")
        if settings.seed is not None:
            generator.manual_seed(settings.seed)
        else:
            generator.seed()

        with TemporaryDirectory(prefix=f"{settings.method}_worker_") as temp_dir:
            temp_path = Path(temp_dir)
            memory_before_inference = _cuda_memory_stats(torch, prefix="pre_inference_")
            _reset_cuda_peak_memory(torch)
            reporter.emit(
                "inference_start",
                stage="inference",
                message="Starting Qwen-Image LanPaint inference.",
                progress={"current": 0, "total": settings.num_inference_steps},
                metadata={
                    "method": settings.method,
                    "backend_revision": QWEN_IMAGE_LANPAINT_BACKEND_REVISION,
                    "num_inference_steps": settings.num_inference_steps,
                    "true_cfg_scale": settings.true_cfg_scale,
                    "seed": settings.seed,
                    "lanpaint_inner_steps": settings.lanpaint_inner_steps,
                    "qwen_source_strategy": settings.qwen_source_strategy,
                },
            )
            inference_started = time.perf_counter()
            adapter = QwenImageLanPaintAdapter(pipe)
            adapter.prepare_prompt(
                prompt=settings.prompt,
                negative_prompt=settings.negative_prompt,
                true_cfg_scale=settings.true_cfg_scale,
                max_sequence_length=settings.max_sequence_length,
            )
            _free_qwen_offloaded_modules(pipe, torch)
            latents, noise, source_latent, timesteps = adapter.prepare_source_latents(
                source_image=source_image,
                generator=generator,
                num_inference_steps=settings.num_inference_steps,
            )
            keep_mask, edit_mask = adapter.mask_edit_to_latents(mask)
            _free_qwen_offloaded_modules(pipe, torch)
            model = QwenImageLanPaintModelWrapper(adapter)

            try:
                from LanPaint.lanpaint import LanPaint
            except ImportError as exc:  # pragma: no cover - checked in container smoke.
                raise ImportError(
                    "The pinned LanPaint package is required for qwen_image_lanpaint."
                ) from exc

            lanpaint = LanPaint(
                Model=model,
                NSteps=settings.lanpaint_inner_steps,
                Friction=settings.lanpaint_friction,
                Lambda=settings.lanpaint_lambda,
                Beta=settings.lanpaint_beta,
                StepSize=settings.lanpaint_step_size,
                IS_FLOW=True,
            )
            step_trace: list[dict[str, Any]] = []
            with torch.inference_mode():
                for index, timestep in enumerate(timesteps):
                    flow_value = float((timestep / 1000).item())
                    inner_steps = (
                        0
                        if len(timesteps) - index
                        <= settings.lanpaint_final_outer_steps_without_inner
                        else None
                    )
                    current_times = _make_qwen_current_times(
                        flow_value,
                        adapter.device,
                        latents.dtype,
                    )
                    x0 = lanpaint(
                        x=latents,
                        latent_image=source_latent,
                        noise=noise,
                        sigma=torch.tensor(
                            [flow_value],
                            device=adapter.device,
                            dtype=latents.dtype,
                        ),
                        latent_mask=keep_mask.to(device=adapter.device, dtype=latents.dtype),
                        current_times=current_times,
                        model_options={},
                        seed=settings.seed or 0,
                        n_steps=inner_steps,
                    )
                    noise_pred = (
                        (latents - x0.to(latents.dtype)) / max(flow_value, 1e-6)
                    ).to(latents.dtype)
                    latents_dtype = latents.dtype
                    latents = pipe.scheduler.step(
                        noise_pred,
                        timestep,
                        latents,
                        return_dict=False,
                    )[0].to(latents_dtype)
                    if index < len(timesteps) - 1:
                        next_flow = (timesteps[index + 1] / 1000).reshape(1).to(
                            device=adapter.device,
                            dtype=source_latent.dtype,
                        )
                        keep_reference = model.noise_scaling(
                            next_flow,
                            noise,
                            source_latent,
                        ).to(source_latent.dtype)
                    else:
                        keep_reference = source_latent
                    latents = edit_mask.to(device=adapter.device, dtype=latents.dtype) * latents + (
                        keep_mask.to(device=adapter.device, dtype=latents.dtype) * keep_reference
                    )
                    step_record = {
                        "step_index": index,
                        "timestep": _json_timestep(timestep),
                        "flow_t": flow_value,
                        "lanpaint_inner_steps_override": inner_steps,
                    }
                    step_trace.append(step_record)
                    reporter.emit(
                        "inference_step",
                        stage="inference",
                        message=f"Running Qwen Image LanPaint step {index + 1}/{len(timesteps)}.",
                        progress={"current": index + 1, "total": len(timesteps)},
                        metadata=step_record,
                    )
                _free_qwen_offloaded_modules(pipe, torch)
                raw = adapter.decode_latents(latents)
            inference_seconds = time.perf_counter() - inference_started
            inference_memory = {
                **memory_before_inference,
                **_cuda_memory_stats(torch, prefix="inference_"),
            }
            schedule_debug = {**adapter.schedule_debug, "step_trace": step_trace}

            if raw.size != image.size:
                raw = raw.resize(image.size, Image.Resampling.LANCZOS)
            composite = hard_composite(image, raw, mask)
            changed_outside_mask = outside_mask_changed(image, composite, mask)
            if changed_outside_mask:
                raise RuntimeError("Hard composite changed pixels outside the edit mask.")

            output_path = temp_path / f"composite.{settings.output_format}"
            composite.save(output_path)
            encoded = image_to_base64(composite, output_format=settings.output_format)

        output = {
            "image_base64": encoded,
            "output_format": settings.output_format,
            "width": composite.width,
            "height": composite.height,
            "mask_convention": "white = inpaint/edit, black = preserve",
            "timings": {**loaded.timings, "inference_seconds": inference_seconds},
            "gpu_memory": inference_memory,
            "model": {
                **loaded.model,
                "method": settings.method,
                "generation_mode": "text_to_image_lanpaint_inpainting",
                "backend_revision": QWEN_IMAGE_LANPAINT_BACKEND_REVISION,
                "denoising_backend": "LanPaint",
                "source_image_route": settings.qwen_source_strategy,
            },
            "inference_settings": {
                "method": settings.method,
                "backend_revision": QWEN_IMAGE_LANPAINT_BACKEND_REVISION,
                "prompt": settings.prompt,
                "negative_prompt": settings.negative_prompt,
                "true_cfg_scale": settings.true_cfg_scale,
                "guidance_scale_passed_as": "true_cfg_scale",
                "num_inference_steps": settings.num_inference_steps,
                "max_sequence_length": settings.max_sequence_length,
                "seed": settings.seed,
                "mask_coverage": mask_fraction,
                "qwen_source_strategy": settings.qwen_source_strategy,
                "lanpaint_inner_steps": settings.lanpaint_inner_steps,
                "lanpaint_friction": settings.lanpaint_friction,
                "lanpaint_lambda": settings.lanpaint_lambda,
                "lanpaint_beta": settings.lanpaint_beta,
                "lanpaint_step_size": settings.lanpaint_step_size,
                "lanpaint_final_outer_steps_without_inner": (
                    settings.lanpaint_final_outer_steps_without_inner
                ),
            },
            "schedule_debug": schedule_debug,
            "outside_mask_changed_after_hard_composite": changed_outside_mask,
        }
        if include_progress_history:
            output["run_report"] = {"progress_events": reporter.history}
        return output

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
        try:
            method = normalize_method(payload.get("method"))
        except ValueError as exc:
            raise WorkerInputError(str(exc)) from exc
        if method == QWEN_IMAGE_LANPAINT_METHOD:
            return self._run_lanpaint(
                payload,
                reporter=reporter,
                include_progress_history=include_progress_history,
            )
        settings = parse_qwen_image_settings(payload)
        reporter.emit(
            "input_decode_done",
            stage="input",
            message="Qwen-Image text prompt validated.",
            metadata={
                "width": settings.width,
                "height": settings.height,
                "backend_revision": QWEN_IMAGE_BACKEND_REVISION,
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
            "true_cfg_scale": settings.true_cfg_scale,
            "height": settings.height,
            "width": settings.width,
            "num_inference_steps": settings.num_inference_steps,
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
                    "backend_revision": QWEN_IMAGE_BACKEND_REVISION,
                    "num_inference_steps": settings.num_inference_steps,
                    "true_cfg_scale": settings.true_cfg_scale,
                    "seed": settings.seed,
                    "width": settings.width,
                    "height": settings.height,
                },
            )
            inference_started = time.perf_counter()
            with torch.inference_mode():
                image = pipe(**call_kwargs).images[0].convert("RGB")
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

            output_path = temp_path / f"generated.{settings.output_format}"
            reporter.emit(
                "output_encode_start",
                stage="output",
                message="Encoding generated image.",
                metadata={"output_format": settings.output_format},
            )
            image.save(output_path)
            encoded = image_to_base64(image, output_format=settings.output_format)
            reporter.emit(
                "output_encode_done",
                stage="output",
                message="Generated image encoded.",
                metadata={
                    "output_format": settings.output_format,
                    "width": image.width,
                    "height": image.height,
                },
            )

        output = {
            "image_base64": encoded,
            "output_format": settings.output_format,
            "width": image.width,
            "height": image.height,
            "timings": {
                **loaded.timings,
                "inference_seconds": inference_seconds,
            },
            "gpu_memory": inference_memory,
            "model": loaded.model,
            "inference_settings": {
                "method": settings.method,
                "backend_revision": QWEN_IMAGE_BACKEND_REVISION,
                "prompt": settings.prompt,
                "negative_prompt": settings.negative_prompt,
                "true_cfg_scale": settings.true_cfg_scale,
                "guidance_scale_passed_as": "true_cfg_scale",
                "num_inference_steps": settings.num_inference_steps,
                "max_sequence_length": settings.max_sequence_length,
                "seed": settings.seed,
                "width": settings.width,
                "height": settings.height,
            },
        }
        if include_progress_history:
            output["run_report"] = {"progress_events": reporter.history}
        return output


__all__ = [
    "QWEN_IMAGE_BACKEND_REVISION",
    "QWEN_IMAGE_MODEL_ID",
    "QwenImageInferenceService",
    "parse_qwen_image_settings",
]
