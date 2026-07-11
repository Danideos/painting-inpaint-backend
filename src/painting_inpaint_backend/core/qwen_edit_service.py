"""Qwen-Image-Edit masked inpainting through LanPaint."""

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

from .image_helpers import binarize_mask, hard_composite, mask_coverage, outside_mask_changed
from .image_io import image_to_base64, normalize_output_format
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
from .methods import QWEN_EDIT_METHOD, RestorationMethod, normalize_method
from .model_loading import env_flag, resolve_model_load_target
from .progress import ProgressReporter
from .sd_inpaint_base import LoadedSDPipeline

LOGGER = logging.getLogger(__name__)

QWEN_EDIT_MODEL_ID = "Qwen/Qwen-Image-Edit"
QWEN_MASKED_REGION_FILL_RGB = (255, 255, 255)
QWEN_LANPAINT_BACKEND_REVISION = "qwen-image-edit-lanpaint-v1"
QWEN_NATIVE_EXPERIMENT_REVISION = "qwen-image-edit-native-experiment-v1"
QWEN_LANPAINT_SOURCE_FILL_RADIUS = 3.0
QWEN_LANPAINT_INNER_STEPS = 10
QWEN_LANPAINT_FRICTION = 15.0
QWEN_LANPAINT_LAMBDA = 10.0
QWEN_LANPAINT_BETA = 1.0
QWEN_LANPAINT_STEP_SIZE = 0.2
QWEN_LANPAINT_FINAL_OUTER_STEPS_WITHOUT_INNER = 3


def make_qwen_source_image(image: Image.Image, mask: Image.Image) -> Image.Image:
    """Hide masked pixels before Qwen's image-conditioning path sees the source."""

    source = image.convert("RGB").copy()
    source.paste(QWEN_MASKED_REGION_FILL_RGB, mask=binarize_mask(mask))
    return source


def make_qwen_lanpaint_source_image(image: Image.Image, mask: Image.Image) -> Image.Image:
    """Hide masked truth with content-aware fill before Qwen image conditioning."""

    from .canny_lanpaint import make_lanpaint_source_image

    return make_lanpaint_source_image(
        image,
        mask,
        fill_radius=QWEN_LANPAINT_SOURCE_FILL_RADIUS,
    )


def _make_qwen_current_times(flow_t: float, device: Any, dtype: Any) -> tuple[Any, Any, Any]:
    import torch

    flow = torch.tensor([flow_t], device=device, dtype=dtype)
    alpha_bar = (1.0 - flow) ** 2 / ((1.0 - flow) ** 2 + flow**2)
    ve_sigma = flow / torch.clamp(1.0 - flow, min=1e-6)
    return ve_sigma, alpha_bar, flow


def _qwen_noise_scaling(sigma: Any, noise: Any, latent_image: Any) -> Any:
    while len(sigma.shape) < len(noise.shape):
        sigma = sigma.unsqueeze(-1)
    return (1.0 - sigma) * latent_image + sigma * noise


def _free_qwen_offloaded_modules(pipe: Any, torch: Any) -> None:
    """Force offloaded Qwen modules off GPU between custom pipeline phases."""

    maybe_free = getattr(pipe, "maybe_free_model_hooks", None)
    if maybe_free is not None:
        maybe_free()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def parse_qwen_native_pipeline_flag(payload: dict[str, Any]) -> bool:
    """Validate the hidden native-pipeline experiment switch."""

    value = payload.get("qwen_native_pipeline", False)
    if not isinstance(value, bool):
        raise WorkerInputError("qwen_native_pipeline must be a boolean.")
    return value


@dataclass(frozen=True)
class QwenEditSettings:
    """Validated scalar settings for Qwen LanPaint masked inpainting."""

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
    lanpaint_inner_steps: int
    lanpaint_friction: float
    lanpaint_lambda: float
    lanpaint_beta: float
    lanpaint_step_size: float
    lanpaint_final_outer_steps_without_inner: int


def parse_qwen_edit_settings(payload: dict[str, Any]) -> QwenEditSettings:
    """Validate request scalar parameters for Qwen LanPaint masked inpainting."""

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
        raise WorkerInputError("padding_mask_crop is not supported by Qwen LanPaint inpainting.")

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

    lanpaint_friction = _optional_float(
        payload,
        "lanpaint_friction",
        QWEN_LANPAINT_FRICTION,
    )
    if lanpaint_friction < 0:
        raise WorkerInputError("lanpaint_friction must be non-negative.")

    lanpaint_lambda = _optional_float(payload, "lanpaint_lambda", QWEN_LANPAINT_LAMBDA)
    if lanpaint_lambda < 0:
        raise WorkerInputError("lanpaint_lambda must be non-negative.")

    lanpaint_beta = _optional_float(payload, "lanpaint_beta", QWEN_LANPAINT_BETA)
    if lanpaint_beta < 0:
        raise WorkerInputError("lanpaint_beta must be non-negative.")

    lanpaint_step_size = _optional_float(
        payload,
        "lanpaint_step_size",
        QWEN_LANPAINT_STEP_SIZE,
    )
    if lanpaint_step_size <= 0:
        raise WorkerInputError("lanpaint_step_size must be greater than 0.")

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
        lanpaint_inner_steps=lanpaint_inner_steps,
        lanpaint_friction=lanpaint_friction,
        lanpaint_lambda=lanpaint_lambda,
        lanpaint_beta=lanpaint_beta,
        lanpaint_step_size=lanpaint_step_size,
        lanpaint_final_outer_steps_without_inner=lanpaint_final_without_inner,
    )


class QwenLanPaintAdapter:
    """Expose Qwen-Image-Edit flow components to official LanPaint."""

    def __init__(self, pipe: Any) -> None:
        self.pipe = pipe
        self.height: int | None = None
        self.width: int | None = None
        self.calculated_height: int | None = None
        self.calculated_width: int | None = None
        self.prompt_embeds = None
        self.prompt_embeds_mask = None
        self.negative_prompt_embeds = None
        self.negative_prompt_embeds_mask = None
        self.image_latents = None
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

    def calculate_dimensions(self, image: Image.Image) -> tuple[int, int]:
        from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_inpaint import (
            calculate_dimensions,
        )

        width, height, _ = calculate_dimensions(1024 * 1024, image.width / image.height)
        multiple_of = self.pipe.vae_scale_factor * 2
        width = int(width) // multiple_of * multiple_of
        height = int(height) // multiple_of * multiple_of
        return width, height

    def prepare_prompt(
        self,
        *,
        prompt_image: Image.Image,
        prompt: str,
        negative_prompt: str,
        true_cfg_scale: float,
        guidance_scale: float | None,
        max_sequence_length: int,
    ) -> None:
        self.do_true_cfg = true_cfg_scale > 1 and negative_prompt is not None
        self.true_cfg_scale = true_cfg_scale
        self.prompt_embeds, self.prompt_embeds_mask = self.pipe.encode_prompt(
            image=prompt_image,
            prompt=prompt,
            device=self.device,
            num_images_per_prompt=1,
            max_sequence_length=max_sequence_length,
        )
        if self.do_true_cfg:
            self.negative_prompt_embeds, self.negative_prompt_embeds_mask = (
                self.pipe.encode_prompt(
                    image=prompt_image,
                    prompt=negative_prompt,
                    device=self.device,
                    num_images_per_prompt=1,
                    max_sequence_length=max_sequence_length,
                )
            )

        if self.pipe.transformer.config.guidance_embeds and guidance_scale is None:
            raise WorkerInputError("guidance_scale is required for guidance-distilled Qwen models.")
        if self.pipe.transformer.config.guidance_embeds:
            import torch

            self.guidance = torch.full(
                [1],
                guidance_scale,
                device=self.device,
                dtype=torch.float32,
            )

    def prepare_source_latents(
        self,
        *,
        source_image: Image.Image,
        generator: Any,
        num_inference_steps: int,
        strength: float,
    ) -> tuple[Any, Any, Any, Any]:
        from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_inpaint import (
            calculate_shift,
            retrieve_timesteps,
        )

        self.width, self.height = self.calculate_dimensions(source_image)
        self.calculated_width = self.width
        self.calculated_height = self.height
        resized = self.pipe.image_processor.resize(source_image, self.height, self.width)
        image_tensor = self.pipe.image_processor.preprocess(
            resized,
            height=self.height,
            width=self.width,
        ).to(dtype=self.prompt_embeds.dtype)

        sigmas = np.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps)
        image_seq_len = (
            (int(self.height) // self.pipe.vae_scale_factor // 2)
            * (int(self.width) // self.pipe.vae_scale_factor // 2)
        )
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
        timesteps, effective_steps = self.pipe.get_timesteps(resolved_steps, strength, self.device)
        if effective_steps < 1:
            raise WorkerInputError(
                "Qwen LanPaint strength leaves zero denoising steps; increase strength."
            )

        num_channels_latents = self.pipe.transformer.config.in_channels // 4
        latents, noise, image_latents = self.pipe.prepare_latents(
            image_tensor,
            timesteps[:1].repeat(1),
            1,
            num_channels_latents,
            self.height,
            self.width,
            self.prompt_embeds.dtype,
            self.device,
            generator,
            None,
        )
        self.image_latents = image_latents
        self.img_shapes = [
            [
                (
                    1,
                    self.height // self.pipe.vae_scale_factor // 2,
                    self.width // self.pipe.vae_scale_factor // 2,
                ),
                (
                    1,
                    self.calculated_height // self.pipe.vae_scale_factor // 2,
                    self.calculated_width // self.pipe.vae_scale_factor // 2,
                ),
            ]
        ]
        self.schedule_debug = {
            "scheduler_class": type(self.pipe.scheduler).__name__,
            "requested_num_inference_steps": int(num_inference_steps),
            "effective_num_inference_steps": int(effective_steps),
            "strength": float(strength),
            "mu": float(mu),
        }
        return latents, noise, image_latents, timesteps

    def mask_edit_to_latents(self, mask_edit: Image.Image) -> tuple[Any, Any]:
        import torch

        if self.height is None or self.width is None:
            raise RuntimeError("Source latents must be prepared before the mask.")

        num_channels_latents = self.pipe.transformer.config.in_channels // 4
        latent_height = 2 * (int(self.height) // (self.pipe.vae_scale_factor * 2))
        latent_width = 2 * (int(self.width) // (self.pipe.vae_scale_factor * 2))
        mask = self.pipe.mask_processor.preprocess(
            mask_edit,
            height=self.height,
            width=self.width,
        )
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
        if self.image_latents is not None and keep.shape != self.image_latents.shape:
            raise RuntimeError("Packed Qwen mask and source latent shapes do not match.")
        return keep, edit

    def predict_x0(self, latents: Any, flow_t: float) -> Any:
        import torch

        if self.image_latents is None or self.img_shapes is None:
            raise RuntimeError("Qwen source latents must be prepared before prediction.")

        latent_model_input = torch.cat(
            [latents, self.image_latents.to(device=latents.device, dtype=latents.dtype)],
            dim=1,
        )
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
                hidden_states=latent_model_input.to(self.dtype),
                timestep=timestep.to(self.dtype),
                guidance=self.guidance,
                encoder_hidden_states_mask=self.prompt_embeds_mask,
                encoder_hidden_states=self.prompt_embeds,
                img_shapes=self.img_shapes,
                attention_kwargs=self.attention_kwargs,
                return_dict=False,
            )[0]
            noise_pred = noise_pred[:, : latents.size(1)]

        if self.do_true_cfg:
            context = cache_context("uncond") if cache_context is not None else nullcontext()
            with context:
                neg_noise_pred = self.pipe.transformer(
                    hidden_states=latent_model_input.to(self.dtype),
                    timestep=timestep.to(self.dtype),
                    guidance=self.guidance,
                    encoder_hidden_states_mask=self.negative_prompt_embeds_mask,
                    encoder_hidden_states=self.negative_prompt_embeds,
                    img_shapes=self.img_shapes,
                    attention_kwargs=self.attention_kwargs,
                    return_dict=False,
                )[0]
                neg_noise_pred = neg_noise_pred[:, : latents.size(1)]
            combined = neg_noise_pred + self.true_cfg_scale * (noise_pred - neg_noise_pred)
            cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
            noise_norm = torch.norm(combined, dim=-1, keepdim=True)
            noise_pred = combined * (cond_norm / noise_norm)

        flow = torch.as_tensor(flow_t, device=latents.device, dtype=noise_pred.dtype)
        return (latents.to(noise_pred.dtype) - flow * noise_pred).to(latents.dtype)

    def decode_latents(self, latents: Any) -> Image.Image:
        import torch

        if self.height is None or self.width is None:
            raise RuntimeError("Qwen dimensions must be prepared before decoding.")

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


class QwenLanPaintModelWrapper:
    """Minimal model interface consumed by official LanPaint for Qwen flow latents."""

    def __init__(self, adapter: QwenLanPaintAdapter) -> None:
        self.adapter = adapter
        self.inner_model = self
        self.model_sampling = self

    def noise_scaling(self, sigma: Any, noise: Any, latent_image: Any) -> Any:
        return _qwen_noise_scaling(sigma, noise, latent_image)

    def __call__(self, x: Any, t: Any, **_kwargs: Any) -> tuple[Any, Any]:
        flow_t = float(t.flatten()[0])
        x0 = self.adapter.predict_x0(x, flow_t)
        return x0, x0


class QwenEditInferenceService:
    """Lazy-loading Qwen-Image-Edit service using LanPaint for masked inpainting."""

    method = QWEN_EDIT_METHOD
    model_id = QWEN_EDIT_MODEL_ID
    pipeline_display_name = "Qwen-Image-Edit LanPaint inpainting"

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
            "mask_route": "lanpaint_keep_mask",
            "source_image_route": "telea_filled_before_qwen_image_conditioning",
            "denoising_backend": "LanPaint",
            "backend_revision": QWEN_LANPAINT_BACKEND_REVISION,
            "native_pipeline_call": False,
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

    def _run_native_pipeline(
        self,
        *,
        image: Image.Image,
        mask: Image.Image,
        mask_fraction: float,
        settings: QwenEditSettings,
        reporter: ProgressReporter,
        include_progress_history: bool,
    ) -> dict[str, Any]:
        """Run an explicit native-pipeline experiment without changing LanPaint defaults."""

        loaded = self.get_loaded(reporter=reporter)
        pipe = loaded.pipe
        torch = loaded.torch
        generator = torch.Generator(device="cpu")
        if settings.seed is not None:
            generator.manual_seed(settings.seed)
        else:
            generator.seed()

        call_kwargs: dict[str, Any] = {
            "prompt": settings.prompt,
            "negative_prompt": settings.negative_prompt,
            "image": image,
            "mask_image": mask,
            "height": image.height,
            "width": image.width,
            "strength": settings.strength,
            "num_inference_steps": settings.num_inference_steps,
            "true_cfg_scale": settings.guidance_scale,
            "max_sequence_length": settings.max_sequence_length,
            "generator": generator,
        }
        _add_inference_progress_callback(
            call_kwargs=call_kwargs,
            pipe=pipe,
            settings=settings,  # type: ignore[arg-type]
            reporter=reporter,
        )

        with TemporaryDirectory(prefix="qwen_native_experiment_") as temp_dir:
            temp_path = Path(temp_dir)
            memory_before_inference = _cuda_memory_stats(torch, prefix="pre_inference_")
            _reset_cuda_peak_memory(torch)
            reporter.emit(
                "inference_start",
                stage="inference",
                message="Starting native Qwen-Image-Edit inpainting inference.",
                progress={"current": 0, "total": settings.num_inference_steps},
                metadata={
                    "method": settings.method,
                    "backend_revision": QWEN_NATIVE_EXPERIMENT_REVISION,
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
                message="Native Qwen-Image-Edit inpainting inference completed.",
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
            composite = hard_composite(image, raw, mask)
            changed_outside_mask = outside_mask_changed(image, composite, mask)
            if changed_outside_mask:
                raise RuntimeError("Hard composite changed pixels outside the edit mask.")

            output_path = temp_path / f"composite.{settings.output_format}"
            composite.save(output_path)
            encoded = image_to_base64(composite, output_format=settings.output_format)

        native_model = {
            **loaded.model,
            "mask_route": "native_diffusers_mask_image",
            "source_image_route": "original_image_passed_to_native_pipeline",
            "denoising_backend": "QwenImageEditInpaintPipeline",
            "backend_revision": QWEN_NATIVE_EXPERIMENT_REVISION,
            "native_pipeline_call": True,
        }
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
            "model": native_model,
            "inference_settings": {
                "method": settings.method,
                "backend_revision": QWEN_NATIVE_EXPERIMENT_REVISION,
                "prompt": settings.prompt,
                "negative_prompt": settings.negative_prompt,
                "strength": settings.strength,
                "guidance_scale": settings.guidance_scale,
                "guidance_scale_passed_as": "true_cfg_scale",
                "num_inference_steps": settings.num_inference_steps,
                "max_sequence_length": settings.max_sequence_length,
                "seed": settings.seed,
                "mask_coverage": mask_fraction,
                "native_pipeline_call": True,
            },
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
        settings = parse_qwen_edit_settings(payload)
        native_pipeline = parse_qwen_native_pipeline_flag(payload)
        reporter.emit(
            "input_decode_start",
            stage="input",
            message="Decoding request image and mask.",
        )
        image, mask = load_request_images(payload)
        mask_fraction = mask_coverage(mask)
        if native_pipeline:
            reporter.emit(
                "input_decode_done",
                stage="input",
                message="Request image and mask decoded for native Qwen experiment.",
                metadata={
                    "image_width": image.width,
                    "image_height": image.height,
                    "mask_width": mask.width,
                    "mask_height": mask.height,
                    "mask_coverage": mask_fraction,
                    "backend_revision": QWEN_NATIVE_EXPERIMENT_REVISION,
                    "native_pipeline_call": True,
                },
            )
            return self._run_native_pipeline(
                image=image,
                mask=mask,
                mask_fraction=mask_fraction,
                settings=settings,
                reporter=reporter,
                include_progress_history=include_progress_history,
            )

        qwen_source_image = make_qwen_lanpaint_source_image(image, mask)
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
                "backend_revision": QWEN_LANPAINT_BACKEND_REVISION,
                "qwen_source_fill_radius": QWEN_LANPAINT_SOURCE_FILL_RADIUS,
            },
        )

        loaded = self.get_loaded(reporter=reporter)
        pipe = loaded.pipe
        torch = loaded.torch

        generator = torch.Generator(device=pipe._execution_device)
        if settings.seed is not None:
            generator.manual_seed(settings.seed)
        else:
            generator.seed()

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
                    "backend_revision": QWEN_LANPAINT_BACKEND_REVISION,
                    "num_inference_steps": settings.num_inference_steps,
                    "strength": settings.strength,
                    "guidance_scale": settings.guidance_scale,
                    "seed": settings.seed,
                    "lanpaint_inner_steps": settings.lanpaint_inner_steps,
                    "lanpaint_friction": settings.lanpaint_friction,
                    "lanpaint_lambda": settings.lanpaint_lambda,
                    "lanpaint_beta": settings.lanpaint_beta,
                    "lanpaint_step_size": settings.lanpaint_step_size,
                    "lanpaint_final_outer_steps_without_inner": (
                        settings.lanpaint_final_outer_steps_without_inner
                    ),
                },
            )
            inference_started = time.perf_counter()
            adapter = QwenLanPaintAdapter(pipe)
            source_width, source_height = adapter.calculate_dimensions(qwen_source_image)
            prompt_source = pipe.image_processor.resize(
                qwen_source_image,
                source_height,
                source_width,
            )
            adapter.prepare_prompt(
                prompt_image=prompt_source,
                prompt=settings.prompt,
                negative_prompt=settings.negative_prompt,
                true_cfg_scale=settings.guidance_scale,
                guidance_scale=None,
                max_sequence_length=settings.max_sequence_length,
            )
            _free_qwen_offloaded_modules(pipe, torch)
            latents, noise, source_latent, timesteps = adapter.prepare_source_latents(
                source_image=qwen_source_image,
                generator=generator,
                num_inference_steps=settings.num_inference_steps,
                strength=settings.strength,
            )
            keep_mask, edit_mask = adapter.mask_edit_to_latents(mask)
            _free_qwen_offloaded_modules(pipe, torch)
            model = QwenLanPaintModelWrapper(adapter)

            try:
                from LanPaint.lanpaint import LanPaint
            except ImportError as exc:  # pragma: no cover - checked in container smoke.
                raise ImportError("The pinned LanPaint package is required for qwen_edit.") from exc

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
                        message=f"Running Qwen LanPaint step {index + 1}/{len(timesteps)}.",
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
            schedule_debug = {
                **adapter.schedule_debug,
                "step_trace": step_trace,
            }
            reporter.emit(
                "inference_done",
                stage="inference",
                message=f"{self.pipeline_display_name} inference completed.",
                progress={
                    "current": len(timesteps),
                    "total": len(timesteps),
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
                "backend_revision": QWEN_LANPAINT_BACKEND_REVISION,
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
                "qwen_source_strategy": "telea_filled_before_qwen_image_conditioning",
                "qwen_source_fill_radius": QWEN_LANPAINT_SOURCE_FILL_RADIUS,
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
            "latent_init_debug": {
                "source_latents": list(source_latent.shape),
                "mask_keep_latents": list(keep_mask.shape),
                "mask_edit_latents": list(edit_mask.shape),
            },
            "outside_mask_changed_after_hard_composite": changed_outside_mask,
        }
        if include_progress_history:
            output["run_report"] = {"progress_events": reporter.history}
        return output


__all__ = [
    "QWEN_EDIT_MODEL_ID",
    "QWEN_LANPAINT_BACKEND_REVISION",
    "QwenEditInferenceService",
    "make_qwen_lanpaint_source_image",
    "parse_qwen_edit_settings",
    "parse_qwen_native_pipeline_flag",
]
