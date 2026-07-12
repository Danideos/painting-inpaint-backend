"""SD3-medium masked inpainting through LanPaint."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
from PIL import Image

from .canny_lanpaint import (
    LANPAINT_BETA,
    LANPAINT_FINAL_OUTER_STEPS_WITHOUT_INNER,
    LANPAINT_FRICTION,
    LANPAINT_INNER_STEPS,
    LANPAINT_LAMBDA,
    LANPAINT_SOURCE_FILL_RADIUS,
    LANPAINT_STEP_SIZE,
    make_current_times,
    make_lanpaint_source_image,
    reinject_keep_latents,
)
from .image_helpers import binarize_mask, hard_composite, mask_coverage, outside_mask_changed
from .image_io import image_to_base64, normalize_output_format
from .inference import (
    WorkerInputError,
    _cuda_memory_stats,
    _json_timestep,
    _optional_float,
    _optional_int,
    _payload_flag,
    _reset_cuda_peak_memory,
    load_request_images,
)
from .methods import SD3_LANPAINT_METHOD, RestorationMethod, normalize_method
from .model_loading import env_flag, resolve_model_load_target
from .partial_noise import normalize_partial_noise, partial_noise_start_index
from .progress import ProgressReporter
from .sd35_service import SD35_BASE_MODEL_ID
from .sd_inpaint_base import LoadedSDPipeline

LOGGER = logging.getLogger(__name__)

SD3_LANPAINT_BACKEND_REVISION = "sd3-medium-lanpaint-v1"
SD3_SOURCE_STRATEGY_TELEA = "telea"
SD3_SOURCE_STRATEGY_ORIGINAL = "original"


@dataclass(frozen=True)
class SD3LanPaintSettings:
    """Validated scalar settings for SD3-medium LanPaint masked inpainting."""

    method: RestorationMethod
    prompt: str
    negative_prompt: str
    guidance_scale: float
    num_inference_steps: int
    partial_noise: float
    seed: int | None
    output_format: str
    max_sequence_length: int
    lanpaint_inner_steps: int
    lanpaint_friction: float
    lanpaint_lambda: float
    lanpaint_beta: float
    lanpaint_step_size: float
    lanpaint_final_outer_steps_without_inner: int
    sd3_source_strategy: str


def parse_sd3_source_strategy(payload: dict[str, Any]) -> str:
    value = str(
        payload.get(
            "sd3_source_strategy",
            payload.get("source_strategy", SD3_SOURCE_STRATEGY_TELEA),
        )
    ).strip().lower()
    aliases = {
        "telea": SD3_SOURCE_STRATEGY_TELEA,
        "telea_fill": SD3_SOURCE_STRATEGY_TELEA,
        "telea_filled": SD3_SOURCE_STRATEGY_TELEA,
        "original": SD3_SOURCE_STRATEGY_ORIGINAL,
        "raw": SD3_SOURCE_STRATEGY_ORIGINAL,
        "raw_input": SD3_SOURCE_STRATEGY_ORIGINAL,
    }
    try:
        return aliases[value]
    except KeyError as exc:
        raise WorkerInputError(
            "sd3_source_strategy must be one of: telea, original."
        ) from exc


def make_sd3_lanpaint_source_by_strategy(
    image: Image.Image,
    mask: Image.Image,
    *,
    strategy: str,
) -> Image.Image:
    if strategy == SD3_SOURCE_STRATEGY_ORIGINAL:
        return image.convert("RGB").copy()
    if strategy == SD3_SOURCE_STRATEGY_TELEA:
        return make_lanpaint_source_image(
            image,
            mask,
            fill_radius=LANPAINT_SOURCE_FILL_RADIUS,
        )
    raise WorkerInputError("sd3_source_strategy must be one of: telea, original.")


def parse_sd3_lanpaint_settings(payload: dict[str, Any]) -> SD3LanPaintSettings:
    """Validate request scalar parameters for SD3-medium LanPaint inpainting."""

    try:
        method = normalize_method(payload.get("method"))
    except ValueError as exc:
        raise WorkerInputError(str(exc)) from exc
    if method != SD3_LANPAINT_METHOD:
        raise WorkerInputError(f"The service only accepts method={SD3_LANPAINT_METHOD!r}.")

    prompt = payload.get("prompt", "")
    if not isinstance(prompt, str):
        raise WorkerInputError("prompt must be a string.")

    guidance_scale = _optional_float(payload, "guidance_scale", 5.0)
    if guidance_scale < 0:
        raise WorkerInputError("guidance_scale must be non-negative.")

    num_inference_steps = _optional_int(payload, "num_inference_steps", 50)
    if num_inference_steps <= 0:
        raise WorkerInputError("num_inference_steps must be positive.")

    try:
        partial_noise = normalize_partial_noise(payload.get("partial_noise"))
    except ValueError as exc:
        raise WorkerInputError(str(exc)) from exc

    max_sequence_length = _optional_int(payload, "max_sequence_length", 256)
    if max_sequence_length <= 0:
        raise WorkerInputError("max_sequence_length must be positive.")

    lanpaint_inner_steps = _optional_int(
        payload,
        "lanpaint_inner_steps",
        LANPAINT_INNER_STEPS,
    )
    if lanpaint_inner_steps < 0:
        raise WorkerInputError("lanpaint_inner_steps must be non-negative.")

    lanpaint_final_without_inner = _optional_int(
        payload,
        "lanpaint_final_outer_steps_without_inner",
        LANPAINT_FINAL_OUTER_STEPS_WITHOUT_INNER,
    )
    if lanpaint_final_without_inner < 0:
        raise WorkerInputError(
            "lanpaint_final_outer_steps_without_inner must be non-negative."
        )

    lanpaint_friction = _optional_float(payload, "lanpaint_friction", LANPAINT_FRICTION)
    if lanpaint_friction < 0:
        raise WorkerInputError("lanpaint_friction must be non-negative.")

    lanpaint_lambda = _optional_float(payload, "lanpaint_lambda", LANPAINT_LAMBDA)
    if lanpaint_lambda < 0:
        raise WorkerInputError("lanpaint_lambda must be non-negative.")

    lanpaint_beta = _optional_float(payload, "lanpaint_beta", LANPAINT_BETA)
    if lanpaint_beta < 0:
        raise WorkerInputError("lanpaint_beta must be non-negative.")

    lanpaint_step_size = _optional_float(payload, "lanpaint_step_size", LANPAINT_STEP_SIZE)
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

    return SD3LanPaintSettings(
        method=method,
        prompt=prompt,
        negative_prompt=str(payload.get("negative_prompt", "")),
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        partial_noise=partial_noise,
        seed=seed,
        output_format=output_format,
        max_sequence_length=max_sequence_length,
        lanpaint_inner_steps=lanpaint_inner_steps,
        lanpaint_friction=lanpaint_friction,
        lanpaint_lambda=lanpaint_lambda,
        lanpaint_beta=lanpaint_beta,
        lanpaint_step_size=lanpaint_step_size,
        lanpaint_final_outer_steps_without_inner=lanpaint_final_without_inner,
        sd3_source_strategy=parse_sd3_source_strategy(payload),
    )


class SD3LanPaintAdapter:
    """Expose SD3-medium flow components to official LanPaint."""

    def __init__(self, pipe: Any) -> None:
        self.pipe = pipe
        self.prompt_embeds = None
        self.negative_prompt_embeds = None
        self.pooled_prompt_embeds = None
        self.negative_pooled_prompt_embeds = None
        self.source_latent = None
        self.height: int | None = None
        self.width: int | None = None
        self.guidance_scale = 1.0
        self.do_classifier_free_guidance = False
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
        guidance_scale: float,
        max_sequence_length: int,
    ) -> None:
        self.guidance_scale = float(guidance_scale)
        self.do_classifier_free_guidance = self.guidance_scale > 1.0
        self.pipe._guidance_scale = self.guidance_scale
        self.pipe._clip_skip = None
        self.pipe._joint_attention_kwargs = {}
        (
            self.prompt_embeds,
            self.negative_prompt_embeds,
            self.pooled_prompt_embeds,
            self.negative_pooled_prompt_embeds,
        ) = self.pipe.encode_prompt(
            prompt=prompt,
            prompt_2=None,
            prompt_3=None,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            negative_prompt_2=None,
            negative_prompt_3=None,
            max_sequence_length=max_sequence_length,
        )

    def _validate_dimensions(self) -> None:
        if self.height is None or self.width is None:
            raise RuntimeError("SD3 dimensions must be set before latent encoding.")
        divisor = int(self.pipe.vae_scale_factor * self.pipe.patch_size)
        if self.height % divisor != 0 or self.width % divisor != 0:
            raise WorkerInputError(
                f"image width and height must be divisible by {divisor} for SD3 LanPaint."
            )

    def encode_source_latents(self, image: Image.Image, generator: Any) -> Any:
        import torch

        if self.height is None or self.width is None:
            self.width, self.height = image.size
        self._validate_dimensions()
        image_tensor = self.pipe.image_processor.preprocess(
            image.convert("RGB"),
            height=self.height,
            width=self.width,
        ).to(device=self.device, dtype=self.pipe.vae.dtype)
        with torch.inference_mode():
            encoded = self.pipe.vae.encode(image_tensor).latent_dist.sample(generator=generator)
        encoded = (encoded - self.pipe.vae.config.shift_factor) * (
            self.pipe.vae.config.scaling_factor
        )
        self.source_latent = encoded.to(self.dtype)
        return self.source_latent

    def mask_edit_to_latents(self, mask_edit: Image.Image) -> tuple[Any, Any]:
        import torch

        if self.source_latent is None:
            raise RuntimeError("Source latents must be encoded before the mask.")
        latent_height, latent_width = self.source_latent.shape[-2:]
        edit_array = np.asarray(binarize_mask(mask_edit), dtype=np.float32) / 255.0
        edit = torch.from_numpy(edit_array)[None, None].to(self.device, dtype=torch.float32)
        edit = torch.nn.functional.interpolate(
            edit,
            size=(latent_height, latent_width),
            mode="nearest",
        )
        edit = (edit > 0.5).to(dtype=self.source_latent.dtype)
        keep = 1.0 - edit
        return keep.contiguous(), edit.contiguous()

    def prepare_timesteps(self, num_steps: int) -> tuple[Any, Any]:
        from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import (
            calculate_shift,
            retrieve_timesteps,
        )

        if self.source_latent is None:
            raise RuntimeError("Source latents must be encoded before timesteps.")
        latent_height, latent_width = self.source_latent.shape[-2:]
        patch_size = int(self.pipe.transformer.config.patch_size)
        image_seq_len = (latent_height // patch_size) * (latent_width // patch_size)
        scheduler_kwargs: dict[str, Any] = {}
        if self.pipe.scheduler.config.get("use_dynamic_shifting", None):
            mu = calculate_shift(
                image_seq_len,
                self.pipe.scheduler.config.get("base_image_seq_len", 256),
                self.pipe.scheduler.config.get("max_image_seq_len", 4096),
                self.pipe.scheduler.config.get("base_shift", 0.5),
                self.pipe.scheduler.config.get("max_shift", 1.16),
            )
            scheduler_kwargs["mu"] = mu
        else:
            mu = None
        timesteps, _ = retrieve_timesteps(
            self.pipe.scheduler,
            num_steps,
            self.device,
            **scheduler_kwargs,
        )
        flow_times = self.pipe.scheduler.sigmas.to(self.device)[:-1]
        self.schedule_debug = {
            "requested_num_inference_steps": int(num_steps),
            "scheduler_class": type(self.pipe.scheduler).__name__,
            "mu": None if mu is None else float(mu),
            "image_seq_len": int(image_seq_len),
        }
        return timesteps[: len(flow_times)], flow_times

    def predict_x0(self, latents: Any, flow_t: float) -> Any:
        import torch

        latent_model_input = (
            torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
        )
        timestep_value = flow_t * float(self.pipe.scheduler.config.num_train_timesteps)
        timestep = torch.full(
            (latent_model_input.shape[0],),
            timestep_value,
            device=latents.device,
            dtype=torch.float32,
        )
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat(
                [self.negative_prompt_embeds, self.prompt_embeds],
                dim=0,
            )
            pooled_prompt_embeds = torch.cat(
                [self.negative_pooled_prompt_embeds, self.pooled_prompt_embeds],
                dim=0,
            )
        else:
            prompt_embeds = self.prompt_embeds
            pooled_prompt_embeds = self.pooled_prompt_embeds
        noise_pred = self.pipe.transformer(
            hidden_states=latent_model_input.to(self.dtype),
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
            joint_attention_kwargs={},
            return_dict=False,
        )[0]
        if self.do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )
        flow = torch.as_tensor(flow_t, device=latents.device, dtype=noise_pred.dtype)
        return (latents.to(noise_pred.dtype) - flow * noise_pred).to(latents.dtype)

    def decode_latents(self, latents: Any) -> Image.Image:
        import torch

        with torch.no_grad():
            decoded_latents = (latents.detach() / self.pipe.vae.config.scaling_factor) + (
                self.pipe.vae.config.shift_factor
            )
            image = self.pipe.vae.decode(
                decoded_latents.to(self.pipe.vae.dtype),
                return_dict=False,
            )[0]
        return self.pipe.image_processor.postprocess(image.detach(), output_type="pil")[0].convert(
            "RGB"
        )


class SD3LanPaintModelWrapper:
    """Minimal model interface consumed by official LanPaint for SD3 flow latents."""

    def __init__(self, adapter: SD3LanPaintAdapter) -> None:
        self.adapter = adapter
        self.inner_model = self
        self.model_sampling = self

    def noise_scaling(self, sigma: Any, noise: Any, latent_image: Any) -> Any:
        return (1.0 - sigma) * latent_image + sigma * noise

    def __call__(self, x: Any, t: Any, **_kwargs: Any) -> tuple[Any, Any]:
        flow_t = float(t.flatten()[0])
        x0 = self.adapter.predict_x0(x, flow_t)
        return x0, x0


class SD3LanPaintInferenceService:
    """Lazy-loading SD3-medium service using LanPaint for masked inpainting."""

    method = SD3_LANPAINT_METHOD
    base_model_id = SD35_BASE_MODEL_ID
    pipeline_display_name = "SD3-medium LanPaint inpainting"

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
        from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import (
            StableDiffusion3Pipeline,
        )

        if reporter is not None:
            reporter.emit(
                "model_load_start",
                stage="model",
                message=f"Loading {self.pipeline_display_name} pipeline.",
                metadata={"method": self.method, "base_model_id": self.base_model_id},
            )
        timings: dict[str, float] = {}
        resolve_started = time.perf_counter()
        base_resolution = resolve_model_load_target(self.base_model_id, env_name="MODEL_PATH")
        timings["model_path_discovery_seconds"] = time.perf_counter() - resolve_started

        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        kwargs: dict[str, Any] = {
            "torch_dtype": torch.float16,
            "local_files_only": base_resolution.local_files_only,
        }
        if token:
            kwargs["token"] = token

        load_started = time.perf_counter()
        pipe = StableDiffusion3Pipeline.from_pretrained(
            base_resolution.load_target,
            **kwargs,
        )
        timings["base_from_pretrained_seconds"] = time.perf_counter() - load_started

        for name in ("text_encoder", "text_encoder_2", "text_encoder_3"):
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
            "torch_dtype": str(torch.float16),
            "device_mode": device_mode,
            "pipeline_class": type(pipe).__name__,
            "mask_route": "lanpaint_keep_mask",
            "denoising_backend": "LanPaint",
            "backend_revision": SD3_LANPAINT_BACKEND_REVISION,
            "model_family_note": "This route uses SD3-medium, not SD3.5-Large.",
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
        settings = parse_sd3_lanpaint_settings(payload)
        reporter.emit(
            "input_decode_start",
            stage="input",
            message="Decoding request image and mask.",
        )
        image, mask = load_request_images(payload)
        mask_fraction = mask_coverage(mask)
        source_image = make_sd3_lanpaint_source_by_strategy(
            image,
            mask,
            strategy=settings.sd3_source_strategy,
        )
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
                "backend_revision": SD3_LANPAINT_BACKEND_REVISION,
                "sd3_source_strategy": settings.sd3_source_strategy,
                "sd3_source_fill_radius": (
                    LANPAINT_SOURCE_FILL_RADIUS
                    if settings.sd3_source_strategy == SD3_SOURCE_STRATEGY_TELEA
                    else None
                ),
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
                    "backend_revision": SD3_LANPAINT_BACKEND_REVISION,
                    "num_inference_steps": settings.num_inference_steps,
                    "partial_noise": settings.partial_noise,
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
                    "sd3_source_strategy": settings.sd3_source_strategy,
                },
            )
            inference_started = time.perf_counter()
            adapter = SD3LanPaintAdapter(pipe)
            adapter.width, adapter.height = image.size
            adapter.prepare_prompt(
                prompt=settings.prompt,
                negative_prompt=settings.negative_prompt,
                guidance_scale=settings.guidance_scale,
                max_sequence_length=settings.max_sequence_length,
            )
            source_latent = adapter.encode_source_latents(source_image, generator)
            keep_mask, edit_mask = adapter.mask_edit_to_latents(mask)
            model = SD3LanPaintModelWrapper(adapter)

            try:
                from LanPaint.lanpaint import LanPaint
            except ImportError as exc:  # pragma: no cover - checked in container smoke.
                raise ImportError(
                    "The pinned LanPaint package is required for sd3_lanpaint."
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
            timesteps, flow_times = adapter.prepare_timesteps(settings.num_inference_steps)
            full_steps = len(flow_times)
            start_index = partial_noise_start_index(full_steps, settings.partial_noise)
            active_timesteps = timesteps[start_index:]
            active_flow_times = flow_times[start_index:]
            effective_steps = len(active_flow_times)
            if hasattr(pipe.scheduler, "set_begin_index"):
                pipe.scheduler.set_begin_index(start_index)
            noise = torch.randn(
                source_latent.shape,
                generator=generator,
                device=adapter.device,
                dtype=source_latent.dtype,
            )
            first_flow = active_flow_times[0:1].reshape(1, 1, 1, 1).to(
                device=adapter.device,
                dtype=source_latent.dtype,
            )
            latents = model.noise_scaling(first_flow, noise, source_latent).to(
                source_latent.dtype
            )

            step_trace: list[dict[str, Any]] = []
            with torch.inference_mode():
                for index, (timestep, flow_t) in enumerate(
                    zip(active_timesteps, active_flow_times, strict=True)
                ):
                    flow_value = float(flow_t.item())
                    inner_steps = (
                        0
                        if effective_steps - index
                        <= settings.lanpaint_final_outer_steps_without_inner
                        else None
                    )
                    current_times = make_current_times(
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
                    if index < effective_steps - 1:
                        next_flow = active_flow_times[index + 1 : index + 2].reshape(
                            1,
                            1,
                            1,
                            1,
                        ).to(
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
                    latents = reinject_keep_latents(
                        latents,
                        keep_reference,
                        edit_mask.to(device=adapter.device, dtype=latents.dtype),
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
                        message=f"Running SD3 LanPaint step {index + 1}/{effective_steps}.",
                        progress={"current": index + 1, "total": effective_steps},
                        metadata=step_record,
                    )
                raw = adapter.decode_latents(latents)
            inference_seconds = time.perf_counter() - inference_started
            inference_memory = {
                **memory_before_inference,
                **_cuda_memory_stats(torch, prefix="inference_"),
            }
            schedule_debug = {
                **adapter.schedule_debug,
                "partial_noise": settings.partial_noise,
                "full_num_steps": full_steps,
                "start_index": start_index,
                "effective_num_steps": effective_steps,
                "step_trace": step_trace,
            }
            reporter.emit(
                "inference_done",
                stage="inference",
                message=f"{self.pipeline_display_name} inference completed.",
                progress={"current": effective_steps, "total": effective_steps},
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
            "model": {
                **loaded.model,
                "source_image_route": settings.sd3_source_strategy,
            },
            "inference_settings": {
                "method": settings.method,
                "backend_revision": SD3_LANPAINT_BACKEND_REVISION,
                "prompt": settings.prompt,
                "negative_prompt": settings.negative_prompt,
                "partial_noise": settings.partial_noise,
                "guidance_scale": settings.guidance_scale,
                "num_inference_steps": settings.num_inference_steps,
                "effective_num_inference_steps": effective_steps,
                "max_sequence_length": settings.max_sequence_length,
                "seed": settings.seed,
                "mask_coverage": mask_fraction,
                "sd3_source_strategy": settings.sd3_source_strategy,
                "sd3_source_fill_radius": (
                    LANPAINT_SOURCE_FILL_RADIUS
                    if settings.sd3_source_strategy == SD3_SOURCE_STRATEGY_TELEA
                    else None
                ),
                "lanpaint_inner_steps": settings.lanpaint_inner_steps,
                "lanpaint_friction": settings.lanpaint_friction,
                "lanpaint_lambda": settings.lanpaint_lambda,
                "lanpaint_beta": settings.lanpaint_beta,
                "lanpaint_step_size": settings.lanpaint_step_size,
                "lanpaint_final_outer_steps_without_inner": (
                    settings.lanpaint_final_outer_steps_without_inner
                ),
                "reinject_keep_latents": True,
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
    "SD3_LANPAINT_BACKEND_REVISION",
    "SD3LanPaintAdapter",
    "SD3LanPaintInferenceService",
    "make_sd3_lanpaint_source_by_strategy",
    "parse_sd3_lanpaint_settings",
]
