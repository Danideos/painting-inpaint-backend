"""FLUX.1-Canny restoration using LanPaint masked denoising."""

from __future__ import annotations

import logging
import math
import os
import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image, ImageFilter

from .image_helpers import (
    binarize_mask,
    ensure_same_size,
    hard_composite,
    mask_coverage,
    outside_mask_changed,
)
from .image_io import image_to_base64, load_request_image
from .inference import (
    WorkerInputError,
    _cuda_memory_stats,
    _json_timestep,
    _reset_cuda_peak_memory,
    load_request_images,
    parse_request_settings,
)
from .methods import FLUX_CANNY_LANPAINT_METHOD
from .model_loading import (
    LoadedPipeline,
    env_flag,
    load_lora_if_available,
    resolve_model_load_target,
    resolve_torch_dtype,
    set_lora_scale,
)
from .partial_noise import partial_noise_start_index
from .progress import ProgressReporter

LOGGER = logging.getLogger(__name__)

CANNY_MODEL_ID = "black-forest-labs/FLUX.1-Canny-dev"
CANNY_LANPAINT_BACKEND_REVISION = "lanpaint-telea-source-v1"
CANNY_LANPAINT_NATIVE_BACKEND_REVISION = "flux-canny-lanpaint-native-v1"
CANNY_LOW_THRESHOLD = 75
CANNY_HIGH_THRESHOLD = 150
CANNY_BLUR_RADIUS = 0.0
CANNY_INK_THRESHOLD = 245
CANNY_BOUNDARY_FILL_RADIUS = 3.0
CANNY_MASK_GUARD_PIXELS = 0
LANPAINT_SOURCE_FILL_RADIUS = 3.0
LANPAINT_INNER_STEPS = 10
LANPAINT_FRICTION = 15.0
LANPAINT_LAMBDA = 10.0
LANPAINT_BETA = 1.0
LANPAINT_STEP_SIZE = 0.2
LANPAINT_FINAL_OUTER_STEPS_WITHOUT_INNER = 3
LANPAINT_SCHEDULE_MODE = "diffusers_default"


@dataclass(frozen=True)
class CannyLanPaintRequestSettings:
    """Request-overridable Canny/LanPaint controls with validated defaults."""

    canny_low_threshold: int = CANNY_LOW_THRESHOLD
    canny_high_threshold: int = CANNY_HIGH_THRESHOLD
    canny_blur_radius: float = CANNY_BLUR_RADIUS
    canny_ink_threshold: int = CANNY_INK_THRESHOLD
    canny_boundary_fill_radius: float = CANNY_BOUNDARY_FILL_RADIUS
    canny_mask_guard_pixels: int = CANNY_MASK_GUARD_PIXELS
    lanpaint_inner_steps: int = LANPAINT_INNER_STEPS
    lanpaint_friction: float = LANPAINT_FRICTION
    lanpaint_lambda: float = LANPAINT_LAMBDA
    lanpaint_beta: float = LANPAINT_BETA
    lanpaint_step_size: float = LANPAINT_STEP_SIZE
    lanpaint_final_outer_steps_without_inner: int = (
        LANPAINT_FINAL_OUTER_STEPS_WITHOUT_INNER
    )
    reencode_source_latent_interval: int = 0
    canny_control_strategy: str | None = None  # None = auto; "masked_ink" = force masked_ink


def _payload_int(payload: dict[str, Any], key: str, default: int) -> int:
    value = payload.get(key, default)
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise WorkerInputError(f"{key} must be an integer.") from exc


def _payload_float(payload: dict[str, Any], key: str, default: float) -> float:
    value = payload.get(key, default)
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise WorkerInputError(f"{key} must be a number.") from exc
    if not math.isfinite(number):
        raise WorkerInputError(f"{key} must be finite.")
    return number


def parse_canny_lanpaint_settings(payload: dict[str, Any]) -> CannyLanPaintRequestSettings:
    """Parse method-specific request controls without changing Fill behavior."""

    low = _payload_int(payload, "canny_low_threshold", CANNY_LOW_THRESHOLD)
    high = _payload_int(payload, "canny_high_threshold", CANNY_HIGH_THRESHOLD)
    if not 0 <= low <= 255:
        raise WorkerInputError("canny_low_threshold must be between 0 and 255.")
    if not 0 <= high <= 255:
        raise WorkerInputError("canny_high_threshold must be between 0 and 255.")
    if high < low:
        raise WorkerInputError(
            "canny_high_threshold must be greater than or equal to canny_low_threshold."
        )

    blur = _payload_float(payload, "canny_blur_radius", CANNY_BLUR_RADIUS)
    if blur < 0:
        raise WorkerInputError("canny_blur_radius must be non-negative.")
    ink_threshold = _payload_int(payload, "canny_ink_threshold", CANNY_INK_THRESHOLD)
    if not 0 <= ink_threshold <= 255:
        raise WorkerInputError("canny_ink_threshold must be between 0 and 255.")
    boundary_fill_radius = _payload_float(
        payload,
        "canny_boundary_fill_radius",
        CANNY_BOUNDARY_FILL_RADIUS,
    )
    if boundary_fill_radius < 0:
        raise WorkerInputError("canny_boundary_fill_radius must be non-negative.")
    mask_guard_pixels = _payload_int(
        payload,
        "canny_mask_guard_pixels",
        CANNY_MASK_GUARD_PIXELS,
    )
    if mask_guard_pixels < 0:
        raise WorkerInputError("canny_mask_guard_pixels must be non-negative.")

    inner_steps = _payload_int(payload, "lanpaint_inner_steps", LANPAINT_INNER_STEPS)
    if inner_steps < 0:
        raise WorkerInputError("lanpaint_inner_steps must be non-negative.")

    final_without_inner = _payload_int(
        payload,
        "lanpaint_final_outer_steps_without_inner",
        LANPAINT_FINAL_OUTER_STEPS_WITHOUT_INNER,
    )
    if final_without_inner < 0:
        raise WorkerInputError(
            "lanpaint_final_outer_steps_without_inner must be non-negative."
        )

    friction = _payload_float(payload, "lanpaint_friction", LANPAINT_FRICTION)
    lanpaint_lambda = _payload_float(payload, "lanpaint_lambda", LANPAINT_LAMBDA)
    beta = _payload_float(payload, "lanpaint_beta", LANPAINT_BETA)
    step_size = _payload_float(payload, "lanpaint_step_size", LANPAINT_STEP_SIZE)
    if friction < 0:
        raise WorkerInputError("lanpaint_friction must be non-negative.")
    if lanpaint_lambda < 0:
        raise WorkerInputError("lanpaint_lambda must be non-negative.")
    if beta < 0:
        raise WorkerInputError("lanpaint_beta must be non-negative.")
    if step_size <= 0:
        raise WorkerInputError("lanpaint_step_size must be greater than 0.")

    raw_interval = payload.get("reencode_source_latent_interval", None)
    if raw_interval is None:
        legacy = payload.get("reencode_source_latent_every_step", False)
        reencode_interval = 1 if legacy else 0
    else:
        reencode_interval = int(raw_interval)
    if reencode_interval < 0:
        raise WorkerInputError("reencode_source_latent_interval must be non-negative.")

    canny_control_strategy = payload.get("canny_control_strategy", None)
    if canny_control_strategy is not None and canny_control_strategy not in {"masked_ink"}:
        raise WorkerInputError(
            "canny_control_strategy must be 'masked_ink' or omitted."
        )

    return CannyLanPaintRequestSettings(
        canny_low_threshold=low,
        canny_high_threshold=high,
        canny_blur_radius=blur,
        canny_ink_threshold=ink_threshold,
        canny_boundary_fill_radius=boundary_fill_radius,
        canny_mask_guard_pixels=mask_guard_pixels,
        lanpaint_inner_steps=inner_steps,
        lanpaint_friction=friction,
        lanpaint_lambda=lanpaint_lambda,
        lanpaint_beta=beta,
        lanpaint_step_size=step_size,
        lanpaint_final_outer_steps_without_inner=final_without_inner,
        reencode_source_latent_interval=reencode_interval,
        canny_control_strategy=canny_control_strategy,
    )


def make_canny_control(
    image: Image.Image,
    *,
    low_threshold: int = CANNY_LOW_THRESHOLD,
    high_threshold: int = CANNY_HIGH_THRESHOLD,
    blur_radius: float = CANNY_BLUR_RADIUS,
) -> Image.Image:
    """Create the validated three-channel Canny control image."""

    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - exercised by image import smoke.
        raise ImportError(
            "FLUX-Canny control generation requires opencv-python-headless."
        ) from exc

    source = image.convert("RGB")
    if blur_radius > 0:
        source = source.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    array = np.asarray(source, dtype=np.uint8)
    gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, threshold1=low_threshold, threshold2=high_threshold)
    return Image.fromarray(np.stack([edges, edges, edges], axis=-1).astype(np.uint8)).convert("RGB")


def _canny_edges(
    image: Image.Image,
    *,
    low_threshold: int,
    high_threshold: int,
    blur_radius: float,
) -> np.ndarray:
    """Create a one-channel Canny edge map."""

    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - exercised by image import smoke.
        raise ImportError(
            "FLUX-Canny control generation requires opencv-python-headless."
        ) from exc

    source = image.convert("RGB")
    if blur_radius > 0:
        source = source.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    array = np.asarray(source, dtype=np.uint8)
    gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
    return cv2.Canny(gray, threshold1=low_threshold, threshold2=high_threshold)


def _erode_binary_mask(mask: Image.Image, pixels: int) -> Image.Image:
    if pixels <= 0:
        return binarize_mask(mask)
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - exercised by image import smoke.
        raise ImportError(
            "FLUX-Canny control generation requires opencv-python-headless."
        ) from exc

    mask_array = np.asarray(binarize_mask(mask), dtype=np.uint8)
    kernel = np.ones((pixels * 2 + 1, pixels * 2 + 1), dtype=np.uint8)
    return Image.fromarray(cv2.erode(mask_array, kernel)).convert("L")


def _fill_excluded_region_for_canny(
    image: Image.Image,
    excluded_region: Image.Image,
    *,
    radius: float,
) -> Image.Image:
    if radius <= 0:
        return image.convert("RGB")
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - exercised by image import smoke.
        raise ImportError(
            "FLUX-Canny control generation requires opencv-python-headless."
        ) from exc

    image_array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    mask_array = np.asarray(binarize_mask(excluded_region), dtype=np.uint8)
    filled = cv2.inpaint(bgr, mask_array, radius, cv2.INPAINT_TELEA)
    rgb = cv2.cvtColor(filled, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb).convert("RGB")


def make_lanpaint_source_image(
    image: Image.Image,
    mask: Image.Image,
    *,
    fill_radius: float = LANPAINT_SOURCE_FILL_RADIUS,
) -> Image.Image:
    """Prepare a content-aware inpaint source before VAE source-latent encoding."""

    source = image.convert("RGB")
    edit_mask = binarize_mask(mask)
    if mask_coverage(edit_mask) <= 0:
        return source.copy()
    if mask_coverage(edit_mask) >= 1:
        return Image.new("RGB", source.size, (127, 127, 127))
    if fill_radius < 0:
        raise ValueError("fill_radius must be non-negative.")

    filled = _fill_excluded_region_for_canny(source, edit_mask, radius=fill_radius)
    return hard_composite(source, filled, edit_mask)


def _ink_mask_from_guidance(
    image: Image.Image,
    mask: Image.Image,
    *,
    ink_threshold: int,
) -> Image.Image:
    gray = np.asarray(image.convert("L"), dtype=np.uint8)
    edit_mask = np.asarray(binarize_mask(mask), dtype=np.uint8) > 0
    ink = ((gray < ink_threshold) & edit_mask).astype(np.uint8) * 255
    return Image.fromarray(ink).convert("L")


def _zhang_suen_thinning(binary_image: Image.Image) -> Image.Image:
    image = (np.asarray(binary_image.convert("L"), dtype=np.uint8) > 0).astype(np.uint8)

    while True:
        changed = False
        for step in (0, 1):
            padded = np.pad(image, 1, mode="constant")
            p2 = padded[:-2, 1:-1]
            p3 = padded[:-2, 2:]
            p4 = padded[1:-1, 2:]
            p5 = padded[2:, 2:]
            p6 = padded[2:, 1:-1]
            p7 = padded[2:, :-2]
            p8 = padded[1:-1, :-2]
            p9 = padded[:-2, :-2]

            neighbor_count = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
            transitions = (
                ((p2 == 0) & (p3 == 1)).astype(np.uint8)
                + ((p3 == 0) & (p4 == 1)).astype(np.uint8)
                + ((p4 == 0) & (p5 == 1)).astype(np.uint8)
                + ((p5 == 0) & (p6 == 1)).astype(np.uint8)
                + ((p6 == 0) & (p7 == 1)).astype(np.uint8)
                + ((p7 == 0) & (p8 == 1)).astype(np.uint8)
                + ((p8 == 0) & (p9 == 1)).astype(np.uint8)
                + ((p9 == 0) & (p2 == 1)).astype(np.uint8)
            )

            if step == 0:
                condition_a = p2 * p4 * p6 == 0
                condition_b = p4 * p6 * p8 == 0
            else:
                condition_a = p2 * p4 * p8 == 0
                condition_b = p2 * p6 * p8 == 0

            remove = (
                (image == 1)
                & (neighbor_count >= 2)
                & (neighbor_count <= 6)
                & (transitions == 1)
                & condition_a
                & condition_b
            )
            if np.any(remove):
                image[remove] = 0
                changed = True

        if not changed:
            break

    return Image.fromarray(image * 255).convert("L")


def make_masked_canny_control(
    image: Image.Image,
    mask: Image.Image,
    *,
    low_threshold: int = CANNY_LOW_THRESHOLD,
    high_threshold: int = CANNY_HIGH_THRESHOLD,
    blur_radius: float = CANNY_BLUR_RADIUS,
    ink_threshold: int = CANNY_INK_THRESHOLD,
    boundary_fill_radius: float = CANNY_BOUNDARY_FILL_RADIUS,
    mask_guard_pixels: int = CANNY_MASK_GUARD_PIXELS,
) -> Image.Image:
    """Create Canny control from submitted image + edit mask without mask-border edges."""

    source = image.convert("RGB")
    edit_mask = binarize_mask(mask)
    outside_mask = Image.eval(edit_mask, lambda value: 255 - value)
    inside_core = _erode_binary_mask(edit_mask, mask_guard_pixels)
    outside_core = _erode_binary_mask(outside_mask, mask_guard_pixels)

    outside_source = _fill_excluded_region_for_canny(
        source,
        edit_mask,
        radius=boundary_fill_radius,
    )
    outside_edges = _canny_edges(
        outside_source,
        low_threshold=low_threshold,
        high_threshold=high_threshold,
        blur_radius=blur_radius,
    )
    outside = np.where(np.asarray(outside_core, dtype=np.uint8) > 0, outside_edges, 0)

    ink = _ink_mask_from_guidance(source, edit_mask, ink_threshold=ink_threshold)
    thinned_ink = _zhang_suen_thinning(ink)
    inside = np.where(np.asarray(inside_core, dtype=np.uint8) > 0, np.asarray(thinned_ink), 0)

    combined = np.maximum(outside, inside).astype(np.uint8)
    return Image.fromarray(np.stack([combined, combined, combined], axis=-1)).convert("RGB")


def mask_edit_to_lanpaint_keep(mask_edit: Image.Image) -> Image.Image:
    """Convert white-edit project masks to white-keep LanPaint masks."""

    edit = binarize_mask(mask_edit)
    return Image.eval(edit, lambda value: 255 - value)


def reinject_keep_latents(latents: Any, keep_reference: Any, edit_mask: Any) -> Any:
    """Restore the noised source latent outside the editable region."""

    return edit_mask * latents + (1.0 - edit_mask) * keep_reference


def make_current_times(flow_t: float, device: Any, dtype: Any) -> tuple[Any, Any, Any]:
    """Return LanPaint's variance-exploding, alpha-bar, and flow time tuple."""

    import torch

    flow = torch.tensor([flow_t], device=device, dtype=dtype)
    alpha_bar = (1.0 - flow) ** 2 / ((1.0 - flow) ** 2 + flow**2)
    ve_sigma = flow / torch.clamp(1.0 - flow, min=1e-6)
    return ve_sigma, alpha_bar, flow


class FluxCannyLanPaintAdapter:
    """Expose Diffusers FLUX-Canny components to official LanPaint."""

    def __init__(self, pipe: Any) -> None:
        self.pipe = pipe
        self.prompt_embeds = None
        self.pooled_prompt_embeds = None
        self.text_ids = None
        self.source_latent = None
        self.control_latent = None
        self.latent_image_ids = None
        self.height: int | None = None
        self.width: int | None = None
        self.schedule_debug: dict[str, Any] = {}

    @property
    def device(self) -> Any:
        return self.pipe._execution_device

    @property
    def dtype(self) -> Any:
        return self.pipe.transformer.dtype

    def encode_prompt(self, prompt: str, *, max_sequence_length: int) -> None:
        kwargs: dict[str, Any] = {
            "prompt": prompt,
            "prompt_2": None,
            "device": self.device,
            "num_images_per_prompt": 1,
            "max_sequence_length": max_sequence_length,
        }
        self.prompt_embeds, self.pooled_prompt_embeds, self.text_ids = (
            self.pipe.encode_prompt(**kwargs)
        )

    def encode_source_latents(self, image: Image.Image, generator: Any) -> Any:
        import torch

        if self.height is None or self.width is None:
            self.width, self.height = image.size
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
        batch_size, channels = encoded.shape[:2]
        latent_height = 2 * (self.height // (self.pipe.vae_scale_factor * 2))
        latent_width = 2 * (self.width // (self.pipe.vae_scale_factor * 2))
        self.source_latent = self.pipe._pack_latents(
            encoded.to(self.dtype),
            batch_size,
            channels,
            latent_height,
            latent_width,
        )
        self.latent_image_ids = self.pipe._prepare_latent_image_ids(
            batch_size,
            latent_height // 2,
            latent_width // 2,
            self.device,
            self.prompt_embeds.dtype,
        )
        return self.source_latent

    def encode_control_latents(self, control_image: Image.Image, generator: Any) -> Any:
        import torch

        if self.height is None or self.width is None:
            raise RuntimeError("Source latents must be encoded before control latents.")
        tensor = self.pipe.prepare_image(
            image=control_image.convert("RGB"),
            width=self.width,
            height=self.height,
            batch_size=1,
            num_images_per_prompt=1,
            device=self.device,
            dtype=self.pipe.vae.dtype,
        )
        with torch.inference_mode():
            encoded = self.pipe.vae.encode(tensor).latent_dist.sample(generator=generator)
        encoded = (encoded - self.pipe.vae.config.shift_factor) * (
            self.pipe.vae.config.scaling_factor
        )
        batch_size, channels, latent_height, latent_width = encoded.shape
        self.control_latent = self.pipe._pack_latents(
            encoded.to(self.dtype),
            batch_size,
            channels,
            latent_height,
            latent_width,
        )
        if self.control_latent.shape[:2] != self.source_latent.shape[:2]:
            raise RuntimeError("Packed Canny control and source latent shapes do not match.")
        return self.control_latent

    def mask_keep_to_latents(self, mask_keep: Image.Image) -> tuple[Any, Any]:
        import torch

        if self.source_latent is None or self.height is None or self.width is None:
            raise RuntimeError("Source latents must be encoded before the mask.")
        unpacked = self.pipe._unpack_latents(
            self.source_latent,
            self.height,
            self.width,
            self.pipe.vae_scale_factor,
        )
        latent_height, latent_width = unpacked.shape[-2:]
        mask_array = np.asarray(mask_keep.convert("L"), dtype=np.float32) / 255.0
        keep = torch.from_numpy(mask_array)[None, None].to(self.device, dtype=torch.float32)
        keep = torch.nn.functional.interpolate(
            keep,
            size=(latent_height, latent_width),
            mode="nearest",
        )
        edit = 1.0 - keep
        packed_edit = torch.nn.functional.max_pool2d(edit, kernel_size=2, stride=2)
        packed_keep = 1.0 - packed_edit
        packed_keep = packed_keep.flatten(2).transpose(1, 2).contiguous()
        packed_edit = packed_edit.flatten(2).transpose(1, 2).contiguous()
        if packed_keep.shape[1] != self.source_latent.shape[1]:
            raise RuntimeError("Packed LanPaint mask and source latent lengths do not match.")
        return packed_keep, packed_edit

    def prepare_timesteps(self, num_steps: int) -> tuple[Any, Any]:
        from diffusers.pipelines.flux.pipeline_flux import calculate_shift, retrieve_timesteps

        image_seq_len = self.source_latent.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.pipe.scheduler.config.get("base_image_seq_len", 256),
            self.pipe.scheduler.config.get("max_image_seq_len", 4096),
            self.pipe.scheduler.config.get("base_shift", 0.5),
            self.pipe.scheduler.config.get("max_shift", 1.15),
        )
        use_default_flow_sigmas = bool(
            getattr(self.pipe.scheduler.config, "use_flow_sigmas", False)
        )
        sigmas = None if use_default_flow_sigmas else np.linspace(1.0, 1.0 / num_steps, num_steps)
        timesteps, _ = retrieve_timesteps(
            self.pipe.scheduler,
            num_steps,
            self.device,
            sigmas=sigmas,
            mu=mu,
        )
        flow_times = self.pipe.scheduler.sigmas.to(self.device)[:-1]
        self.schedule_debug = {
            "schedule_mode": LANPAINT_SCHEDULE_MODE,
            "requested_num_inference_steps": int(num_steps),
            "scheduler_class": type(self.pipe.scheduler).__name__,
            "used_scheduler_default_flow_sigmas": use_default_flow_sigmas,
            "mu": float(mu),
        }
        return timesteps[: len(flow_times)], flow_times

    def predict_x0(self, latents: Any, flow_t: float, guidance_scale: float) -> Any:
        import torch

        timestep = torch.full(
            (latents.shape[0],),
            flow_t,
            device=latents.device,
            dtype=self.dtype,
        )
        guidance = None
        if self.pipe.transformer.config.guidance_embeds:
            guidance = torch.full(
                (latents.shape[0],),
                guidance_scale,
                device=latents.device,
                dtype=torch.float32,
            )
        model_input = torch.cat(
            [latents, self.control_latent.to(device=latents.device, dtype=latents.dtype)],
            dim=2,
        )
        cache_context = getattr(self.pipe.transformer, "cache_context", None)
        context = cache_context("cond") if cache_context is not None else nullcontext()
        with context:
            noise_pred = self.pipe.transformer(
                hidden_states=model_input.to(self.dtype),
                timestep=timestep,
                guidance=guidance,
                pooled_projections=self.pooled_prompt_embeds,
                encoder_hidden_states=self.prompt_embeds,
                txt_ids=self.text_ids,
                img_ids=self.latent_image_ids,
                joint_attention_kwargs={},
                return_dict=False,
            )[0]
        flow = torch.as_tensor(flow_t, device=latents.device, dtype=noise_pred.dtype)
        return (latents.to(noise_pred.dtype) - flow * noise_pred).to(latents.dtype)

    def decode_latents(self, latents: Any) -> Image.Image:
        import torch

        with torch.no_grad():
            unpacked = self.pipe._unpack_latents(
                latents.detach(),
                self.height,
                self.width,
                self.pipe.vae_scale_factor,
            )
            unpacked = (unpacked / self.pipe.vae.config.scaling_factor) + (
                self.pipe.vae.config.shift_factor
            )
            image = self.pipe.vae.decode(
                unpacked.to(self.pipe.vae.dtype),
                return_dict=False,
            )[0]
        return self.pipe.image_processor.postprocess(image.detach(), output_type="pil")[0].convert(
            "RGB"
        )

    def reencode_source_from_prediction(
        self,
        x0: Any,
        original_image: Image.Image,
        mask_edit: Image.Image,
    ) -> Any:
        """Decode x0, composite original keep-region pixels back in, re-encode.

        Using latent_dist.mean rather than .sample keeps each re-encode
        deterministic — no generator state consumed, no stochastic jitter
        between steps.
        """
        import torch

        decoded = self.decode_latents(x0)
        composited = hard_composite(
            original_image.convert("RGB"),
            decoded,
            binarize_mask(mask_edit),
        )
        image_tensor = self.pipe.image_processor.preprocess(
            composited,
            height=self.height,
            width=self.width,
        ).to(device=self.device, dtype=self.pipe.vae.dtype)
        with torch.inference_mode():
            encoded = self.pipe.vae.encode(image_tensor).latent_dist.mean
        encoded = (encoded - self.pipe.vae.config.shift_factor) * (
            self.pipe.vae.config.scaling_factor
        )
        batch_size, channels = encoded.shape[:2]
        latent_height = 2 * (self.height // (self.pipe.vae_scale_factor * 2))
        latent_width = 2 * (self.width // (self.pipe.vae_scale_factor * 2))
        self.source_latent = self.pipe._pack_latents(
            encoded.to(self.dtype),
            batch_size,
            channels,
            latent_height,
            latent_width,
        )
        return self.source_latent


class FluxCannyLanPaintModelWrapper:
    """Minimal model interface consumed by official LanPaint."""

    def __init__(self, adapter: FluxCannyLanPaintAdapter, guidance_scale: float) -> None:
        self.adapter = adapter
        self.guidance_scale = float(guidance_scale)
        self.inner_model = self
        self.model_sampling = self

    def noise_scaling(self, sigma: Any, noise: Any, latent_image: Any) -> Any:
        return (1.0 - sigma) * latent_image + sigma * noise

    def __call__(self, x: Any, t: Any, **_kwargs: Any) -> tuple[Any, Any]:
        flow_t = float(t.flatten()[0])
        x0 = self.adapter.predict_x0(x, flow_t, self.guidance_scale)
        return x0, x0


def load_flux_canny_pipeline(
    *, reporter: ProgressReporter | None = None
) -> LoadedPipeline:
    """Load FLUX-Canny and its required restoration LoRA from local assets."""

    import torch
    from diffusers import FluxControlPipeline

    if not torch.cuda.is_available():
        raise RuntimeError("FLUX-Canny/LanPaint requires a CUDA GPU.")
    if reporter is not None:
        reporter.emit(
            "model_load_start",
            stage="model",
            message="Loading FLUX-Canny pipeline.",
            metadata={"model_id": CANNY_MODEL_ID, "method": FLUX_CANNY_LANPAINT_METHOD},
        )
    timings: dict[str, float] = {}
    resolve_started = time.perf_counter()
    resolution = resolve_model_load_target(CANNY_MODEL_ID)
    timings["model_path_discovery_seconds"] = time.perf_counter() - resolve_started
    torch_dtype = resolve_torch_dtype(torch)
    kwargs: dict[str, Any] = {
        "torch_dtype": torch_dtype,
        "local_files_only": resolution.local_files_only,
    }
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if token:
        kwargs["token"] = token
    load_started = time.perf_counter()
    pipe = FluxControlPipeline.from_pretrained(resolution.load_target, **kwargs)
    timings["from_pretrained_seconds"] = time.perf_counter() - load_started
    device_started = time.perf_counter()
    if env_flag("ENABLE_MODEL_CPU_OFFLOAD", default=True) and hasattr(
        pipe, "enable_model_cpu_offload"
    ):
        pipe.enable_model_cpu_offload()
        device_mode = "model_cpu_offload"
    else:
        pipe.to("cuda")
        device_mode = "cuda"
    if getattr(pipe, "vae", None) is not None:
        if env_flag("ENABLE_VAE_TILING", default=True) and hasattr(pipe.vae, "enable_tiling"):
            pipe.vae.enable_tiling()
        if env_flag("ENABLE_VAE_SLICING", default=True) and hasattr(pipe.vae, "enable_slicing"):
            pipe.vae.enable_slicing()
    timings["device_setup_seconds"] = time.perf_counter() - device_started
    model = {
        "method": FLUX_CANNY_LANPAINT_METHOD,
        "model_id": CANNY_MODEL_ID,
        "load_target": resolution.load_target,
        "source": resolution.source,
        "local_files_only": resolution.local_files_only,
        "snapshot_path": resolution.snapshot_path,
        "torch_dtype": str(torch_dtype),
        "device_mode": device_mode,
        "pipeline_class": type(pipe).__name__,
        "denoising_backend": "LanPaint",
    }
    if reporter is not None:
        reporter.emit(
            "model_load_done",
            stage="model",
            message="FLUX-Canny pipeline loaded.",
            metadata={**model, "timings": timings},
        )
    lora = load_lora_if_available(pipe, reporter=reporter)
    timings["lora_loading_seconds"] = float(lora.get("elapsed_seconds", 0.0))
    return LoadedPipeline(
        pipe=pipe,
        torch=torch,
        torch_dtype=torch_dtype,
        model=model,
        lora=lora,
        supports_negative_prompt=False,
        timings=timings,
    )


def load_control_image(
    payload: dict[str, Any],
    *,
    fallback: Image.Image,
) -> tuple[Image.Image, str]:
    """Decode an optional control source or reuse the restoration input."""

    if "control_image_base64" not in payload and "control_image_url" not in payload:
        return fallback.copy(), "input_image"
    try:
        control = load_request_image(
            payload,
            label="control image",
            url_key="control_image_url",
            base64_key="control_image_base64",
            mode="RGB",
        )
        ensure_same_size([fallback, control])
    except ValueError as exc:
        raise WorkerInputError(str(exc)) from exc
    return control.convert("RGB"), "request_control_image"


@dataclass
class FluxCannyLanPaintService:
    """Lazy-loading Modal service for FLUX-Canny/LanPaint restoration."""

    method: str = FLUX_CANNY_LANPAINT_METHOD
    native_lanpaint: bool = False
    _loaded: LoadedPipeline | None = None

    @property
    def backend_revision(self) -> str:
        if self.native_lanpaint:
            return CANNY_LANPAINT_NATIVE_BACKEND_REVISION
        return CANNY_LANPAINT_BACKEND_REVISION

    def get_loaded(self, *, reporter: ProgressReporter | None = None) -> LoadedPipeline:
        if self._loaded is None:
            self._loaded = load_flux_canny_pipeline(reporter=reporter)
            self._loaded.model["method"] = self.method
            self._loaded.model["backend_revision"] = self.backend_revision
        elif reporter is not None:
            reporter.emit(
                "model_load_start",
                stage="model",
                message="Reusing cached FLUX-Canny pipeline.",
                metadata={"cached": True, "method": self.method},
            )
            reporter.emit(
                "model_load_done",
                stage="model",
                message="Cached FLUX-Canny pipeline ready.",
                metadata={"cached": True, "model": self._loaded.model},
            )
        return self._loaded

    def run(
        self,
        payload: dict[str, Any],
        *,
        reporter: ProgressReporter | None = None,
    ) -> dict[str, Any]:
        reporter = reporter or ProgressReporter(enabled=False)
        settings = parse_request_settings(payload)
        method_settings = parse_canny_lanpaint_settings(payload)
        if settings.method != self.method:
            raise WorkerInputError(
                f"The FLUX-Canny service requires method={self.method!r}."
            )
        backend_revision = self.backend_revision
        if self.native_lanpaint:
            latent_source_strategy = "raw_input_image"
            latent_source_fill_radius = None
        else:
            latent_source_strategy = "telea_filled_source"
            latent_source_fill_radius = LANPAINT_SOURCE_FILL_RADIUS
        reporter.emit("input_decode_start", stage="input", message="Decoding request inputs.")
        image, mask = load_request_images(payload)
        control_source, control_source_name = load_control_image(payload, fallback=image)
        reporter.emit(
            "input_decode_done",
            stage="input",
            message="Request image, mask, and control source decoded.",
            metadata={
                "image_width": image.width,
                "image_height": image.height,
                "mask_coverage": mask_coverage(mask),
                "control_source": control_source_name,
                "backend_revision": backend_revision,
            },
        )
        reporter.emit(
            "canny_control_start",
            stage="input",
            message="Preparing Canny control image.",
            metadata={
                "control_source": control_source_name,
                "backend_revision": backend_revision,
                "low_threshold": method_settings.canny_low_threshold,
                "high_threshold": method_settings.canny_high_threshold,
                "blur_radius": method_settings.canny_blur_radius,
                "ink_threshold": method_settings.canny_ink_threshold,
                "boundary_fill_radius": method_settings.canny_boundary_fill_radius,
                "mask_guard_pixels": method_settings.canny_mask_guard_pixels,
            },
        )
        control_started = time.perf_counter()
        use_masked_ink = (
            method_settings.canny_control_strategy == "masked_ink"
            or (not self.native_lanpaint and control_source_name == "input_image")
        )
        if use_masked_ink:
            canny_control = make_masked_canny_control(
                control_source,
                mask,
                low_threshold=method_settings.canny_low_threshold,
                high_threshold=method_settings.canny_high_threshold,
                blur_radius=method_settings.canny_blur_radius,
                ink_threshold=method_settings.canny_ink_threshold,
                boundary_fill_radius=method_settings.canny_boundary_fill_radius,
                mask_guard_pixels=method_settings.canny_mask_guard_pixels,
            )
            control_strategy = "masked_ink_composite"
        elif self.native_lanpaint:
            canny_control = make_canny_control(
                control_source,
                low_threshold=method_settings.canny_low_threshold,
                high_threshold=method_settings.canny_high_threshold,
                blur_radius=method_settings.canny_blur_radius,
            )
            control_strategy = "plain_canny_control"
        else:
            canny_control = make_canny_control(
                control_source,
                low_threshold=method_settings.canny_low_threshold,
                high_threshold=method_settings.canny_high_threshold,
                blur_radius=method_settings.canny_blur_radius,
            )
            control_strategy = "explicit_control_image_canny"
        control_seconds = time.perf_counter() - control_started
        reporter.emit(
            "canny_control_done",
            stage="input",
            message="Canny control image prepared.",
            metadata={
                "control_source": control_source_name,
                "backend_revision": backend_revision,
                "low_threshold": method_settings.canny_low_threshold,
                "high_threshold": method_settings.canny_high_threshold,
                "blur_radius": method_settings.canny_blur_radius,
                "ink_threshold": method_settings.canny_ink_threshold,
                "boundary_fill_radius": method_settings.canny_boundary_fill_radius,
                "mask_guard_pixels": method_settings.canny_mask_guard_pixels,
                "control_strategy": control_strategy,
                "elapsed_seconds": control_seconds,
            },
        )

        loaded = self.get_loaded(reporter=reporter)
        reporter.emit(
            "lora_scale_apply_start",
            stage="lora",
            message="Applying request LoRA scale.",
            metadata={"requested_scale": settings.lora_scale},
        )
        scale = set_lora_scale(
            loaded.pipe,
            adapter_name=loaded.lora.get("adapter_name") if loaded.lora.get("loaded") else None,
            lora_scale=settings.lora_scale,
        )
        reporter.emit(
            "lora_scale_apply_done",
            stage="lora",
            message="Request LoRA scale applied.",
            metadata={
                "requested_scale": settings.lora_scale,
                "effective_scale": scale["effective_scale"],
                "mode": scale["mode"],
            },
        )
        torch = loaded.torch
        adapter = FluxCannyLanPaintAdapter(loaded.pipe)
        adapter.width, adapter.height = image.size
        latent_source_image = (
            image.convert("RGB")
            if self.native_lanpaint
            else make_lanpaint_source_image(
                image,
                mask,
            )
        )
        generator = torch.Generator(device=adapter.device)
        if settings.seed is not None:
            generator.manual_seed(settings.seed)
        else:
            generator.seed()

        memory_before = _cuda_memory_stats(torch, prefix="pre_inference_")
        _reset_cuda_peak_memory(torch)
        inference_started = time.perf_counter()
        adapter.encode_prompt(
            settings.prompt,
            max_sequence_length=settings.max_sequence_length,
        )
        source_latent = adapter.encode_source_latents(latent_source_image, generator)
        control_latent = adapter.encode_control_latents(canny_control, generator)
        keep_mask = mask_edit_to_lanpaint_keep(mask)
        keep_latent, edit_latent = adapter.mask_keep_to_latents(keep_mask)
        model = FluxCannyLanPaintModelWrapper(adapter, settings.guidance_scale)

        try:
            from LanPaint.lanpaint import LanPaint
        except ImportError as exc:  # pragma: no cover - checked in container smoke.
            raise ImportError("The pinned LanPaint package is required for this method.") from exc

        lanpaint = LanPaint(
            Model=model,
            NSteps=method_settings.lanpaint_inner_steps,
            Friction=method_settings.lanpaint_friction,
            Lambda=method_settings.lanpaint_lambda,
            Beta=method_settings.lanpaint_beta,
            StepSize=method_settings.lanpaint_step_size,
            IS_FLUX=True,
            IS_FLOW=True,
        )
        timesteps, flow_times = adapter.prepare_timesteps(settings.num_inference_steps)
        full_steps = len(flow_times)
        start_index = partial_noise_start_index(full_steps, settings.partial_noise)
        active_timesteps = timesteps[start_index:]
        active_flow_times = flow_times[start_index:]
        effective_steps = len(active_flow_times)
        if hasattr(adapter.pipe.scheduler, "set_begin_index"):
            adapter.pipe.scheduler.set_begin_index(start_index)
        noise = torch.randn(
            source_latent.shape,
            generator=generator,
            device=adapter.device,
            dtype=source_latent.dtype,
        )
        first_flow = active_flow_times[0:1].reshape(1, 1, 1).to(
            device=adapter.device,
            dtype=source_latent.dtype,
        )
        latents = model.noise_scaling(first_flow, noise, source_latent).to(source_latent.dtype)
        reporter.emit(
            "inference_start",
            stage="inference",
            message="Starting FLUX-Canny LanPaint inference.",
            progress={"current": 0, "total": effective_steps},
            metadata={
                "method": settings.method,
                "backend_revision": backend_revision,
                "requested_num_inference_steps": settings.num_inference_steps,
                "effective_num_inference_steps": effective_steps,
                "partial_noise": settings.partial_noise,
                "guidance_scale": settings.guidance_scale,
                "seed": settings.seed,
                "canny_low_threshold": method_settings.canny_low_threshold,
                "canny_high_threshold": method_settings.canny_high_threshold,
                "canny_blur_radius": method_settings.canny_blur_radius,
                "canny_ink_threshold": method_settings.canny_ink_threshold,
                "canny_boundary_fill_radius": method_settings.canny_boundary_fill_radius,
                "canny_mask_guard_pixels": method_settings.canny_mask_guard_pixels,
                "latent_source_strategy": latent_source_strategy,
                "latent_source_fill_radius": latent_source_fill_radius,
                "lanpaint_inner_steps": method_settings.lanpaint_inner_steps,
                "lanpaint_friction": method_settings.lanpaint_friction,
                "lanpaint_lambda": method_settings.lanpaint_lambda,
                "lanpaint_beta": method_settings.lanpaint_beta,
                "lanpaint_step_size": method_settings.lanpaint_step_size,
                "lanpaint_final_outer_steps_without_inner": (
                    method_settings.lanpaint_final_outer_steps_without_inner
                ),
                "reencode_source_latent_interval": (
                    method_settings.reencode_source_latent_interval
                ),
            },
        )
        step_trace: list[dict[str, Any]] = []
        reencode_seconds_total = 0.0
        with torch.inference_mode():
            for index, (timestep, flow_t) in enumerate(
                zip(active_timesteps, active_flow_times, strict=True)
            ):
                flow_value = float(flow_t.item())
                inner_steps = (
                    0
                    if effective_steps - index
                    <= method_settings.lanpaint_final_outer_steps_without_inner
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
                    latent_mask=keep_latent.to(
                        device=adapter.device,
                        dtype=latents.dtype,
                    ),
                    current_times=current_times,
                    model_options={},
                    seed=settings.seed or 0,
                    n_steps=inner_steps,
                )
                noise_pred = (
                    (latents - x0.to(latents.dtype)) / max(flow_value, 1e-6)
                ).to(latents.dtype)
                latents = adapter.pipe.scheduler.step(
                    noise_pred,
                    timestep,
                    latents,
                    return_dict=False,
                )[0].to(source_latent.dtype)
                reencode_step_seconds = 0.0
                interval = method_settings.reencode_source_latent_interval
                if interval > 0 and (index + 1) % interval == 0:
                    reencode_started = time.perf_counter()
                    source_latent = adapter.reencode_source_from_prediction(x0, image, mask)
                    reencode_step_seconds = time.perf_counter() - reencode_started
                    reencode_seconds_total += reencode_step_seconds

                if index < effective_steps - 1:
                    next_flow = active_flow_times[index + 1 : index + 2].reshape(1, 1, 1).to(
                        device=adapter.device,
                        dtype=source_latent.dtype,
                    )
                    keep_reference = model.noise_scaling(next_flow, noise, source_latent).to(
                        source_latent.dtype
                    )
                else:
                    keep_reference = source_latent
                edit = edit_latent.to(device=adapter.device, dtype=latents.dtype)
                latents = reinject_keep_latents(latents, keep_reference, edit)
                step_record = {
                    "step_index": index,
                    "timestep": _json_timestep(timestep),
                    "flow_t": flow_value,
                    "lanpaint_inner_steps_override": inner_steps,
                    "reencode_seconds": reencode_step_seconds,
                }
                step_trace.append(step_record)
                reporter.emit(
                    "inference_step",
                    stage="inference",
                    message=f"Inference step {index + 1}/{effective_steps}.",
                    progress={"current": index + 1, "total": effective_steps},
                    metadata=step_record,
                )
        raw = adapter.decode_latents(latents)
        inference_seconds = time.perf_counter() - inference_started
        gpu_memory = {
            **memory_before,
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
            message="FLUX-Canny LanPaint inference completed.",
            progress={"current": effective_steps, "total": effective_steps},
            metadata={
                "inference_seconds": inference_seconds,
                "schedule_debug": schedule_debug,
                "gpu_memory": gpu_memory,
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
        outside_changed = outside_mask_changed(image, composite, mask)
        reporter.emit(
            "hard_composite_done",
            stage="output",
            message="Hard composite completed.",
            metadata={"outside_mask_changed": outside_changed},
        )
        if outside_changed:
            raise RuntimeError("Hard composite changed pixels outside the edit mask.")
        reporter.emit(
            "output_encode_start",
            stage="output",
            message="Encoding output image.",
            metadata={"output_format": settings.output_format},
        )
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
        return {
            "image_base64": encoded,
            "output_format": settings.output_format,
            "width": composite.width,
            "height": composite.height,
            "mask_convention": "white = inpaint/edit, black = preserve",
            "timings": {
                **loaded.timings,
                "canny_control_seconds": control_seconds,
                "inference_seconds": inference_seconds,
                "reencode_source_latent_seconds": reencode_seconds_total,
            },
            "gpu_memory": gpu_memory,
            "model": loaded.model,
            "lora": {
                **loaded.lora,
                "rank": 64,
                "checkpoint": "final",
                "requested_scale": settings.lora_scale,
                "effective_scale": scale["effective_scale"],
                "request_strength_mode": scale["mode"],
            },
            "inference_settings": {
                "method": settings.method,
                "backend_revision": backend_revision,
                "prompt": settings.prompt,
                "negative_prompt": settings.negative_prompt,
                "negative_prompt_passed_to_pipeline": False,
                "partial_noise": settings.partial_noise,
                "guidance_scale": settings.guidance_scale,
                "num_inference_steps": settings.num_inference_steps,
                "effective_num_inference_steps": effective_steps,
                "max_sequence_length": settings.max_sequence_length,
                "seed": settings.seed,
                "mask_coverage": mask_coverage(mask),
                "control_source": control_source_name,
                "control_strategy": control_strategy,
                "canny_low_threshold": method_settings.canny_low_threshold,
                "canny_high_threshold": method_settings.canny_high_threshold,
                "canny_blur_radius": method_settings.canny_blur_radius,
                "canny_ink_threshold": method_settings.canny_ink_threshold,
                "canny_boundary_fill_radius": method_settings.canny_boundary_fill_radius,
                "canny_mask_guard_pixels": method_settings.canny_mask_guard_pixels,
                "latent_source_strategy": latent_source_strategy,
                "latent_source_fill_radius": latent_source_fill_radius,
                "lanpaint_inner_steps": method_settings.lanpaint_inner_steps,
                "lanpaint_friction": method_settings.lanpaint_friction,
                "lanpaint_lambda": method_settings.lanpaint_lambda,
                "lanpaint_beta": method_settings.lanpaint_beta,
                "lanpaint_step_size": method_settings.lanpaint_step_size,
                "lanpaint_final_outer_steps_without_inner": (
                    method_settings.lanpaint_final_outer_steps_without_inner
                ),
                "reinject_keep_latents": True,
                "use_transformer_cache_context": True,
            },
            "schedule_debug": schedule_debug,
            "latent_init_debug": {
                "source_latents": list(source_latent.shape),
                "control_latents": list(control_latent.shape),
                "mask_keep_latents": list(keep_latent.shape),
                "mask_edit_latents": list(edit_latent.shape),
            },
            "outside_mask_changed_after_hard_composite": outside_changed,
        }


__all__ = [
    "CANNY_HIGH_THRESHOLD",
    "CANNY_LANPAINT_BACKEND_REVISION",
    "CANNY_LANPAINT_NATIVE_BACKEND_REVISION",
    "CANNY_LOW_THRESHOLD",
    "CANNY_MODEL_ID",
    "CannyLanPaintRequestSettings",
    "FluxCannyLanPaintAdapter",
    "FluxCannyLanPaintModelWrapper",
    "FluxCannyLanPaintService",
    "load_control_image",
    "make_canny_control",
    "make_lanpaint_source_image",
    "make_masked_canny_control",
    "mask_edit_to_lanpaint_keep",
    "parse_canny_lanpaint_settings",
    "reinject_keep_latents",
]
