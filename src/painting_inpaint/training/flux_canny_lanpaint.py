"""LanPaint validation for FLUX.1-Canny training runs.

Diffusers loads the full ``FluxControlPipeline`` components for the Canny model,
but the pipeline call is not used. LanPaint owns the masked denoising loop.
"""

from __future__ import annotations

import gc
import inspect
import os
import re
import time
from contextlib import nullcontext
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from painting_inpaint.compositing import hard_composite, outside_mask_changed
from painting_inpaint.controls import make_canny_control
from painting_inpaint.experiments.manifest import package_versions, write_json
from painting_inpaint.masks import (
    allow_large_images,
    binarize_mask,
    make_mask_overlay,
    mask_coverage,
)
from painting_inpaint.paths import resolve_data_path
from painting_inpaint.tiling import crop_tile, tile_from_top_left_fraction
from painting_inpaint.visualization import (
    draw_windows_overview,
    make_comparison_sheet,
    resize_to_long_side,
    save_preview,
)

VALIDATION_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}


def clear_cuda_cache() -> None:
    """Release Python and CUDA cache memory when Torch is available."""

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def safe_signature(obj: Any) -> str:
    """Return a printable signature for metadata/debug reports."""

    try:
        return str(inspect.signature(obj))
    except Exception as exc:
        return f"<uninspectable: {exc!r}>"


def scalar_float(value: Any) -> float | None:
    """Convert scalar Python/Torch values to a JSON-safe float."""

    if value is None:
        return None
    try:
        import torch

        if torch.is_tensor(value):
            value = value.detach().float().cpu()
            if value.numel() != 1:
                return None
            return float(value.item())
    except Exception:
        pass
    return float(value)


def _version_tuple(value: str) -> tuple[int, int, int]:
    parts = [int(part) for part in re.findall(r"\d+", value)[:3]]
    return tuple((parts + [0, 0, 0])[:3])


def torchao_lora_compatibility_error(torchao_version: str | None = None) -> str | None:
    """Return a clear LoRA-loading error when an old optional TorchAO is installed."""

    if torchao_version is None:
        try:
            torchao_version = version("torchao")
        except PackageNotFoundError:
            return None

    if _version_tuple(torchao_version) >= (0, 16, 0):
        return None
    return (
        f"Found incompatible optional torchao {torchao_version}. PEFT LoRA loading "
        "fails when torchao is installed below 0.16.0. This validation workflow does "
        "not use torchao quantization, so remove it from the runtime. In Colab, run "
        "`%pip uninstall -y torchao`, then restart/rerun the notebook setup cells."
    )


def mask_edit_to_lanpaint_keep(mask_edit: Image.Image) -> Image.Image:
    """Convert project mask semantics to LanPaint mask semantics."""

    edit = binarize_mask(mask_edit)
    return Image.eval(edit, lambda p: 255 - p)


def make_current_times(flow_t: float, device: Any, dtype: Any = None) -> tuple[Any, Any, Any]:
    """Return the LanPaint current-time tuple for a FLUX flow value."""

    import torch

    if dtype is None:
        dtype = torch.float32
    flow = torch.tensor([flow_t], device=device, dtype=dtype)
    abt = (1.0 - flow) ** 2 / ((1.0 - flow) ** 2 + flow**2)
    ve_sigma = flow / torch.clamp(1.0 - flow, min=1e-6)
    return ve_sigma, abt, flow


def lanpaint_strength_schedule_estimate(
    num_inference_steps: int,
    strength: float,
) -> dict[str, Any]:
    """Estimate how partial-noise strength slices the scheduler."""

    schedule_length = int(num_inference_steps)
    start_index = min(schedule_length - 1, int((1.0 - float(strength)) * schedule_length))
    return {
        "source": "notebook-local LanPaint schedule slicing",
        "strength": float(strength),
        "requested_num_inference_steps": schedule_length,
        "estimated_start_index": int(start_index),
        "estimated_effective_num_steps": int(schedule_length - start_index),
        "estimated_skipped_num_steps": int(start_index),
        "interpretation": (
            "higher strength starts earlier in the noise schedule; strength=1.0 uses "
            "the full schedule"
        ),
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


def _resolve_required_data_path(path: str | None, label: str) -> Path:
    resolved = resolve_data_path(path)
    if resolved is None:
        raise ValueError(f"validation.{label} is required.")
    resolved = resolved.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Missing validation {label}: {resolved}")
    return resolved


def _list_validation_images(root: str | Path) -> list[Path]:
    folder = Path(root)
    if not folder.exists():
        raise FileNotFoundError(f"Missing validation dataset directory: {folder}")
    if not folder.is_dir():
        raise NotADirectoryError(f"Validation dataset path is not a directory: {folder}")
    return sorted(
        (
            path
            for path in folder.rglob("*")
            if path.is_file() and path.suffix.lower() in VALIDATION_IMAGE_EXTENSIONS
        ),
        key=lambda path: str(path.relative_to(folder)).lower(),
    )


def _resolve_validation_dataset_dir(cfg: Any) -> Path:
    source_dir = cfg.validation_dataset_source_dir or cfg.source_dir
    resolved = resolve_data_path(source_dir)
    if resolved is None:
        raise ValueError("validation.dataset_source_dir or dataset.source_dir is required.")
    resolved = resolved.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Missing validation dataset directory: {resolved}")
    return resolved


def _resize_min_side(image: Image.Image, min_side: int) -> Image.Image:
    width, height = image.size
    short_side = min(width, height)
    if short_side >= min_side:
        return image
    scale = min_side / short_side
    return image.resize((round(width * scale), round(height * scale)), Image.Resampling.LANCZOS)


def _synthetic_validation_mask(
    size: tuple[int, int],
    *,
    mask_kind: str,
    mask_fraction: float,
    x_fraction: float,
    y_fraction: float,
) -> Image.Image:
    if mask_kind != "center_square":
        raise ValueError(f"Unsupported validation mask kind: {mask_kind}")
    width, height = size
    side = max(1, min(width, height, round(min(width, height) * float(mask_fraction))))
    left = round(max(0.0, min(1.0, float(x_fraction))) * (width - side))
    top = round(max(0.0, min(1.0, float(y_fraction))) * (height - side))
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    draw.rectangle((left, top, left + side - 1, top + side - 1), fill=255)
    return mask


class FluxCannyLanPaintAdapter:
    """Adapter exposing FLUX.1-Canny internals to official LanPaint."""

    def __init__(
        self,
        pipe: Any,
        *,
        use_transformer_cache_context: bool = True,
    ) -> None:
        self.pipe = pipe
        self.use_transformer_cache_context = use_transformer_cache_context
        self.prompt_embeds = None
        self.pooled_prompt_embeds = None
        self.text_ids = None
        self.y_latent = None
        self.control_latent = None
        self.latent_image_ids = None
        self.height = None
        self.width = None
        self.last_schedule_report = None

    @property
    def device(self) -> Any:
        return self.pipe._execution_device

    @property
    def dtype(self) -> Any:
        return self.pipe.transformer.dtype

    @property
    def scheduler(self) -> Any:
        return self.pipe.scheduler

    def encode_prompt(
        self,
        prompt: str,
        device: Any,
        *,
        max_sequence_length: int = 512,
    ) -> None:
        encode_kwargs = {
            "prompt": prompt,
            "prompt_2": None,
            "device": device,
            "num_images_per_prompt": 1,
            "max_sequence_length": max_sequence_length,
        }
        if "lora_scale" in inspect.signature(self.pipe.encode_prompt).parameters:
            encode_kwargs["lora_scale"] = None
        self.prompt_embeds, self.pooled_prompt_embeds, self.text_ids = self.pipe.encode_prompt(
            **encode_kwargs
        )

    def encode_image_latents(
        self,
        image: Image.Image,
        height: int,
        width: int,
        generator: Any,
        device: Any,
    ) -> Any:
        self.height = int(height)
        self.width = int(width)
        image_tensor = self.pipe.image_processor.preprocess(
            image.convert("RGB"),
            height=self.height,
            width=self.width,
        )
        image_tensor = image_tensor.to(device=device, dtype=self.pipe.vae.dtype)

        import torch

        with torch.inference_mode():
            encoded = self.pipe.vae.encode(image_tensor).latent_dist.sample(generator=generator)
        encoded = (encoded - self.pipe.vae.config.shift_factor) * (
            self.pipe.vae.config.scaling_factor
        )

        batch_size = encoded.shape[0]
        num_channels = encoded.shape[1]
        latent_height = 2 * (self.height // (self.pipe.vae_scale_factor * 2))
        latent_width = 2 * (self.width // (self.pipe.vae_scale_factor * 2))
        self.y_latent = self.pipe._pack_latents(
            encoded.to(self.dtype),
            batch_size,
            num_channels,
            latent_height,
            latent_width,
        )
        self.latent_image_ids = self.pipe._prepare_latent_image_ids(
            batch_size,
            latent_height // 2,
            latent_width // 2,
            device,
            self.prompt_embeds.dtype,
        )
        return self.y_latent

    def encode_control_latents(
        self,
        control_image: Image.Image,
        generator: Any,
        device: Any,
    ) -> Any:
        if self.height is None or self.width is None:
            raise RuntimeError("Encode image latents before control latents.")
        control_tensor = self.pipe.prepare_image(
            image=control_image.convert("RGB"),
            width=self.width,
            height=self.height,
            batch_size=1,
            num_images_per_prompt=1,
            device=device,
            dtype=self.pipe.vae.dtype,
        )

        import torch

        with torch.inference_mode():
            encoded = self.pipe.vae.encode(control_tensor).latent_dist.sample(generator=generator)
        encoded = (encoded - self.pipe.vae.config.shift_factor) * (
            self.pipe.vae.config.scaling_factor
        )

        batch_size = encoded.shape[0]
        num_channels = encoded.shape[1]
        latent_height, latent_width = encoded.shape[2:]
        self.control_latent = self.pipe._pack_latents(
            encoded.to(self.dtype),
            batch_size,
            num_channels,
            latent_height,
            latent_width,
        )
        if self.y_latent is not None and self.control_latent.shape[:2] != self.y_latent.shape[:2]:
            raise RuntimeError(
                "Packed control latent shape does not match source latents: "
                f"control={self.control_latent.shape}, source={self.y_latent.shape}"
            )
        return self.control_latent

    def mask_keep_to_latents(self, mask_keep_image: Image.Image) -> tuple[Any, Any]:
        if self.y_latent is None:
            raise RuntimeError("Encode image latents before mask conversion.")

        import torch

        unpacked = self.pipe._unpack_latents(
            self.y_latent,
            self.height,
            self.width,
            self.pipe.vae_scale_factor,
        )
        latent_h, latent_w = unpacked.shape[-2:]
        mask_arr = np.asarray(mask_keep_image.convert("L"), dtype=np.float32) / 255.0
        mask_tensor = torch.from_numpy(mask_arr)[None, None].to(self.device, dtype=torch.float32)
        mask_tensor = torch.nn.functional.interpolate(
            mask_tensor,
            size=(latent_h, latent_w),
            mode="nearest",
        )

        edit_tensor = 1.0 - mask_tensor
        packed_edit = torch.nn.functional.max_pool2d(edit_tensor, kernel_size=2, stride=2)
        packed_keep = 1.0 - packed_edit
        packed_keep = packed_keep.flatten(2).transpose(1, 2).contiguous()
        packed_edit = packed_edit.flatten(2).transpose(1, 2).contiguous()
        if packed_keep.shape[1] != self.y_latent.shape[1]:
            raise RuntimeError(
                "Packed mask sequence length does not match Flux latents: "
                f"mask={packed_keep.shape}, latents={self.y_latent.shape}"
            )
        return packed_keep, packed_edit

    def prepare_timesteps(
        self,
        num_steps: int,
        device: Any,
        *,
        schedule_mode: str = "diffusers_default",
    ) -> tuple[Any, Any]:
        from diffusers.pipelines.flux.pipeline_flux import calculate_shift, retrieve_timesteps

        normalized_schedule_mode = str(schedule_mode).strip().lower()
        if normalized_schedule_mode not in {"diffusers_default", "linear_sigmas_approx"}:
            raise ValueError(
                "schedule_mode must be one of: diffusers_default, linear_sigmas_approx; "
                f"got {schedule_mode!r}"
            )

        linear_sigmas = np.linspace(1.0, 1.0 / num_steps, num_steps)
        sigmas = linear_sigmas
        used_scheduler_default_flow_sigmas = False
        if normalized_schedule_mode == "diffusers_default":
            if (
                hasattr(self.scheduler.config, "use_flow_sigmas")
                and self.scheduler.config.use_flow_sigmas
            ):
                sigmas = None
                used_scheduler_default_flow_sigmas = True
        image_seq_len = self.y_latent.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, _ = retrieve_timesteps(
            self.scheduler,
            num_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )
        flow_ts = self.scheduler.sigmas.to(device)[:-1]
        self.last_schedule_report = {
            "schedule_mode": normalized_schedule_mode,
            "exact_comfy_simple_equivalence": False,
            "note": (
                "Uses explicitly supplied linear sigmas as an approximation to the "
                "Comfy euler/simple reference."
                if normalized_schedule_mode == "linear_sigmas_approx"
                else "Uses Diffusers FLUX retrieve_timesteps behavior for the loaded scheduler."
            ),
            "num_requested_steps": int(num_steps),
            "used_scheduler_default_flow_sigmas": used_scheduler_default_flow_sigmas,
            "provided_sigmas": None if sigmas is None else [float(value) for value in sigmas],
            "mu": float(mu),
            "scheduler_class": type(self.scheduler).__name__,
            "scheduler_config": dict(getattr(self.scheduler, "config", {}) or {}),
        }
        return timesteps[: len(flow_ts)], flow_ts

    def predict_x0(self, x: Any, flow_t: float, guidance_scale: float) -> tuple[Any, Any]:
        if self.control_latent is None:
            raise RuntimeError("Encode Canny control latents before denoising.")

        import torch

        model_dtype = self.dtype
        timestep = torch.full((x.shape[0],), flow_t, device=x.device, dtype=model_dtype)
        if self.pipe.transformer.config.guidance_embeds:
            guidance = torch.full(
                (x.shape[0],),
                guidance_scale,
                device=x.device,
                dtype=torch.float32,
            )
        else:
            guidance = None

        latent_model_input = torch.cat(
            [x, self.control_latent.to(device=x.device, dtype=x.dtype)],
            dim=2,
        )
        cache_context = getattr(self.pipe.transformer, "cache_context", None)
        transformer_context = (
            cache_context("cond")
            if self.use_transformer_cache_context and cache_context is not None
            else nullcontext()
        )
        with transformer_context:
            noise_pred = self.pipe.transformer(
                hidden_states=latent_model_input.to(model_dtype),
                timestep=timestep,
                guidance=guidance,
                pooled_projections=self.pooled_prompt_embeds,
                encoder_hidden_states=self.prompt_embeds,
                txt_ids=self.text_ids,
                img_ids=self.latent_image_ids,
                joint_attention_kwargs={},
                return_dict=False,
            )[0]

        flow = torch.as_tensor(flow_t, device=x.device, dtype=noise_pred.dtype)
        x0 = x.to(noise_pred.dtype) - flow * noise_pred
        x0 = x0.to(x.dtype)
        return x0, x0

    def decode_latents(self, latents: Any) -> Image.Image:
        import torch

        latents = latents.detach()
        with torch.no_grad():
            latents = self.pipe._unpack_latents(
                latents,
                self.height,
                self.width,
                self.pipe.vae_scale_factor,
            )
            latents = (latents / self.pipe.vae.config.scaling_factor) + (
                self.pipe.vae.config.shift_factor
            )
            image = self.pipe.vae.decode(latents.to(self.pipe.vae.dtype), return_dict=False)[0]
        image = image.detach()
        return self.pipe.image_processor.postprocess(image, output_type="pil")[0].convert("RGB")


class FluxCannyLanPaintModelWrapper:
    """Small model wrapper expected by official LanPaint."""

    def __init__(
        self,
        adapter: FluxCannyLanPaintAdapter,
        guidance_scale: float,
        *,
        big_branch_mode: str = "identical",
        big_guidance_scale: float | None = None,
    ) -> None:
        if big_branch_mode not in {"identical", "approx_cfg_big_1"}:
            raise ValueError(
                "big_branch_mode must be one of: identical, approx_cfg_big_1; "
                f"got {big_branch_mode!r}"
            )
        self.adapter = adapter
        self.guidance_scale = float(guidance_scale)
        self.big_branch_mode = big_branch_mode
        self.big_guidance_scale = 1.0 if big_guidance_scale is None else float(big_guidance_scale)
        self.inner_model = self
        self.model_sampling = self
        self.branch_diff_records: list[dict[str, Any]] = []

    def noise_scaling(self, sigma: Any, noise: Any, latent_image: Any) -> Any:
        return (1.0 - sigma) * latent_image + sigma * noise

    def _record_branch_difference(self, flow_t: float, x0: Any, x0_big: Any) -> None:
        import torch

        with torch.no_grad():
            diff = (x0.detach().float() - x0_big.detach().float()).flatten()
            self.branch_diff_records.append(
                {
                    "flow_t": float(flow_t),
                    "l2": float(torch.linalg.vector_norm(diff).item()),
                    "mae": float(diff.abs().mean().item()),
                    "max_abs": float(diff.abs().max().item()),
                }
            )

    def branch_summary(self) -> dict[str, Any]:
        """Return JSON-safe diagnostics for the LanPaint BiG branch."""

        if not self.branch_diff_records:
            return {
                "big_branch_mode": self.big_branch_mode,
                "flux_guidance_main": self.guidance_scale,
                "flux_guidance_big": self.big_guidance_scale
                if self.big_branch_mode == "approx_cfg_big_1"
                else self.guidance_scale,
                "call_count": 0,
                "outputs_identical_all": None,
                "cfg_big_exact_comfy_equivalence": False,
                "records": [],
            }

        l2_values = [record["l2"] for record in self.branch_diff_records]
        mae_values = [record["mae"] for record in self.branch_diff_records]
        max_abs_values = [record["max_abs"] for record in self.branch_diff_records]
        return {
            "big_branch_mode": self.big_branch_mode,
            "flux_guidance_main": self.guidance_scale,
            "flux_guidance_big": self.big_guidance_scale
            if self.big_branch_mode == "approx_cfg_big_1"
            else self.guidance_scale,
            "call_count": len(self.branch_diff_records),
            "outputs_identical_all": all(value <= 1e-8 for value in max_abs_values),
            "l2_mean": float(np.mean(l2_values)),
            "l2_max": float(np.max(l2_values)),
            "mae_mean": float(np.mean(mae_values)),
            "mae_max": float(np.max(mae_values)),
            "max_abs": float(np.max(max_abs_values)),
            "cfg_big_exact_comfy_equivalence": False,
            "equivalence_note": (
                "approx_cfg_big_1 uses FLUX guidance=1.0 for the BiG branch; this is "
                "an approximation of scraed/LanPaint cfg_BIG=1.0, not a proven exact "
                "ComfyUI CFG implementation."
                if self.big_branch_mode == "approx_cfg_big_1"
                else "identical mode is the previous adapter behavior and collapses "
                "LanPaint's normal and BiG predictions."
            ),
            "records": self.branch_diff_records[:20],
        }

    def __call__(
        self,
        x: Any,
        t: Any,
        model_options: dict[str, Any] | None = None,
        seed: int | None = None,
    ) -> tuple[Any, Any]:
        del model_options, seed
        flow_t = float(t.flatten()[0])
        x0, _ = self.adapter.predict_x0(x, flow_t, self.guidance_scale)
        if self.big_branch_mode == "identical":
            x0_big = x0
        else:
            x0_big, _ = self.adapter.predict_x0(x, flow_t, self.big_guidance_scale)
        self._record_branch_difference(flow_t, x0, x0_big)
        return x0, x0_big


def _prepare_configured_file_validation_window(
    cfg: Any,
    validation_dir: Path,
    window: dict[str, Any],
) -> dict[str, Any]:
    """Prepare a validation window from configured base/mask/prior files."""

    window_id = str(window["window_id"])
    window_size = int(window["window_size"])
    x_fraction = float(window["x_fraction"])
    y_fraction = float(window["y_fraction"])

    base_path = _resolve_required_data_path(cfg.validation_base_image_path, "base_image_path")
    mask_path = _resolve_required_data_path(cfg.validation_mask_image_path, "mask_image_path")
    prior_path = resolve_data_path(cfg.validation_conditioning_prior_path)
    prior_path = prior_path.resolve() if prior_path is not None and prior_path.exists() else None

    with allow_large_images():
        base = Image.open(base_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        prior = Image.open(prior_path).convert("RGB") if prior_path is not None else None

    if base.size != mask.size:
        raise ValueError(f"Validation base/mask sizes differ: base={base.size}, mask={mask.size}")
    if prior is not None and prior.size != base.size:
        raise ValueError(f"Validation prior size differs: prior={prior.size}, base={base.size}")

    base_work = resize_to_long_side(base, cfg.validation_target_long_side)
    mask_work = resize_to_long_side(mask, cfg.validation_target_long_side, Image.Resampling.NEAREST)
    prior_work = (
        resize_to_long_side(prior, cfg.validation_target_long_side)
        if prior is not None
        else None
    )
    if base_work.size != mask_work.size:
        raise ValueError(
            f"Validation downscaled base/mask sizes differ: base={base_work.size}, "
            f"mask={mask_work.size}"
        )

    save_preview(base_work, validation_dir / "base_image_5000_preview.png")
    base_work.save(validation_dir / "base_image_5000.png")
    mask_work.save(validation_dir / "mask_5000.png")
    if prior_work is not None:
        prior_work.save(validation_dir / "conditioning_prior_image_5000.png")

    tile = tile_from_top_left_fraction(
        base_work.size,
        window_size,
        x_fraction=x_fraction,
        y_fraction=y_fraction,
    )
    mask_crop = crop_tile(mask_work, tile)
    input_crop = crop_tile(base_work, tile)
    if cfg.validation_control_source == "conditioning_prior" and prior_work is not None:
        control_source = crop_tile(prior_work, tile)
        resolved_control_source = "conditioning_prior"
        control_source_path = prior_path
    else:
        control_source = input_crop
        resolved_control_source = "base_image"
        control_source_path = base_path

    window_record = {
        "window_id": window_id,
        "x": tile.x,
        "y": tile.y,
        "width": tile.width,
        "height": tile.height,
        "x_fraction": x_fraction,
        "y_fraction": y_fraction,
        "mask_fraction": mask_coverage(mask_crop),
    }
    overview_records = []
    for overview_window in cfg.validation_windows:
        overview_tile = tile_from_top_left_fraction(
            base_work.size,
            int(overview_window["window_size"]),
            x_fraction=float(overview_window["x_fraction"]),
            y_fraction=float(overview_window["y_fraction"]),
        )
        overview_records.append(
            {
                "window_id": str(overview_window["window_id"]),
                "x": overview_tile.x,
                "y": overview_tile.y,
                "width": overview_tile.width,
                "height": overview_tile.height,
                "x_fraction": float(overview_window["x_fraction"]),
                "y_fraction": float(overview_window["y_fraction"]),
                "mask_fraction": mask_coverage(crop_tile(mask_work, overview_tile)),
            }
        )
    draw_windows_overview(
        base_work,
        overview_records,
        validation_dir / "window_overview.png",
        max_side=1800,
    )

    window_dir = validation_dir / window_id
    window_dir.mkdir(parents=True, exist_ok=True)
    canny_control = make_canny_control(
        control_source,
        low_threshold=cfg.canny_low_threshold,
        high_threshold=cfg.canny_high_threshold,
        blur_radius=cfg.validation_canny_blur_radius,
    )

    input_path = window_dir / "input.png"
    mask_out_path = window_dir / "mask.png"
    control_source_out_path = window_dir / "control_source.png"
    canny_control_path = window_dir / "canny_control.png"
    control_preview_path = window_dir / "control_preview.png"
    prompt_path = window_dir / "prompt.txt"

    input_crop.save(input_path)
    mask_crop.save(mask_out_path)
    control_source.save(control_source_out_path)
    canny_control.save(canny_control_path)
    make_comparison_sheet(
        [control_source, canny_control],
        ["control source", "canny control"],
        output_path=control_preview_path,
        max_panel_side=512,
        label_height=28,
    )
    prompt_path.write_text(cfg.validation_prompt + "\n", encoding="utf-8")

    return {
        "base_path": base_path,
        "mask_path": mask_path,
        "prior_path": prior_path,
        "control_source_path": control_source_path,
        "resolved_control_source": resolved_control_source,
        "window_record": window_record,
        "window_size": window_size,
        "window_dir": window_dir,
        "input_crop": input_crop,
        "mask_crop": mask_crop,
        "canny_control": canny_control,
        "paths": {
            "input": input_path,
            "mask": mask_out_path,
            "control_source": control_source_out_path,
            "canny_control": canny_control_path,
            "control_preview": control_preview_path,
            "prompt": prompt_path,
        },
    }


def _prepare_dataset_validation_window(
    cfg: Any,
    validation_dir: Path,
    window: dict[str, Any],
) -> dict[str, Any]:
    """Prepare a fixed validation crop from the configured training dataset."""

    window_id = str(window["window_id"])
    window_size = int(window["window_size"])
    x_fraction = float(window["x_fraction"])
    y_fraction = float(window["y_fraction"])
    mask_fraction = float(window.get("mask_fraction", cfg.validation_mask_fraction))
    mask_x_fraction = float(window.get("mask_x_fraction", cfg.validation_mask_x_fraction))
    mask_y_fraction = float(window.get("mask_y_fraction", cfg.validation_mask_y_fraction))

    dataset_dir = _resolve_validation_dataset_dir(cfg)
    image_paths = _list_validation_images(dataset_dir)
    if not image_paths:
        raise FileNotFoundError(f"No validation images found under {dataset_dir}")

    if window.get("image_path"):
        relative_image_path = Path(str(window["image_path"]))
        image_path = (
            relative_image_path
            if relative_image_path.is_absolute()
            else dataset_dir / relative_image_path
        ).resolve()
        if not image_path.exists():
            raise FileNotFoundError(f"Missing validation image_path: {image_path}")
        image_index = None
    else:
        image_index = int(window.get("image_index", 0))
        if image_index < 0 or image_index >= len(image_paths):
            raise IndexError(
                f"validation image_index {image_index} is out of range for "
                f"{len(image_paths)} image(s) in {dataset_dir}"
            )
        image_path = image_paths[image_index].resolve()

    with allow_large_images():
        base = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")

    base_work = _resize_min_side(base, window_size)
    tile = tile_from_top_left_fraction(
        base_work.size,
        window_size,
        x_fraction=x_fraction,
        y_fraction=y_fraction,
    )
    input_crop = crop_tile(base_work, tile)
    mask_crop = _synthetic_validation_mask(
        input_crop.size,
        mask_kind=cfg.validation_mask_kind,
        mask_fraction=mask_fraction,
        x_fraction=mask_x_fraction,
        y_fraction=mask_y_fraction,
    )
    control_source = input_crop
    resolved_control_source = "base_image"
    control_source_path = image_path

    window_record = {
        "window_id": window_id,
        "source": "dataset",
        "dataset_dir": str(dataset_dir),
        "image_index": image_index,
        "image_path": str(image_path),
        "original_width": base.size[0],
        "original_height": base.size[1],
        "resized_width": base_work.size[0],
        "resized_height": base_work.size[1],
        "x": tile.x,
        "y": tile.y,
        "width": tile.width,
        "height": tile.height,
        "x_fraction": x_fraction,
        "y_fraction": y_fraction,
        "mask_kind": cfg.validation_mask_kind,
        "mask_fraction": mask_coverage(mask_crop),
        "requested_mask_fraction": mask_fraction,
        "mask_x_fraction": mask_x_fraction,
        "mask_y_fraction": mask_y_fraction,
    }

    window_dir = validation_dir / window_id
    window_dir.mkdir(parents=True, exist_ok=True)
    canny_control = make_canny_control(
        control_source,
        low_threshold=cfg.canny_low_threshold,
        high_threshold=cfg.canny_high_threshold,
        blur_radius=cfg.validation_canny_blur_radius,
    )

    input_path = window_dir / "input.png"
    mask_out_path = window_dir / "mask.png"
    control_source_out_path = window_dir / "control_source.png"
    canny_control_path = window_dir / "canny_control.png"
    control_preview_path = window_dir / "control_preview.png"
    source_preview_path = window_dir / "source_image_preview.png"
    prompt_path = window_dir / "prompt.txt"

    input_crop.save(input_path)
    mask_crop.save(mask_out_path)
    control_source.save(control_source_out_path)
    canny_control.save(canny_control_path)
    save_preview(base_work, source_preview_path)
    make_comparison_sheet(
        [control_source, make_mask_overlay(control_source, mask_crop), canny_control],
        ["validation crop", "synthetic edit mask", "canny control"],
        output_path=control_preview_path,
        max_panel_side=512,
        label_height=28,
    )
    prompt_path.write_text(cfg.validation_prompt + "\n", encoding="utf-8")

    return {
        "base_path": image_path,
        "mask_path": None,
        "prior_path": None,
        "control_source_path": control_source_path,
        "resolved_control_source": resolved_control_source,
        "window_record": window_record,
        "window_size": window_size,
        "window_dir": window_dir,
        "input_crop": input_crop,
        "mask_crop": mask_crop,
        "canny_control": canny_control,
        "paths": {
            "input": input_path,
            "mask": mask_out_path,
            "control_source": control_source_out_path,
            "canny_control": canny_control_path,
            "control_preview": control_preview_path,
            "source_preview": source_preview_path,
            "prompt": prompt_path,
        },
    }


def _prepare_validation_window(
    cfg: Any,
    validation_dir: Path,
    window: dict[str, Any],
) -> dict[str, Any]:
    """Prepare the configured LanPaint validation inputs and Canny control."""

    if cfg.validation_source == "dataset":
        return _prepare_dataset_validation_window(cfg, validation_dir, window)
    return _prepare_configured_file_validation_window(cfg, validation_dir, window)


def _write_api_report(pipe: Any, cfg: Any, validation_dir: Path, *, lora_enabled: bool) -> None:
    """Validate required Diffusers internals before LanPaint denoising."""

    try:
        from diffusers.pipelines.flux.pipeline_flux import calculate_shift, retrieve_timesteps

        del calculate_shift, retrieve_timesteps
        flux_helpers_available = True
        flux_helpers_error = None
    except Exception as exc:
        flux_helpers_available = False
        flux_helpers_error = repr(exc)

    required_attrs = [
        "encode_prompt",
        "prepare_image",
        "_pack_latents",
        "_unpack_latents",
        "_prepare_latent_image_ids",
        "vae",
        "transformer",
        "scheduler",
        "image_processor",
    ]
    missing_attrs = [name for name in required_attrs if not hasattr(pipe, name)]
    vae_latent_channels = getattr(getattr(pipe.vae, "config", None), "latent_channels", None)
    packed_latent_channels = (
        int(vae_latent_channels) * 4 if vae_latent_channels is not None else None
    )
    transformer_in_channels = getattr(
        getattr(pipe.transformer, "config", None),
        "in_channels",
        None,
    )
    expected_transformer_in_channels = (
        packed_latent_channels * 2 if packed_latent_channels is not None else None
    )
    transformer_shape_ok = transformer_in_channels == expected_transformer_in_channels
    api_report = {
        "model_id": cfg.model_id,
        "validation_backend": "lanpaint",
        "component_loader_class": type(pipe).__name__,
        "pipeline_call_used": False,
        "lanpaint_used_for_denoising": True,
        "lora_enabled": lora_enabled,
        "flux_helpers_available": flux_helpers_available,
        "flux_helpers_error": flux_helpers_error,
        "required_attrs": required_attrs,
        "missing_attrs": missing_attrs,
        "pipe_class": type(pipe).__name__,
        "transformer_class": type(pipe.transformer).__name__
        if hasattr(pipe, "transformer")
        else None,
        "scheduler_class": type(pipe.scheduler).__name__ if hasattr(pipe, "scheduler") else None,
        "vae_class": type(pipe.vae).__name__ if hasattr(pipe, "vae") else None,
        "vae_scale_factor": getattr(pipe, "vae_scale_factor", None),
        "vae_latent_channels": vae_latent_channels,
        "packed_latent_channels": packed_latent_channels,
        "transformer_in_channels": transformer_in_channels,
        "expected_transformer_in_channels": expected_transformer_in_channels,
        "transformer_shape_ok": transformer_shape_ok,
        "transformer_guidance_embeds": getattr(
            getattr(pipe.transformer, "config", None),
            "guidance_embeds",
            None,
        )
        if hasattr(pipe, "transformer")
        else None,
        "transformer_has_cache_context": hasattr(pipe.transformer, "cache_context")
        if hasattr(pipe, "transformer")
        else None,
        "mask_semantics": {
            "project_mask_edit": "white = edit, black = preserve",
            "lanpaint_latent_mask": "1 = keep/known, 0 = edit/unknown",
            "diffusers_pipeline_call": "not used; LanPaint owns masked denoising",
        },
        "signatures": {
            "pipe_call": safe_signature(pipe.__call__),
            "encode_prompt": safe_signature(pipe.encode_prompt)
            if hasattr(pipe, "encode_prompt")
            else None,
            "prepare_image": safe_signature(pipe.prepare_image)
            if hasattr(pipe, "prepare_image")
            else None,
            "scheduler_set_timesteps": safe_signature(pipe.scheduler.set_timesteps)
            if hasattr(pipe, "scheduler")
            else None,
            "scheduler_step": safe_signature(pipe.scheduler.step)
            if hasattr(pipe, "scheduler")
            else None,
            "transformer_forward": safe_signature(pipe.transformer.forward)
            if hasattr(pipe, "transformer")
            else None,
        },
        "decision": "pending",
    }
    if missing_attrs or not flux_helpers_available or not transformer_shape_ok:
        api_report["decision"] = "blocked"
        api_report["blocker"] = (
            "Missing Flux internals or unexpected FLUX.1-Canny-dev transformer channel layout."
        )
        write_json(validation_dir / "api_report.json", _jsonable(api_report))
        raise RuntimeError("Flux.1 Canny LanPaint adapter is blocked. See api_report.json.")

    api_report["decision"] = "api_inspection_passed"
    write_json(validation_dir / "api_report.json", _jsonable(api_report))


def inspect_flux_canny_lanpaint_pipeline(
    pipe: Any,
    cfg: Any,
    validation_dir: str | Path,
    *,
    lora_enabled: bool = False,
) -> None:
    """Write the FLUX-Canny/LanPaint API report for a loaded pipeline."""

    _write_api_report(pipe, cfg, Path(validation_dir), lora_enabled=lora_enabled)


def load_flux_canny_lanpaint_pipeline(cfg: Any) -> tuple[Any, Any]:
    """Load the FLUX-Canny component pipeline once for LanPaint validation."""

    import torch
    from diffusers import FluxControlPipeline
    from huggingface_hub import get_token, login

    if not torch.cuda.is_available():
        raise RuntimeError("LanPaint validation requires a CUDA GPU.")

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN") or get_token()
    if not hf_token:
        raise RuntimeError("Missing HF_TOKEN for gated FLUX.1-Canny-dev validation.")
    login(token=hf_token, add_to_git_credential=False)

    torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    clear_cuda_cache()
    pipe = FluxControlPipeline.from_pretrained(
        cfg.model_id,
        torch_dtype=torch_dtype,
        token=hf_token,
    )

    if hasattr(pipe, "enable_model_cpu_offload"):
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")

    if getattr(pipe, "vae", None) is not None:
        if hasattr(pipe.vae, "enable_tiling"):
            pipe.vae.enable_tiling()
        if hasattr(pipe.vae, "enable_slicing"):
            pipe.vae.enable_slicing()

    return pipe, torch_dtype


def load_flux_canny_lora_adapter(
    pipe: Any,
    lora_dir: str | Path,
    *,
    adapter_name: str,
    adapter_weight: float = 1.0,
) -> dict[str, Any]:
    """Load one LoRA adapter into an already-loaded FLUX-Canny pipeline."""

    torchao_error = torchao_lora_compatibility_error()
    if torchao_error is not None:
        raise RuntimeError(torchao_error)
    if not hasattr(pipe, "load_lora_weights"):
        raise RuntimeError(f"{type(pipe).__name__} does not expose load_lora_weights.")

    lora_path = Path(lora_dir)
    weights_path = lora_path / "pytorch_lora_weights.safetensors"
    if not weights_path.exists():
        raise FileNotFoundError(f"Expected LoRA weights: {weights_path}")

    try:
        pipe.load_lora_weights(str(lora_path), adapter_name=adapter_name)
        named_adapter = True
    except TypeError:
        pipe.load_lora_weights(str(lora_path))
        named_adapter = False

    if named_adapter and hasattr(pipe, "set_adapters"):
        try:
            pipe.set_adapters([adapter_name], adapter_weights=[float(adapter_weight)])
        except TypeError:
            pipe.set_adapters([adapter_name])

    return {
        "adapter_name": adapter_name,
        "adapter_weight": float(adapter_weight),
        "lora_dir": str(lora_path),
        "weights_path": str(weights_path),
        "named_adapter": named_adapter,
        "load_method": "load_lora_weights",
    }


def remove_flux_canny_lora_adapter(
    pipe: Any,
    *,
    adapter_name: str,
    named_adapter: bool,
) -> dict[str, Any]:
    """Remove a LoRA adapter when the installed Diffusers version supports it."""

    if named_adapter and hasattr(pipe, "delete_adapters"):
        try:
            pipe.delete_adapters(adapter_name)
            clear_cuda_cache()
            return {
                "removed": True,
                "method": "delete_adapters",
                "adapter_name": adapter_name,
            }
        except Exception as exc:
            return {
                "removed": False,
                "method": "delete_adapters",
                "adapter_name": adapter_name,
                "error": repr(exc),
            }

    if hasattr(pipe, "unload_lora_weights"):
        try:
            pipe.unload_lora_weights()
            clear_cuda_cache()
            return {
                "removed": True,
                "method": "unload_lora_weights",
                "adapter_name": adapter_name,
            }
        except Exception as exc:
            return {
                "removed": False,
                "method": "unload_lora_weights",
                "adapter_name": adapter_name,
                "error": repr(exc),
            }

    return {
        "removed": False,
        "method": "none_available",
        "adapter_name": adapter_name,
    }


def prepare_flux_canny_lanpaint_validation_inputs(
    cfg: Any,
    inputs_dir: str | Path,
) -> list[dict[str, Any]]:
    """Prepare validation windows once so multiple LoRAs can reuse them."""

    inputs_path = Path(inputs_dir)
    inputs_path.mkdir(parents=True, exist_ok=True)
    prepared_windows = [
        _prepare_validation_window(cfg, inputs_path, window) for window in cfg.validation_windows
    ]
    return prepared_windows


def run_flux_canny_lanpaint_prepared_validation(
    pipe: Any,
    cfg: Any,
    run_dir: str | Path,
    *,
    stage: str,
    prepared_windows: list[dict[str, Any]],
    lora_dir: str | Path | None,
    lora_enabled: bool,
    torch_dtype: Any,
    update_run_metadata: bool = True,
    big_branch_mode: str = "identical",
    big_guidance_scale: float | None = None,
    schedule_mode: str = "diffusers_default",
) -> tuple[Path, list[dict[str, Any]]]:
    """Run LanPaint validation from prepared static inputs using a loaded pipeline."""

    import json

    import torch
    from LanPaint.lanpaint import LanPaint

    run_path = Path(run_dir)
    validation_dir = run_path / stage
    validation_dir.mkdir(parents=True, exist_ok=True)

    lora_path = Path(lora_dir) if lora_dir is not None else None
    window_results: list[dict[str, Any]] = []
    for window_index, prepared in enumerate(prepared_windows):
        window_record = dict(prepared["window_record"])
        window_id = str(window_record["window_id"])
        window_size = int(prepared["window_size"])
        window_seed = int(cfg.validation_seed) + window_index

        adapter = FluxCannyLanPaintAdapter(
            pipe,
            use_transformer_cache_context=cfg.validation_use_transformer_cache_context,
        )
        device = adapter.device
        generator_device = device if device.type == "cuda" else torch.device("cpu")
        generator = torch.Generator(device=generator_device).manual_seed(window_seed)

        input_crop = prepared["input_crop"]
        mask_crop = prepared["mask_crop"]
        canny_control = prepared["canny_control"]
        window_dir = validation_dir / window_id
        window_dir.mkdir(parents=True, exist_ok=True)
        mask_keep_path = window_dir / "mask_keep_for_lanpaint.png"
        raw_output_path = window_dir / "raw_output.png"
        composite_path = window_dir / "composite.png"
        comparison_path = window_dir / "comparison.png"
        window_metadata_path = window_dir / "window_metadata.json"

        mask_keep = mask_edit_to_lanpaint_keep(mask_crop)
        mask_keep.save(mask_keep_path)

        clear_cuda_cache()
        started = time.perf_counter()
        adapter.encode_prompt(
            cfg.validation_prompt,
            device=device,
            max_sequence_length=cfg.validation_max_sequence_length,
        )
        y_latent = adapter.encode_image_latents(
            input_crop,
            window_size,
            window_size,
            generator,
            device,
        )
        control_latent = adapter.encode_control_latents(canny_control, generator, device)
        mask_keep_latent, mask_edit_latent = adapter.mask_keep_to_latents(mask_keep)

        model_wrapper = FluxCannyLanPaintModelWrapper(
            adapter,
            guidance_scale=cfg.validation_guidance_scale,
            big_branch_mode=big_branch_mode,
            big_guidance_scale=big_guidance_scale,
        )
        lanpaint = LanPaint(
            Model=model_wrapper,
            NSteps=cfg.validation_lanpaint_n_steps,
            Friction=cfg.validation_lanpaint_friction,
            Lambda=cfg.validation_lanpaint_lambda,
            Beta=cfg.validation_lanpaint_beta,
            StepSize=cfg.validation_lanpaint_step_size,
            IS_FLUX=True,
            IS_FLOW=True,
        )

        timesteps, flow_ts = adapter.prepare_timesteps(
            cfg.validation_num_inference_steps,
            device,
            schedule_mode=schedule_mode,
        )
        schedule_length = len(flow_ts)
        start_index = min(
            schedule_length - 1,
            int((1.0 - cfg.validation_partial_noise_strength) * schedule_length),
        )
        active_timesteps = timesteps[start_index:]
        active_flow_ts = flow_ts[start_index:]

        noise = torch.randn(
            y_latent.shape,
            generator=generator,
            device=device,
            dtype=y_latent.dtype,
        )
        first_flow = active_flow_ts[0:1].reshape(1, 1, 1).to(
            device=device,
            dtype=y_latent.dtype,
        )
        latents = model_wrapper.noise_scaling(first_flow, noise, y_latent).to(y_latent.dtype)

        if hasattr(adapter.scheduler, "set_begin_index"):
            adapter.scheduler.set_begin_index(start_index)
        torch.manual_seed(window_seed)

        step_trace: list[dict[str, Any]] = []
        print(
            f"Running {stage}/{window_id}: "
            f"box=({window_record['x']}, {window_record['y']}, "
            f"{window_record['x'] + window_record['width']}, "
            f"{window_record['y'] + window_record['height']}), "
            f"mask_fraction={window_record['mask_fraction']:.3%}, "
            f"strength={cfg.validation_partial_noise_strength}, seed={window_seed}, "
            f"lora_enabled={lora_enabled}"
        )

        with torch.inference_mode():
            for i, (t, flow_t) in enumerate(zip(active_timesteps, active_flow_ts, strict=True)):
                flow_t_value = float(flow_t.item())
                n_steps_override = (
                    0 if (len(active_timesteps) - i <= cfg.validation_lanpaint_early_stop) else None
                )
                x0 = lanpaint(
                    x=latents,
                    latent_image=y_latent,
                    noise=noise,
                    sigma=torch.tensor([flow_t_value], device=device, dtype=latents.dtype),
                    latent_mask=mask_keep_latent.to(device=device, dtype=latents.dtype),
                    current_times=make_current_times(flow_t_value, device, dtype=latents.dtype),
                    model_options={},
                    seed=window_seed,
                    n_steps=n_steps_override,
                )
                noise_pred = ((latents - x0.to(latents.dtype)) / max(flow_t_value, 1e-6)).to(
                    latents.dtype
                )
                latents = adapter.scheduler.step(noise_pred, t, latents, return_dict=False)[0].to(
                    y_latent.dtype
                )

                if cfg.validation_reinject_keep_latents:
                    if i < len(active_flow_ts) - 1:
                        next_flow = active_flow_ts[i + 1 : i + 2].reshape(1, 1, 1).to(
                            device=device,
                            dtype=y_latent.dtype,
                        )
                        keep_reference = model_wrapper.noise_scaling(
                            next_flow,
                            noise,
                            y_latent,
                        ).to(y_latent.dtype)
                    else:
                        keep_reference = y_latent
                    latents = mask_edit_latent.to(
                        device=device,
                        dtype=latents.dtype,
                    ) * latents + (
                        1.0 - mask_edit_latent.to(device=device, dtype=latents.dtype)
                    ) * keep_reference

                step_trace.append(
                    {
                        "step_index": int(i),
                        "timestep": scalar_float(t),
                        "flow_t": flow_t_value,
                        "abt": scalar_float(
                            make_current_times(flow_t_value, device, dtype=latents.dtype)[1]
                        ),
                        "ve_sigma": scalar_float(
                            make_current_times(flow_t_value, device, dtype=latents.dtype)[0]
                        ),
                        "lanpaint_inner_steps_override": n_steps_override,
                    }
                )

        raw_output = adapter.decode_latents(latents)
        runtime_seconds = time.perf_counter() - started
        if raw_output.size != input_crop.size:
            raw_output = raw_output.resize(input_crop.size, Image.Resampling.LANCZOS)
        raw_output.save(raw_output_path)

        composite = hard_composite(input_crop, raw_output, mask_crop)
        changed_outside_mask = outside_mask_changed(input_crop, composite, mask_crop)
        if changed_outside_mask:
            raise AssertionError(f"{window_id}: hard composite changed outside mask")
        composite.save(composite_path)

        make_comparison_sheet(
            [
                input_crop,
                make_mask_overlay(input_crop, mask_crop, color=(255, 40, 40), alpha=110 / 255),
                canny_control,
                raw_output,
                composite,
            ],
            ["input crop", "mask overlay", "canny control", "LanPaint raw", "hard composite"],
            output_path=comparison_path,
            max_panel_side=512,
            label_height=28,
        )

        schedule_estimate = lanpaint_strength_schedule_estimate(
            cfg.validation_num_inference_steps,
            cfg.validation_partial_noise_strength,
        )
        actual_schedule = {
            "scheduler_report": getattr(adapter, "last_schedule_report", None),
            "step_trace": step_trace,
            "actual_recorded_num_steps": len(step_trace),
            "first_recorded_timestep": step_trace[0]["timestep"] if step_trace else None,
            "last_recorded_timestep": step_trace[-1]["timestep"] if step_trace else None,
            "first_recorded_flow_t": step_trace[0]["flow_t"] if step_trace else None,
            "last_recorded_flow_t": step_trace[-1]["flow_t"] if step_trace else None,
        }
        window_metadata = {
            **window_record,
            "stage": stage,
            "seed": window_seed,
            "elapsed_seconds": runtime_seconds,
            "model_id": cfg.model_id,
            "component_loader_class": type(pipe).__name__,
            "denoising_method": "official_lanpaint_notebook_adapter",
            "pipeline_call_used": False,
            "lora_enabled": lora_enabled,
            "lora_dir": str(lora_path) if lora_path is not None else None,
            "torch_dtype": str(torch_dtype),
            "prompt": cfg.validation_prompt,
            "control_source": prepared["resolved_control_source"],
            "latent_shapes": {
                "source_latents": list(y_latent.shape),
                "control_latents": list(control_latent.shape),
                "mask_keep_latent": list(mask_keep_latent.shape),
                "mask_edit_latent": list(mask_edit_latent.shape),
            },
            "inference_settings": {
                "strength": float(cfg.validation_partial_noise_strength),
                "partial_noise_control": "notebook_local_lanpaint_schedule_slice",
                "num_inference_steps": int(cfg.validation_num_inference_steps),
                "effective_num_inference_steps": len(step_trace),
                "schedule_start_index": int(start_index),
                "guidance_scale": float(cfg.validation_guidance_scale),
                "height": window_size,
                "width": window_size,
                "max_sequence_length": cfg.validation_max_sequence_length,
                "use_transformer_cache_context": cfg.validation_use_transformer_cache_context,
                "reinject_keep_latents": cfg.validation_reinject_keep_latents,
                "lp_n_steps": cfg.validation_lanpaint_n_steps,
                "lp_friction": cfg.validation_lanpaint_friction,
                "lp_lambda": cfg.validation_lanpaint_lambda,
                "lp_beta": cfg.validation_lanpaint_beta,
                "lp_step_size": cfg.validation_lanpaint_step_size,
                "lp_early_stop": cfg.validation_lanpaint_early_stop,
                "lanpaint_is_flux": True,
                "lanpaint_is_flow": True,
                "lanpaint_schedule_mode": schedule_mode,
                "big_branch_mode": big_branch_mode,
                "big_guidance_scale": big_guidance_scale
                if big_guidance_scale is not None
                else cfg.validation_guidance_scale,
            },
            "big_branch": model_wrapper.branch_summary(),
            "lanpaint_strength_schedule_estimate": schedule_estimate,
            "actual_schedule_trace": actual_schedule,
            "canny_settings": {
                "source": prepared["resolved_control_source"],
                "low_threshold": cfg.canny_low_threshold,
                "high_threshold": cfg.canny_high_threshold,
                "blur_radius": cfg.validation_canny_blur_radius,
            },
            "mask_conventions": {
                "project_mask_edit": "white = edit, black = preserve",
                "lanpaint_mask_keep": "white/1 = keep, black/0 = edit",
            },
            "paths": {
                **{key: str(path) for key, path in prepared["paths"].items()},
                "mask_keep_for_lanpaint": str(mask_keep_path),
                "raw_output": str(raw_output_path),
                "composite": str(composite_path),
                "comparison": str(comparison_path),
                "source_base_image": str(prepared["base_path"]),
                "source_mask": str(prepared["mask_path"])
                if prepared["mask_path"] is not None
                else None,
                "source_conditioning_prior": str(prepared["prior_path"])
                if prepared["prior_path"] is not None
                else None,
                "source_control_source": str(prepared["control_source_path"]),
            },
            "outside_mask_changed_after_hard_composite": changed_outside_mask,
        }
        write_json(window_metadata_path, _jsonable(window_metadata))

        window_results.append(
            {
                "window_id": window_id,
                "window_metadata": str(window_metadata_path),
                "comparison": str(comparison_path),
                "composite": str(composite_path),
                "raw_output": str(raw_output_path),
                "seed": window_seed,
                "elapsed_seconds": runtime_seconds,
                "completed": True,
            }
        )
        print(f"Saved {stage}/{window_id} validation images to {window_dir}")

        del adapter, model_wrapper, lanpaint
        clear_cuda_cache()

    if update_run_metadata:
        run_metadata_path = run_path / "metadata.json"
        if run_metadata_path.exists():
            run_metadata = json.loads(run_metadata_path.read_text(encoding="utf-8"))
        else:
            run_metadata = {}
        validation_records = run_metadata.setdefault("validation_runs", {})
        validation_records[stage] = {
            "backend": "lanpaint",
            "path": str(validation_dir),
            "windows": window_results,
            "num_windows": len(window_results),
            "lora_enabled": lora_enabled,
            "completed": True,
            "package_versions": package_versions(
                ["torch", "diffusers", "transformers", "accelerate", "LanPaint"]
            ),
        }
        write_json(run_metadata_path, _jsonable(run_metadata))

    print(f"Saved {len(window_results)} LanPaint validation window(s) to {validation_dir}")
    return validation_dir, window_results


def run_flux_canny_lanpaint_validation(
    cfg: Any,
    run_dir: str | Path,
    *,
    stage: str,
    lora_dir: str | Path | None = None,
    require_lora: bool = False,
) -> Path | None:
    """Run fixed LanPaint validation windows."""

    if cfg.validation_backend != "lanpaint" or not cfg.run_inference_sanity:
        return None

    import json

    import torch
    from diffusers import FluxControlPipeline
    from huggingface_hub import get_token, login
    from LanPaint.lanpaint import LanPaint

    if not torch.cuda.is_available():
        raise RuntimeError("LanPaint validation requires a CUDA GPU.")

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN") or get_token()
    if not hf_token:
        raise RuntimeError("Missing HF_TOKEN for gated FLUX.1-Canny-dev validation.")
    login(token=hf_token, add_to_git_credential=False)

    run_path = Path(run_dir)
    validation_dir = run_path / stage
    validation_dir.mkdir(parents=True, exist_ok=True)

    lora_path = Path(lora_dir) if lora_dir is not None else None
    lora_weights = lora_path / "pytorch_lora_weights.safetensors" if lora_path is not None else None
    if require_lora and (lora_weights is None or not lora_weights.exists()):
        raise FileNotFoundError(f"Expected LoRA weights for validation: {lora_weights}")

    torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    clear_cuda_cache()
    pipe = FluxControlPipeline.from_pretrained(
        cfg.model_id,
        torch_dtype=torch_dtype,
        token=hf_token,
    )
    lora_enabled = False
    if lora_weights is not None and lora_weights.exists():
        torchao_error = torchao_lora_compatibility_error()
        if torchao_error is not None:
            raise RuntimeError(torchao_error)
        if not hasattr(pipe, "load_lora_weights"):
            raise RuntimeError(f"{type(pipe).__name__} does not expose load_lora_weights.")
        pipe.load_lora_weights(str(lora_path))
        lora_enabled = True

    if hasattr(pipe, "enable_model_cpu_offload"):
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")

    if getattr(pipe, "vae", None) is not None:
        if hasattr(pipe.vae, "enable_tiling"):
            pipe.vae.enable_tiling()
        if hasattr(pipe.vae, "enable_slicing"):
            pipe.vae.enable_slicing()

    _write_api_report(pipe, cfg, validation_dir, lora_enabled=lora_enabled)

    window_results: list[dict[str, Any]] = []
    for window_index, window in enumerate(cfg.validation_windows):
        prepared = _prepare_validation_window(cfg, validation_dir, window)
        window_record = prepared["window_record"]
        window_id = window_record["window_id"]
        window_size = int(prepared["window_size"])
        window_seed = int(cfg.validation_seed) + window_index

        adapter = FluxCannyLanPaintAdapter(
            pipe,
            use_transformer_cache_context=cfg.validation_use_transformer_cache_context,
        )
        device = adapter.device
        generator_device = device if device.type == "cuda" else torch.device("cpu")
        generator = torch.Generator(device=generator_device).manual_seed(window_seed)

        input_crop = prepared["input_crop"]
        mask_crop = prepared["mask_crop"]
        canny_control = prepared["canny_control"]
        window_dir = prepared["window_dir"]
        mask_keep_path = window_dir / "mask_keep_for_lanpaint.png"
        raw_output_path = window_dir / "raw_output.png"
        composite_path = window_dir / "composite.png"
        comparison_path = window_dir / "comparison.png"
        window_metadata_path = window_dir / "window_metadata.json"

        mask_keep = mask_edit_to_lanpaint_keep(mask_crop)
        mask_keep.save(mask_keep_path)

        clear_cuda_cache()
        started = time.perf_counter()
        adapter.encode_prompt(
            cfg.validation_prompt,
            device=device,
            max_sequence_length=cfg.validation_max_sequence_length,
        )
        y_latent = adapter.encode_image_latents(
            input_crop,
            window_size,
            window_size,
            generator,
            device,
        )
        control_latent = adapter.encode_control_latents(canny_control, generator, device)
        mask_keep_latent, mask_edit_latent = adapter.mask_keep_to_latents(mask_keep)

        model_wrapper = FluxCannyLanPaintModelWrapper(
            adapter,
            guidance_scale=cfg.validation_guidance_scale,
        )
        lanpaint = LanPaint(
            Model=model_wrapper,
            NSteps=cfg.validation_lanpaint_n_steps,
            Friction=cfg.validation_lanpaint_friction,
            Lambda=cfg.validation_lanpaint_lambda,
            Beta=cfg.validation_lanpaint_beta,
            StepSize=cfg.validation_lanpaint_step_size,
            IS_FLUX=True,
            IS_FLOW=True,
        )

        timesteps, flow_ts = adapter.prepare_timesteps(cfg.validation_num_inference_steps, device)
        schedule_length = len(flow_ts)
        start_index = min(
            schedule_length - 1,
            int((1.0 - cfg.validation_partial_noise_strength) * schedule_length),
        )
        active_timesteps = timesteps[start_index:]
        active_flow_ts = flow_ts[start_index:]

        noise = torch.randn(
            y_latent.shape,
            generator=generator,
            device=device,
            dtype=y_latent.dtype,
        )
        first_flow = active_flow_ts[0:1].reshape(1, 1, 1).to(
            device=device,
            dtype=y_latent.dtype,
        )
        latents = model_wrapper.noise_scaling(first_flow, noise, y_latent).to(y_latent.dtype)

        if hasattr(adapter.scheduler, "set_begin_index"):
            adapter.scheduler.set_begin_index(start_index)
        torch.manual_seed(window_seed)

        step_trace: list[dict[str, Any]] = []
        print(
            f"Running {stage}/{window_id}: "
            f"box=({window_record['x']}, {window_record['y']}, "
            f"{window_record['x'] + window_record['width']}, "
            f"{window_record['y'] + window_record['height']}), "
            f"mask_fraction={window_record['mask_fraction']:.3%}, "
            f"strength={cfg.validation_partial_noise_strength}, seed={window_seed}, "
            f"lora_enabled={lora_enabled}"
        )

        with torch.inference_mode():
            for i, (t, flow_t) in enumerate(zip(active_timesteps, active_flow_ts, strict=True)):
                flow_t_value = float(flow_t.item())
                n_steps_override = (
                    0 if (len(active_timesteps) - i <= cfg.validation_lanpaint_early_stop) else None
                )
                x0 = lanpaint(
                    x=latents,
                    latent_image=y_latent,
                    noise=noise,
                    sigma=torch.tensor([flow_t_value], device=device, dtype=latents.dtype),
                    latent_mask=mask_keep_latent.to(device=device, dtype=latents.dtype),
                    current_times=make_current_times(flow_t_value, device, dtype=latents.dtype),
                    model_options={},
                    seed=window_seed,
                    n_steps=n_steps_override,
                )
                noise_pred = ((latents - x0.to(latents.dtype)) / max(flow_t_value, 1e-6)).to(
                    latents.dtype
                )
                latents = adapter.scheduler.step(noise_pred, t, latents, return_dict=False)[0].to(
                    y_latent.dtype
                )

                if cfg.validation_reinject_keep_latents:
                    if i < len(active_flow_ts) - 1:
                        next_flow = active_flow_ts[i + 1 : i + 2].reshape(1, 1, 1).to(
                            device=device,
                            dtype=y_latent.dtype,
                        )
                        keep_reference = model_wrapper.noise_scaling(
                            next_flow,
                            noise,
                            y_latent,
                        ).to(y_latent.dtype)
                    else:
                        keep_reference = y_latent
                    latents = mask_edit_latent.to(
                        device=device,
                        dtype=latents.dtype,
                    ) * latents + (
                        1.0 - mask_edit_latent.to(device=device, dtype=latents.dtype)
                    ) * keep_reference

                step_trace.append(
                    {
                        "step_index": int(i),
                        "timestep": scalar_float(t),
                        "flow_t": flow_t_value,
                        "lanpaint_inner_steps_override": n_steps_override,
                    }
                )

        raw_output = adapter.decode_latents(latents)
        runtime_seconds = time.perf_counter() - started
        if raw_output.size != input_crop.size:
            raw_output = raw_output.resize(input_crop.size, Image.Resampling.LANCZOS)
        raw_output.save(raw_output_path)

        composite = hard_composite(input_crop, raw_output, mask_crop)
        changed_outside_mask = outside_mask_changed(input_crop, composite, mask_crop)
        if changed_outside_mask:
            raise AssertionError(f"{window_id}: hard composite changed outside mask")
        composite.save(composite_path)

        make_comparison_sheet(
            [
                input_crop,
                make_mask_overlay(input_crop, mask_crop, color=(255, 40, 40), alpha=110 / 255),
                canny_control,
                raw_output,
                composite,
            ],
            ["input crop", "mask overlay", "canny control", "LanPaint raw", "hard composite"],
            output_path=comparison_path,
            max_panel_side=512,
            label_height=28,
        )

        schedule_estimate = lanpaint_strength_schedule_estimate(
            cfg.validation_num_inference_steps,
            cfg.validation_partial_noise_strength,
        )
        actual_schedule = {
            "step_trace": step_trace,
            "actual_recorded_num_steps": len(step_trace),
            "first_recorded_timestep": step_trace[0]["timestep"] if step_trace else None,
            "last_recorded_timestep": step_trace[-1]["timestep"] if step_trace else None,
            "first_recorded_flow_t": step_trace[0]["flow_t"] if step_trace else None,
            "last_recorded_flow_t": step_trace[-1]["flow_t"] if step_trace else None,
        }
        window_metadata = {
            **window_record,
            "stage": stage,
            "seed": window_seed,
            "elapsed_seconds": runtime_seconds,
            "model_id": cfg.model_id,
            "component_loader_class": type(pipe).__name__,
            "denoising_method": "official_lanpaint_notebook_adapter",
            "pipeline_call_used": False,
            "lora_enabled": lora_enabled,
            "lora_dir": str(lora_path) if lora_path is not None else None,
            "torch_dtype": str(torch_dtype),
            "prompt": cfg.validation_prompt,
            "control_source": prepared["resolved_control_source"],
            "latent_shapes": {
                "source_latents": list(y_latent.shape),
                "control_latents": list(control_latent.shape),
                "mask_keep_latent": list(mask_keep_latent.shape),
                "mask_edit_latent": list(mask_edit_latent.shape),
            },
            "inference_settings": {
                "strength": float(cfg.validation_partial_noise_strength),
                "partial_noise_control": "notebook_local_lanpaint_schedule_slice",
                "num_inference_steps": int(cfg.validation_num_inference_steps),
                "effective_num_inference_steps": len(step_trace),
                "schedule_start_index": int(start_index),
                "guidance_scale": float(cfg.validation_guidance_scale),
                "height": window_size,
                "width": window_size,
                "max_sequence_length": cfg.validation_max_sequence_length,
                "use_transformer_cache_context": cfg.validation_use_transformer_cache_context,
                "reinject_keep_latents": cfg.validation_reinject_keep_latents,
                "lp_n_steps": cfg.validation_lanpaint_n_steps,
                "lp_friction": cfg.validation_lanpaint_friction,
                "lp_lambda": cfg.validation_lanpaint_lambda,
                "lp_beta": cfg.validation_lanpaint_beta,
                "lp_step_size": cfg.validation_lanpaint_step_size,
                "lp_early_stop": cfg.validation_lanpaint_early_stop,
                "lanpaint_is_flux": True,
                "lanpaint_is_flow": True,
            },
            "lanpaint_strength_schedule_estimate": schedule_estimate,
            "actual_schedule_trace": actual_schedule,
            "canny_settings": {
                "source": prepared["resolved_control_source"],
                "low_threshold": cfg.canny_low_threshold,
                "high_threshold": cfg.canny_high_threshold,
                "blur_radius": cfg.validation_canny_blur_radius,
            },
            "mask_conventions": {
                "project_mask_edit": "white = edit, black = preserve",
                "lanpaint_mask_keep": "white/1 = keep, black/0 = edit",
            },
            "paths": {
                **{key: str(path) for key, path in prepared["paths"].items()},
                "mask_keep_for_lanpaint": str(mask_keep_path),
                "raw_output": str(raw_output_path),
                "composite": str(composite_path),
                "comparison": str(comparison_path),
                "source_base_image": str(prepared["base_path"]),
                "source_mask": str(prepared["mask_path"])
                if prepared["mask_path"] is not None
                else None,
                "source_conditioning_prior": str(prepared["prior_path"])
                if prepared["prior_path"] is not None
                else None,
                "source_control_source": str(prepared["control_source_path"]),
            },
            "outside_mask_changed_after_hard_composite": changed_outside_mask,
        }
        write_json(window_metadata_path, _jsonable(window_metadata))

        window_results.append(
            {
                "window_id": window_id,
                "window_metadata": str(window_metadata_path),
                "comparison": str(comparison_path),
                "composite": str(composite_path),
                "raw_output": str(raw_output_path),
                "seed": window_seed,
                "completed": True,
            }
        )
        print(f"Saved {stage}/{window_id} validation images to {window_dir}")

        del adapter, model_wrapper, lanpaint
        clear_cuda_cache()

    run_metadata_path = run_path / "metadata.json"
    if run_metadata_path.exists():
        run_metadata = json.loads(run_metadata_path.read_text(encoding="utf-8"))
    else:
        run_metadata = {}
    validation_records = run_metadata.setdefault("validation_runs", {})
    validation_records[stage] = {
        "backend": "lanpaint",
        "path": str(validation_dir),
        "windows": window_results,
        "num_windows": len(window_results),
        "lora_enabled": lora_enabled,
        "completed": True,
        "package_versions": package_versions(
            ["torch", "diffusers", "transformers", "accelerate", "LanPaint"]
        ),
    }
    write_json(run_metadata_path, _jsonable(run_metadata))

    print(f"Saved {len(window_results)} LanPaint validation window(s) to {validation_dir}")

    del pipe
    clear_cuda_cache()
    return validation_dir


__all__ = [
    "FluxCannyLanPaintAdapter",
    "FluxCannyLanPaintModelWrapper",
    "clear_cuda_cache",
    "inspect_flux_canny_lanpaint_pipeline",
    "lanpaint_strength_schedule_estimate",
    "load_flux_canny_lanpaint_pipeline",
    "load_flux_canny_lora_adapter",
    "make_current_times",
    "mask_edit_to_lanpaint_keep",
    "prepare_flux_canny_lanpaint_validation_inputs",
    "remove_flux_canny_lora_adapter",
    "run_flux_canny_lanpaint_prepared_validation",
    "run_flux_canny_lanpaint_validation",
]
