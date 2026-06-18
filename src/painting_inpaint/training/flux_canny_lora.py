"""FLUX.1-Canny LoRA training helpers.

This module is extracted from the Colab Canny LoRA notebook so the same
training workflow can run from PBS jobs. Heavy ML libraries are imported lazily
inside training/inference functions to keep local tests lightweight.
"""

from __future__ import annotations

import copy
import gc
import json
import math
import random
import shutil
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image, ImageOps

from painting_inpaint.experiments.manifest import (
    create_run_dir,
    git_commit,
    package_versions,
    write_json,
)
from painting_inpaint.paths import resolve_data_path, runs_root

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}
PACKAGE_VERSION_NAMES = [
    "torch",
    "torchvision",
    "diffusers",
    "transformers",
    "accelerate",
    "peft",
    "bitsandbytes",
    "opencv-python",
    "opencv-python-headless",
    "huggingface-hub",
    "safetensors",
    "torchao",
]


def default_validation_windows() -> tuple[dict[str, Any], ...]:
    """Return the default fixed dataset validation windows."""

    return (
        {
            "window_id": "validation_image_001",
            "image_index": 0,
            "x_fraction": 0.50,
            "y_fraction": 0.50,
            "window_size": 1024,
        },
        {
            "window_id": "validation_image_002",
            "image_index": 1,
            "x_fraction": 0.50,
            "y_fraction": 0.50,
            "window_size": 1024,
        },
        {
            "window_id": "validation_image_003",
            "image_index": 2,
            "x_fraction": 0.50,
            "y_fraction": 0.50,
            "window_size": 1024,
        },
    )


@dataclass(frozen=True)
class FluxCannyLoraConfig:
    """Configuration for the FLUX.1-Canny Durer LoRA trainer."""

    name: str = "flux1_canny_dev_curated_durer_lora"
    model_id: str = "black-forest-labs/FLUX.1-Canny-dev"
    source_dir: str = "raw/curated_durer"
    case_name: str = "curated_durer"
    instance_prompt: str = "DURER_RESTO"
    resolution: int = 1024
    canny_low_threshold: int = 100
    canny_high_threshold: int = 200
    rank: int = 32
    learning_rate: float = 1e-4
    train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    max_train_steps: int = 5000
    checkpointing_steps: int = 500
    checkpoints_total_limit: int = 5
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 0
    mixed_precision: str = "bf16"
    gradient_checkpointing: bool = True
    use_8bit_adam: bool = True
    dataloader_num_workers: int = 2
    max_grad_norm: float = 1.0
    seed: int = 42
    guidance_scale: float = 3.5
    weighting_scheme: str = "none"
    logit_mean: float = 0.0
    logit_std: float = 1.0
    mode_scale: float = 1.29
    debug_num_preprocessed_samples: int = 16
    debug_same_image_index: int = 0
    debug_same_image_repeats: int = 6
    resume_checkpoint_path: str | None = None
    resume_from_latest_checkpoint: bool = True
    validation_backend: str = "lanpaint"
    run_pre_training_validation: bool = True
    run_inference_sanity: bool = True
    inference_lora_dir: str | None = None
    validation_source: str = "dataset"
    validation_dataset_source_dir: str | None = None
    validation_num_inference_steps: int = 50
    validation_seed: int = 1234
    validation_case_name: str = "curated_durer"
    validation_base_image_path: str | None = None
    validation_mask_image_path: str | None = None
    validation_conditioning_prior_path: str | None = None
    validation_target_long_side: int = 1024
    validation_window_size: int = 1024
    validation_window_id: str = "validation_image_001"
    validation_window_x_fraction: float = 0.50
    validation_window_y_fraction: float = 0.50
    validation_windows: tuple[dict[str, Any], ...] = field(
        default_factory=default_validation_windows
    )
    validation_control_source: str = "base_image"
    validation_mask_kind: str = "center_square"
    validation_mask_fraction: float = 0.35
    validation_mask_x_fraction: float = 0.50
    validation_mask_y_fraction: float = 0.50
    validation_canny_blur_radius: float = 0.0
    validation_prompt: str = ""
    validation_partial_noise_strength: float = 1.0
    validation_guidance_scale: float = 0.0
    validation_max_sequence_length: int = 512
    validation_use_transformer_cache_context: bool = True
    validation_reinject_keep_latents: bool = True
    validation_lanpaint_n_steps: int = 10
    validation_lanpaint_friction: float = 15.0
    validation_lanpaint_lambda: float = 10.0
    validation_lanpaint_beta: float = 1.0
    validation_lanpaint_step_size: float = 0.1
    validation_lanpaint_early_stop: int = 3


@dataclass(frozen=True)
class FluxCannyLoraRuntime:
    """Generated files and paths for one Canny LoRA run."""

    run_dir: Path
    config_dir: Path
    config_snapshot_path: Path
    metadata_path: Path
    debug_dir: Path
    debug_manifest_path: Path
    dataset_dir: Path
    image_count: int


class DurerCannyDataset:
    """Dataset that returns random image crops and matching Canny controls."""

    def __init__(
        self,
        image_paths: list[Path],
        prompt: str,
        resolution: int,
        canny_low: int,
        canny_high: int,
        random_crop: bool = True,
    ) -> None:
        self.image_paths = list(image_paths)
        self.prompt = prompt
        self.resolution = resolution
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.random_crop = random_crop
        self._image_transform = None

    def __len__(self) -> int:
        return len(self.image_paths)

    def _transform(self, image: Image.Image):
        if self._image_transform is None:
            from torchvision import transforms

            self._image_transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]
            )
        return self._image_transform(image)

    def make_sample_images(self, index: int, rng: random.Random | None = None) -> dict[str, Any]:
        path = self.image_paths[index % len(self.image_paths)]
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
        original_size = image.size
        image = conditional_resize(image, self.resolution)
        resized_size = image.size
        crop, crop_box = crop_image(
            image,
            self.resolution,
            random_crop=self.random_crop,
            rng=rng,
        )
        canny = make_canny_rgb(crop, self.canny_low, self.canny_high)
        return {
            "path": path,
            "crop": crop,
            "canny": canny,
            "original_size": original_size,
            "resized_size": resized_size,
            "crop_box": crop_box,
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.make_sample_images(index)
        return {
            "pixel_values": self._transform(sample["crop"]),
            "conditioning_pixel_values": self._transform(sample["canny"]),
            "captions": self.prompt,
            "image_path": str(sample["path"]),
        }


def load_flux_canny_lora_config(path: str | Path) -> FluxCannyLoraConfig:
    """Load a FLUX.1-Canny LoRA config from YAML."""

    config_path = Path(path)
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Training config must be a mapping: {config_path}")
    return flux_canny_lora_config_from_dict(cfg)


def validation_windows_from_config(validation: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    """Parse validation windows while keeping single-window config compatibility."""

    raw_windows = validation.get("windows")
    fallback_window = {
        "window_id": str(validation.get("window_id", "validation_image_001")),
        "image_index": int(validation.get("image_index", 0)),
        "x_fraction": float(validation.get("window_x_fraction", 0.50)),
        "y_fraction": float(validation.get("window_y_fraction", 0.50)),
        "window_size": int(validation.get("window_size", 1024)),
    }
    if raw_windows is None:
        return (fallback_window,)
    if not isinstance(raw_windows, list) or not raw_windows:
        raise ValueError("validation.windows must be a non-empty list when provided.")

    windows: list[dict[str, Any]] = []
    for index, item in enumerate(raw_windows, start=1):
        if not isinstance(item, dict):
            raise ValueError("Each validation.windows item must be a mapping.")
        window_id = str(item.get("window_id", item.get("id", f"window_{index:02d}")))
        windows.append(
            {
                "window_id": window_id,
                "x_fraction": float(item.get("x_fraction", fallback_window["x_fraction"])),
                "y_fraction": float(item.get("y_fraction", fallback_window["y_fraction"])),
                "window_size": int(item.get("window_size", fallback_window["window_size"])),
                **(
                    {"image_index": int(item["image_index"])}
                    if "image_index" in item and item["image_index"] is not None
                    else {}
                ),
                **(
                    {"image_path": str(item["image_path"])}
                    if item.get("image_path")
                    else {}
                ),
                **(
                    {"mask_fraction": float(item["mask_fraction"])}
                    if "mask_fraction" in item and item["mask_fraction"] is not None
                    else {}
                ),
                **(
                    {"mask_x_fraction": float(item["mask_x_fraction"])}
                    if "mask_x_fraction" in item and item["mask_x_fraction"] is not None
                    else {}
                ),
                **(
                    {"mask_y_fraction": float(item["mask_y_fraction"])}
                    if "mask_y_fraction" in item and item["mask_y_fraction"] is not None
                    else {}
                ),
            }
        )
    return tuple(windows)


def flux_canny_lora_config_from_dict(cfg: dict[str, Any]) -> FluxCannyLoraConfig:
    """Build a config dataclass from project YAML sections."""

    model = cfg.get("model", {})
    dataset = cfg.get("dataset", {})
    training = cfg.get("training", {})
    debug = cfg.get("debug", {})
    validation = cfg.get("validation", {})

    if (
        not isinstance(model, dict)
        or not isinstance(dataset, dict)
        or not isinstance(training, dict)
    ):
        raise ValueError("model, dataset, and training sections must be mappings.")
    if not isinstance(debug, dict) or not isinstance(validation, dict):
        raise ValueError("debug and validation sections must be mappings.")

    source_dir = dataset.get("source_dir")
    if not source_dir:
        raise ValueError("dataset.source_dir is required.")
    raw_validation_source = str(validation.get("source", "dataset"))
    validation_source_aliases = {
        "curated_durer": "dataset",
        "dataset_images": "dataset",
        "training_dataset": "dataset",
        "files": "configured_files",
        "rosary_files": "configured_files",
    }
    validation_source = validation_source_aliases.get(raw_validation_source, raw_validation_source)
    if validation_source not in {"dataset", "configured_files"}:
        raise ValueError("validation.source must be 'dataset' or 'configured_files'.")
    validation_control_source = str(validation.get("control_source", "conditioning_prior"))
    if validation_control_source not in {"conditioning_prior", "base_image"}:
        raise ValueError("validation.control_source must be 'conditioning_prior' or 'base_image'.")
    if validation_source == "dataset" and validation_control_source == "conditioning_prior":
        validation_control_source = "base_image"
    validation_partial_noise_strength = float(validation.get("partial_noise_strength", 1.0))
    if not (0.0 < validation_partial_noise_strength <= 1.0):
        raise ValueError("validation.partial_noise_strength must be in the interval (0.0, 1.0].")
    validation_mask_kind = str(validation.get("mask_kind", "center_square"))
    if validation_mask_kind != "center_square":
        raise ValueError("validation.mask_kind currently supports only 'center_square'.")
    validation_mask_fraction = float(validation.get("mask_fraction", 0.35))
    if not (0.0 < validation_mask_fraction <= 1.0):
        raise ValueError("validation.mask_fraction must be in the interval (0.0, 1.0].")
    validation_windows = validation_windows_from_config(validation)
    first_validation_window = validation_windows[0]

    return FluxCannyLoraConfig(
        name=str(cfg.get("name", "flux1_canny_dev_curated_durer_lora")),
        model_id=str(model.get("model_id", "black-forest-labs/FLUX.1-Canny-dev")),
        source_dir=str(source_dir),
        case_name=str(dataset.get("case_name", "curated_durer")),
        instance_prompt=str(dataset.get("instance_prompt", "DURER_RESTO")),
        resolution=int(dataset.get("resolution", 1024)),
        canny_low_threshold=int(dataset.get("canny_low_threshold", 100)),
        canny_high_threshold=int(dataset.get("canny_high_threshold", 200)),
        rank=int(training.get("rank", 32)),
        learning_rate=float(training.get("learning_rate", 1e-4)),
        train_batch_size=int(training.get("train_batch_size", 1)),
        gradient_accumulation_steps=int(training.get("gradient_accumulation_steps", 4)),
        max_train_steps=int(training.get("max_train_steps", 5000)),
        checkpointing_steps=int(training.get("checkpointing_steps", 500)),
        checkpoints_total_limit=int(training.get("checkpoints_total_limit", 5)),
        lr_scheduler=str(training.get("lr_scheduler", "constant")),
        lr_warmup_steps=int(training.get("lr_warmup_steps", 0)),
        mixed_precision=str(training.get("mixed_precision", "bf16")),
        gradient_checkpointing=bool(training.get("gradient_checkpointing", True)),
        use_8bit_adam=bool(training.get("use_8bit_adam", True)),
        dataloader_num_workers=int(training.get("dataloader_num_workers", 2)),
        max_grad_norm=float(training.get("max_grad_norm", 1.0)),
        seed=int(training.get("seed", 42)),
        guidance_scale=float(training.get("guidance_scale", 3.5)),
        weighting_scheme=str(training.get("weighting_scheme", "none")),
        logit_mean=float(training.get("logit_mean", 0.0)),
        logit_std=float(training.get("logit_std", 1.0)),
        mode_scale=float(training.get("mode_scale", 1.29)),
        debug_num_preprocessed_samples=int(debug.get("num_preprocessed_samples", 16)),
        debug_same_image_index=int(debug.get("same_image_index", 0)),
        debug_same_image_repeats=int(debug.get("same_image_repeats", 6)),
        resume_checkpoint_path=training.get("resume_checkpoint_path"),
        resume_from_latest_checkpoint=bool(training.get("resume_from_latest_checkpoint", True)),
        validation_backend=str(validation.get("backend", "lanpaint")),
        run_pre_training_validation=bool(validation.get("run_pre_training_validation", True)),
        run_inference_sanity=bool(validation.get("run_inference_sanity", True)),
        inference_lora_dir=validation.get("inference_lora_dir"),
        validation_source=validation_source,
        validation_dataset_source_dir=validation.get("dataset_source_dir"),
        validation_num_inference_steps=int(validation.get("num_inference_steps", 50)),
        validation_seed=int(validation.get("seed", 1234)),
        validation_case_name=str(validation.get("case_name", "curated_durer")),
        validation_base_image_path=validation.get("base_image_path"),
        validation_mask_image_path=validation.get("mask_image_path"),
        validation_conditioning_prior_path=validation.get("conditioning_prior_image_path"),
        validation_target_long_side=int(validation.get("target_long_side", 1024)),
        validation_window_size=int(first_validation_window["window_size"]),
        validation_window_id=str(first_validation_window["window_id"]),
        validation_window_x_fraction=float(first_validation_window["x_fraction"]),
        validation_window_y_fraction=float(first_validation_window["y_fraction"]),
        validation_windows=validation_windows,
        validation_control_source=validation_control_source,
        validation_mask_kind=validation_mask_kind,
        validation_mask_fraction=validation_mask_fraction,
        validation_mask_x_fraction=float(validation.get("mask_x_fraction", 0.50)),
        validation_mask_y_fraction=float(validation.get("mask_y_fraction", 0.50)),
        validation_canny_blur_radius=float(validation.get("canny_blur_radius", 0.0)),
        validation_prompt=str(validation.get("prompt", "")),
        validation_partial_noise_strength=validation_partial_noise_strength,
        validation_guidance_scale=float(validation.get("guidance_scale", 0.0)),
        validation_max_sequence_length=int(validation.get("max_sequence_length", 512)),
        validation_use_transformer_cache_context=bool(
            validation.get("use_transformer_cache_context", True)
        ),
        validation_reinject_keep_latents=bool(validation.get("reinject_keep_latents", True)),
        validation_lanpaint_n_steps=int(validation.get("lanpaint_n_steps", 10)),
        validation_lanpaint_friction=float(validation.get("lanpaint_friction", 15.0)),
        validation_lanpaint_lambda=float(validation.get("lanpaint_lambda", 10.0)),
        validation_lanpaint_beta=float(validation.get("lanpaint_beta", 1.0)),
        validation_lanpaint_step_size=float(validation.get("lanpaint_step_size", 0.1)),
        validation_lanpaint_early_stop=int(validation.get("lanpaint_early_stop", 3)),
    )


def config_to_yaml_payload(cfg: FluxCannyLoraConfig) -> dict[str, Any]:
    """Return a sectioned YAML payload for snapshots."""

    return {
        "name": cfg.name,
        "model": {"model_id": cfg.model_id},
        "dataset": {
            "source_dir": cfg.source_dir,
            "case_name": cfg.case_name,
            "instance_prompt": cfg.instance_prompt,
            "resolution": cfg.resolution,
            "canny_low_threshold": cfg.canny_low_threshold,
            "canny_high_threshold": cfg.canny_high_threshold,
        },
        "training": {
            "rank": cfg.rank,
            "learning_rate": cfg.learning_rate,
            "train_batch_size": cfg.train_batch_size,
            "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
            "max_train_steps": cfg.max_train_steps,
            "checkpointing_steps": cfg.checkpointing_steps,
            "checkpoints_total_limit": cfg.checkpoints_total_limit,
            "lr_scheduler": cfg.lr_scheduler,
            "lr_warmup_steps": cfg.lr_warmup_steps,
            "mixed_precision": cfg.mixed_precision,
            "gradient_checkpointing": cfg.gradient_checkpointing,
            "use_8bit_adam": cfg.use_8bit_adam,
            "dataloader_num_workers": cfg.dataloader_num_workers,
            "max_grad_norm": cfg.max_grad_norm,
            "seed": cfg.seed,
            "guidance_scale": cfg.guidance_scale,
            "weighting_scheme": cfg.weighting_scheme,
            "logit_mean": cfg.logit_mean,
            "logit_std": cfg.logit_std,
            "mode_scale": cfg.mode_scale,
            "resume_checkpoint_path": cfg.resume_checkpoint_path,
            "resume_from_latest_checkpoint": cfg.resume_from_latest_checkpoint,
        },
        "debug": {
            "num_preprocessed_samples": cfg.debug_num_preprocessed_samples,
            "same_image_index": cfg.debug_same_image_index,
            "same_image_repeats": cfg.debug_same_image_repeats,
        },
        "validation": {
            "backend": cfg.validation_backend,
            "run_pre_training_validation": cfg.run_pre_training_validation,
            "run_inference_sanity": cfg.run_inference_sanity,
            "inference_lora_dir": cfg.inference_lora_dir,
            "source": cfg.validation_source,
            "dataset_source_dir": cfg.validation_dataset_source_dir,
            "num_inference_steps": cfg.validation_num_inference_steps,
            "seed": cfg.validation_seed,
            "case_name": cfg.validation_case_name,
            "base_image_path": cfg.validation_base_image_path,
            "mask_image_path": cfg.validation_mask_image_path,
            "conditioning_prior_image_path": cfg.validation_conditioning_prior_path,
            "target_long_side": cfg.validation_target_long_side,
            "window_size": cfg.validation_window_size,
            "window_id": cfg.validation_window_id,
            "window_x_fraction": cfg.validation_window_x_fraction,
            "window_y_fraction": cfg.validation_window_y_fraction,
            "windows": [dict(window) for window in cfg.validation_windows],
            "control_source": cfg.validation_control_source,
            "mask_kind": cfg.validation_mask_kind,
            "mask_fraction": cfg.validation_mask_fraction,
            "mask_x_fraction": cfg.validation_mask_x_fraction,
            "mask_y_fraction": cfg.validation_mask_y_fraction,
            "canny_blur_radius": cfg.validation_canny_blur_radius,
            "prompt": cfg.validation_prompt,
            "partial_noise_strength": cfg.validation_partial_noise_strength,
            "guidance_scale": cfg.validation_guidance_scale,
            "max_sequence_length": cfg.validation_max_sequence_length,
            "use_transformer_cache_context": cfg.validation_use_transformer_cache_context,
            "reinject_keep_latents": cfg.validation_reinject_keep_latents,
            "lanpaint_n_steps": cfg.validation_lanpaint_n_steps,
            "lanpaint_friction": cfg.validation_lanpaint_friction,
            "lanpaint_lambda": cfg.validation_lanpaint_lambda,
            "lanpaint_beta": cfg.validation_lanpaint_beta,
            "lanpaint_step_size": cfg.validation_lanpaint_step_size,
            "lanpaint_early_stop": cfg.validation_lanpaint_early_stop,
        },
    }


def list_image_paths(root: str | Path) -> list[Path]:
    """Return deterministic image paths below ``root``."""

    folder = Path(root)
    if not folder.exists():
        raise FileNotFoundError(folder)
    if not folder.is_dir():
        raise NotADirectoryError(folder)
    return sorted(
        (
            path
            for path in folder.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=lambda path: str(path.relative_to(folder)).lower(),
    )


def resolve_dataset_dir(cfg: FluxCannyLoraConfig) -> Path:
    """Resolve the configured source directory relative to the data root."""

    dataset_dir = resolve_data_path(cfg.source_dir)
    if dataset_dir is None:
        raise ValueError("source_dir is required.")
    return dataset_dir.resolve()


def conditional_resize(image: Image.Image, resolution: int) -> Image.Image:
    """Upscale only when the short side is smaller than ``resolution``."""

    width, height = image.size
    short_side = min(width, height)
    if short_side >= resolution:
        return image
    scale = resolution / short_side
    new_size = (round(width * scale), round(height * scale))
    return image.resize(new_size, Image.Resampling.LANCZOS)


def crop_image(
    image: Image.Image,
    resolution: int,
    *,
    random_crop: bool = True,
    rng: random.Random | None = None,
) -> tuple[Image.Image, dict[str, int]]:
    """Crop a square training sample from an image."""

    width, height = image.size
    if width < resolution or height < resolution:
        raise ValueError(f"Image is too small after resize: {image.size}, requested {resolution}")
    if random_crop:
        rng = rng or random
        left = rng.randint(0, width - resolution)
        top = rng.randint(0, height - resolution)
    else:
        left = (width - resolution) // 2
        top = (height - resolution) // 2
    return image.crop((left, top, left + resolution, top + resolution)), {
        "left": left,
        "top": top,
        "width": resolution,
        "height": resolution,
    }


def make_canny_rgb(image: Image.Image, low: int, high: int) -> Image.Image:
    """Return an RGB Canny edge control image."""

    import cv2

    arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, low, high)
    return Image.fromarray(np.repeat(edges[:, :, None], 3, axis=2))


def make_side_by_side(crop: Image.Image, canny: Image.Image) -> Image.Image:
    """Return a target/control preview image."""

    preview = Image.new("RGB", (crop.width * 2, crop.height), "white")
    preview.paste(crop, (0, 0))
    preview.paste(canny, (crop.width, 0))
    return preview


def collate_fn(examples: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate Canny LoRA examples for torch DataLoader."""

    import torch

    pixel_values = torch.stack([example["pixel_values"] for example in examples]).to(
        memory_format=torch.contiguous_format
    ).float()
    conditioning_pixel_values = torch.stack(
        [example["conditioning_pixel_values"] for example in examples]
    ).to(memory_format=torch.contiguous_format).float()
    return {
        "pixel_values": pixel_values,
        "conditioning_pixel_values": conditioning_pixel_values,
        "captions": [example["captions"] for example in examples],
        "image_paths": [example["image_path"] for example in examples],
    }


def checkpoint_step(path: Path) -> int | None:
    """Return the integer step for a ``checkpoint-N`` path."""

    if not path.is_dir() or not path.name.startswith("checkpoint-"):
        return None
    try:
        return int(path.name.split("-")[-1])
    except ValueError:
        return None


def checkpoint_dirs(run_dir: str | Path) -> list[Path]:
    """Return sorted checkpoint directories for a run."""

    paths: list[tuple[int, Path]] = []
    for path in Path(run_dir).glob("checkpoint-*"):
        step = checkpoint_step(path)
        if step is not None:
            paths.append((step, path))
    return [path for _step, path in sorted(paths)]


def latest_checkpoint(run_dir: str | Path) -> Path | None:
    """Return the latest checkpoint directory in a run folder."""

    checkpoints = checkpoint_dirs(run_dir)
    return checkpoints[-1] if checkpoints else None


def resolve_resume_checkpoint(cfg: FluxCannyLoraConfig, run_dir: str | Path) -> Path | None:
    """Resolve explicit or latest checkpoint for resuming."""

    if cfg.resume_checkpoint_path is not None:
        checkpoint = Path(cfg.resume_checkpoint_path)
        if not checkpoint.exists():
            raise FileNotFoundError(f"resume_checkpoint_path does not exist: {checkpoint}")
        return checkpoint
    if cfg.resume_from_latest_checkpoint:
        return latest_checkpoint(run_dir)
    return None


def prune_checkpoints(run_dir: str | Path, limit: int) -> list[Path]:
    """Delete old checkpoints to keep at most ``limit`` directories."""

    if limit <= 0:
        return []
    checkpoints = checkpoint_dirs(run_dir)
    to_delete = checkpoints[: max(0, len(checkpoints) - limit)]
    for old in to_delete:
        shutil.rmtree(old)
    return to_delete


def load_existing_metadata(run_dir: str | Path) -> dict[str, Any]:
    """Load existing run metadata when present."""

    path = Path(run_dir) / "metadata.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


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


def _write_config_snapshot(
    cfg: FluxCannyLoraConfig,
    *,
    run_dir: Path,
    source_config: str | Path | None = None,
) -> Path:
    config_dir = run_dir / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = config_dir / "config_snapshot.yaml"
    snapshot_path.write_text(
        yaml.safe_dump(config_to_yaml_payload(cfg), sort_keys=False),
        encoding="utf-8",
    )
    if source_config is not None:
        source = Path(source_config)
        if source.exists():
            shutil.copy2(source, config_dir / source.name)
    return snapshot_path


def _metadata_for_config(
    cfg: FluxCannyLoraConfig,
    *,
    run_dir: Path,
    dataset_dir: Path,
    image_count: int,
    debug_dir: Path,
    config_snapshot_path: Path,
    preflight_only: bool,
) -> dict[str, Any]:
    return {
        "name": cfg.name,
        "method": "flux1_canny_lora",
        "base_model": cfg.model_id,
        "dataset_name": cfg.case_name,
        "dataset_path": str(dataset_dir),
        "image_count": image_count,
        "resolution": cfg.resolution,
        "prompt": cfg.instance_prompt,
        "crop_policy": (
            "resize only if min(width, height) < resolution, "
            "then random square crop"
        ),
        "canny_implementation": "OpenCV cv2.Canny",
        "canny_low_threshold": cfg.canny_low_threshold,
        "canny_high_threshold": cfg.canny_high_threshold,
        "canny_randomization_enabled": False,
        "rank": cfg.rank,
        "learning_rate": cfg.learning_rate,
        "max_train_steps": cfg.max_train_steps,
        "train_batch_size": cfg.train_batch_size,
        "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
        "checkpointing_steps": cfg.checkpointing_steps,
        "checkpoints_total_limit": cfg.checkpoints_total_limit,
        "mixed_precision": cfg.mixed_precision,
        "gradient_checkpointing": cfg.gradient_checkpointing,
        "seed": cfg.seed,
        "run_dir": str(run_dir),
        "debug_dir": str(debug_dir),
        "config_snapshot": str(config_snapshot_path.relative_to(run_dir)),
        "resume_from_latest_checkpoint": cfg.resume_from_latest_checkpoint,
        "resume_checkpoint_path": cfg.resume_checkpoint_path,
        "resumed_checkpoint_path": None,
        "completed_steps": None,
        "last_loss": None,
        "optimizer": None,
        "validation": {
            "backend": cfg.validation_backend,
            "run_pre_training_validation": cfg.run_pre_training_validation,
            "run_inference_sanity": cfg.run_inference_sanity,
            "source": cfg.validation_source,
            "dataset_source_dir": cfg.validation_dataset_source_dir,
            "num_inference_steps": cfg.validation_num_inference_steps,
            "seed": cfg.validation_seed,
            "case_name": cfg.validation_case_name,
            "base_image_path": cfg.validation_base_image_path,
            "mask_image_path": cfg.validation_mask_image_path,
            "conditioning_prior_image_path": cfg.validation_conditioning_prior_path,
            "target_long_side": cfg.validation_target_long_side,
            "window_size": cfg.validation_window_size,
            "window_id": cfg.validation_window_id,
            "window_x_fraction": cfg.validation_window_x_fraction,
            "window_y_fraction": cfg.validation_window_y_fraction,
            "windows": [dict(window) for window in cfg.validation_windows],
            "control_source": cfg.validation_control_source,
            "mask_kind": cfg.validation_mask_kind,
            "mask_fraction": cfg.validation_mask_fraction,
            "mask_x_fraction": cfg.validation_mask_x_fraction,
            "mask_y_fraction": cfg.validation_mask_y_fraction,
            "canny_blur_radius": cfg.validation_canny_blur_radius,
            "prompt": cfg.validation_prompt,
            "partial_noise_strength": cfg.validation_partial_noise_strength,
            "guidance_scale": cfg.validation_guidance_scale,
            "max_sequence_length": cfg.validation_max_sequence_length,
            "use_transformer_cache_context": cfg.validation_use_transformer_cache_context,
            "reinject_keep_latents": cfg.validation_reinject_keep_latents,
            "lanpaint_n_steps": cfg.validation_lanpaint_n_steps,
            "lanpaint_friction": cfg.validation_lanpaint_friction,
            "lanpaint_lambda": cfg.validation_lanpaint_lambda,
            "lanpaint_beta": cfg.validation_lanpaint_beta,
            "lanpaint_step_size": cfg.validation_lanpaint_step_size,
            "lanpaint_early_stop": cfg.validation_lanpaint_early_stop,
        },
        "validation_runs": {},
        "preflight_only": preflight_only,
        "package_versions": package_versions(PACKAGE_VERSION_NAMES),
        "git_commit": git_commit(),
        "channel_expansion_performed": False,
    }


def export_debug_samples(
    dataset: DurerCannyDataset,
    cfg: FluxCannyLoraConfig,
    debug_dir: str | Path,
) -> Path:
    """Write debug crop/control previews and return the manifest path."""

    out_dir = Path(debug_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(cfg.seed)
    debug_manifest: list[dict[str, Any]] = []
    sample_count = min(cfg.debug_num_preprocessed_samples, len(dataset))

    for index in range(sample_count):
        sample = dataset.make_sample_images(index, rng=rng)
        crop_path = out_dir / f"{index:03d}_crop.png"
        canny_path = out_dir / f"{index:03d}_canny.png"
        preview_path = out_dir / f"{index:03d}_preview.png"
        sample["crop"].save(crop_path)
        sample["canny"].save(canny_path)
        make_side_by_side(sample["crop"], sample["canny"]).save(preview_path)
        debug_manifest.append(
            {
                "debug_type": "dataset_sequence_crop",
                "index": index,
                "source_path": str(sample["path"]),
                "original_size": list(sample["original_size"]),
                "resized_size": list(sample["resized_size"]),
                "crop_box": sample["crop_box"],
                "crop_path": str(crop_path),
                "canny_path": str(canny_path),
                "preview_path": str(preview_path),
            }
        )

    for repeat in range(cfg.debug_same_image_repeats):
        same_rng = random.Random(cfg.seed + 10_000 + repeat)
        sample = dataset.make_sample_images(cfg.debug_same_image_index, rng=same_rng)
        crop_path = out_dir / f"same_image_{repeat:03d}_crop.png"
        canny_path = out_dir / f"same_image_{repeat:03d}_canny.png"
        preview_path = out_dir / f"same_image_{repeat:03d}_preview.png"
        sample["crop"].save(crop_path)
        sample["canny"].save(canny_path)
        make_side_by_side(sample["crop"], sample["canny"]).save(preview_path)
        debug_manifest.append(
            {
                "debug_type": "same_image_repeated_crop",
                "index": cfg.debug_same_image_index,
                "repeat": repeat,
                "source_path": str(sample["path"]),
                "original_size": list(sample["original_size"]),
                "resized_size": list(sample["resized_size"]),
                "crop_box": sample["crop_box"],
                "crop_path": str(crop_path),
                "canny_path": str(canny_path),
                "preview_path": str(preview_path),
            }
        )

    manifest_path = write_json(out_dir / "manifest.json", {"samples": debug_manifest})
    return manifest_path


def create_flux_canny_lora_run_files(
    cfg: FluxCannyLoraConfig,
    *,
    run_root: str | Path | None = None,
    run_dir: str | Path | None = None,
    source_config: str | Path | None = None,
    preflight_only: bool = True,
) -> FluxCannyLoraRuntime:
    """Create run metadata, config snapshots, and debug samples."""

    if run_root is not None and run_dir is not None:
        raise ValueError("Pass either run_root or run_dir, not both.")

    dataset_dir = resolve_dataset_dir(cfg)
    image_paths = list_image_paths(dataset_dir)
    if not image_paths:
        raise FileNotFoundError(f"No images found under {dataset_dir}")

    if run_dir is not None:
        out_dir = Path(run_dir).expanduser()
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        root = Path(run_root).expanduser() if run_root is not None else runs_root() / "training"
        out_dir = create_run_dir(cfg.name, root=root)

    config_snapshot_path = _write_config_snapshot(cfg, run_dir=out_dir, source_config=source_config)
    debug_dir = out_dir / "debug_preprocessed_samples"
    dataset = DurerCannyDataset(
        image_paths,
        prompt=cfg.instance_prompt,
        resolution=cfg.resolution,
        canny_low=cfg.canny_low_threshold,
        canny_high=cfg.canny_high_threshold,
        random_crop=True,
    )
    debug_manifest_path = export_debug_samples(dataset, cfg, debug_dir)
    metadata = _metadata_for_config(
        cfg,
        run_dir=out_dir,
        dataset_dir=dataset_dir,
        image_count=len(image_paths),
        debug_dir=debug_dir,
        config_snapshot_path=config_snapshot_path,
        preflight_only=preflight_only,
    )
    metadata_path = write_json(out_dir / "metadata.json", _jsonable(metadata))

    return FluxCannyLoraRuntime(
        run_dir=out_dir,
        config_dir=out_dir / "config",
        config_snapshot_path=config_snapshot_path,
        metadata_path=metadata_path,
        debug_dir=debug_dir,
        debug_manifest_path=debug_manifest_path,
        dataset_dir=dataset_dir,
        image_count=len(image_paths),
    )


def unwrap_model(accelerator, model):
    """Return an unwrapped model, handling compiled wrappers."""

    from diffusers.utils.torch_utils import is_compiled_module

    model = accelerator.unwrap_model(model)
    return model._orig_mod if is_compiled_module(model) else model


def encode_images(pixels, vae, weight_dtype):
    """Encode normalized image tensors into FLUX latents."""

    pixel_latents = vae.encode(pixels.to(device=vae.device, dtype=vae.dtype)).latent_dist.sample()
    pixel_latents = (pixel_latents - vae.config.shift_factor) * vae.config.scaling_factor
    return pixel_latents.to(weight_dtype)


def repeat_prompt_tensors(prompt_embeds_base, pooled_prompt_embeds_base, bsz, device, dtype):
    """Repeat precomputed prompt tensors for a batch."""

    prompt_embeds = prompt_embeds_base.to(device=device, dtype=dtype).repeat(bsz, 1, 1)
    pooled_prompt_embeds = pooled_prompt_embeds_base.to(device=device, dtype=dtype).repeat(bsz, 1)
    return prompt_embeds, pooled_prompt_embeds


def tensor_to_pil(tensor) -> Image.Image:
    """Convert a normalized CHW tensor to an RGB image."""

    tensor = tensor.detach().float().cpu().clamp(-1, 1)
    arr = ((tensor + 1) / 2 * 255).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(arr)


def train_flux_canny_lora(cfg: FluxCannyLoraConfig, run_dir: str | Path) -> Path:
    """Run the full FLUX.1-Canny LoRA training loop."""

    import torch
    from accelerate import Accelerator
    from accelerate.utils import ProjectConfiguration, set_seed
    from diffusers import (
        AutoencoderKL,
        FlowMatchEulerDiscreteScheduler,
        FluxControlPipeline,
        FluxTransformer2DModel,
    )
    from diffusers.optimization import get_scheduler
    from diffusers.training_utils import (
        compute_density_for_timestep_sampling,
        compute_loss_weighting_for_sd3,
        free_memory,
    )
    from huggingface_hub import get_token
    from peft import LoraConfig, set_peft_model_state_dict
    from peft.utils import get_peft_model_state_dict
    from torch.utils.data import DataLoader
    from tqdm.auto import tqdm

    run_path = Path(run_dir)
    dataset_dir = resolve_dataset_dir(cfg)
    image_paths = list_image_paths(dataset_dir)
    train_dataset = DurerCannyDataset(
        image_paths,
        prompt=cfg.instance_prompt,
        resolution=cfg.resolution,
        canny_low=cfg.canny_low_threshold,
        canny_high=cfg.canny_high_threshold,
        random_crop=True,
    )

    import os

    hf_token = os.environ.get("HF_TOKEN") or get_token()
    if hf_token is None:
        print(
            "HF token not found. Run huggingface-cli login or set HF_TOKEN before "
            "loading gated FLUX models."
        )

    set_seed(cfg.seed)
    logging_dir = run_path / "logs"
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        mixed_precision=cfg.mixed_precision,
        project_config=ProjectConfiguration(
            project_dir=str(run_path),
            logging_dir=str(logging_dir),
        ),
    )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae = AutoencoderKL.from_pretrained(
        cfg.model_id,
        subfolder="vae",
        token=hf_token,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    transformer = FluxTransformer2DModel.from_pretrained(
        cfg.model_id,
        subfolder="transformer",
        token=hf_token,
        torch_dtype=weight_dtype,
        low_cpu_mem_usage=True,
    )
    original_in_channels = int(transformer.config.in_channels)
    original_x_embedder_in_features = int(transformer.x_embedder.in_features)
    original_x_embedder_shape = tuple(transformer.x_embedder.weight.shape)
    latent_channels = int(vae.config.latent_channels)
    print(f"transformer.config.in_channels: {original_in_channels}")
    print(f"transformer.x_embedder.in_features: {original_x_embedder_in_features}")
    print(f"vae.config.latent_channels: {latent_channels}")
    print(f"original x_embedder weight shape: {original_x_embedder_shape}")
    print(f"expected unpacked target+control latent channels before packing: {latent_channels * 2}")

    transformer.requires_grad_(False)
    vae.requires_grad_(False)
    vae.to(accelerator.device, dtype=torch.float32)
    transformer.to(accelerator.device, dtype=weight_dtype)

    if cfg.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    target_modules = [
        "x_embedder",
        "attn.to_k",
        "attn.to_q",
        "attn.to_v",
        "attn.to_out.0",
        "attn.add_k_proj",
        "attn.add_q_proj",
        "attn.add_v_proj",
        "attn.to_add_out",
        "ff.net.0.proj",
        "ff.net.2",
        "ff_context.net.0.proj",
        "ff_context.net.2",
    ]
    transformer.add_adapter(
        LoraConfig(
            r=cfg.rank,
            lora_alpha=cfg.rank,
            init_lora_weights=True,
            target_modules=target_modules,
        )
    )
    trainable_names = [
        name for name, param in transformer.named_parameters() if param.requires_grad
    ]
    if not trainable_names:
        raise RuntimeError("No trainable LoRA parameters found.")
    non_lora_trainable = [name for name in trainable_names if "lora" not in name]
    if non_lora_trainable:
        raise RuntimeError(f"Unexpected non-LoRA trainable parameters: {non_lora_trainable[:20]}")
    print(f"Trainable LoRA parameter tensors: {len(trainable_names)}")

    if cfg.use_8bit_adam:
        try:
            import bitsandbytes as bnb

            optimizer_cls = bnb.optim.AdamW8bit
            optimizer_name = "AdamW8bit"
        except Exception as exc:
            print(f"bitsandbytes unavailable, using AdamW. Reason: {exc}")
            optimizer_cls = torch.optim.AdamW
            optimizer_name = "AdamW"
    else:
        optimizer_cls = torch.optim.AdamW
        optimizer_name = "AdamW"

    trainable_params = [param for param in transformer.parameters() if param.requires_grad]
    optimizer = optimizer_cls(trainable_params, lr=cfg.learning_rate)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=cfg.dataloader_num_workers,
    )
    lr_scheduler = get_scheduler(
        cfg.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=cfg.max_train_steps * accelerator.num_processes,
    )
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        cfg.model_id,
        subfolder="scheduler",
        token=hf_token,
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)

    transformer_cls = type(transformer)

    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            transformer_lora_layers_to_save = None
            for model in models:
                if isinstance(unwrap_model(accelerator, model), transformer_cls):
                    transformer_lora_layers_to_save = get_peft_model_state_dict(
                        unwrap_model(accelerator, model)
                    )
                else:
                    raise ValueError(f"Unexpected model type in save hook: {model.__class__}")
                if weights:
                    weights.pop()
            FluxControlPipeline.save_lora_weights(
                output_dir,
                transformer_lora_layers=transformer_lora_layers_to_save,
            )

    def load_model_hook(models, input_dir):
        transformer_to_load = None
        while len(models) > 0:
            model = models.pop()
            if isinstance(unwrap_model(accelerator, model), transformer_cls):
                transformer_to_load = model
            else:
                raise ValueError(f"Unexpected model type in load hook: {model.__class__}")
        lora_state_dict = FluxControlPipeline.lora_state_dict(input_dir)
        transformer_lora_state_dict = {
            key.replace("transformer.", ""): value
            for key, value in lora_state_dict.items()
            if key.startswith("transformer.") and "lora" in key
        }
        incompatible_keys = set_peft_model_state_dict(
            transformer_to_load,
            transformer_lora_state_dict,
            adapter_name="default",
        )
        if incompatible_keys is not None and getattr(incompatible_keys, "unexpected_keys", None):
            print(f"Unexpected LoRA keys while resuming: {incompatible_keys.unexpected_keys}")

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)
    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer,
        optimizer,
        train_dataloader,
        lr_scheduler,
    )

    print("Encoding constant prompt embeddings once...")
    text_encoding_pipeline = FluxControlPipeline.from_pretrained(
        cfg.model_id,
        transformer=None,
        vae=None,
        torch_dtype=weight_dtype,
        token=hf_token,
    )
    text_encoding_pipeline = text_encoding_pipeline.to(accelerator.device)
    text_encoding_pipeline.set_progress_bar_config(disable=True)
    with torch.no_grad():
        prompt_embeds_base, pooled_prompt_embeds_base, text_ids = (
            text_encoding_pipeline.encode_prompt(cfg.instance_prompt, prompt_2=None)
        )
    prompt_embeds_base = prompt_embeds_base.detach().to("cpu")
    pooled_prompt_embeds_base = pooled_prompt_embeds_base.detach().to("cpu")
    text_ids = text_ids.detach().to(accelerator.device)
    del text_encoding_pipeline
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == timestep).nonzero().item() for timestep in timesteps]
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    dry_batch = next(iter(train_dataloader))
    with torch.no_grad():
        dry_pixel_latents = encode_images(dry_batch["pixel_values"], vae, weight_dtype)
        dry_control_latents = encode_images(
            dry_batch["conditioning_pixel_values"],
            vae,
            weight_dtype,
        )
        dry_noise = torch.randn_like(
            dry_pixel_latents,
            device=accelerator.device,
            dtype=weight_dtype,
        )
        dry_timestep = noise_scheduler_copy.timesteps[: dry_pixel_latents.shape[0]].to(
            dry_pixel_latents.device
        )
        dry_sigmas = get_sigmas(
            dry_timestep,
            n_dim=dry_pixel_latents.ndim,
            dtype=dry_pixel_latents.dtype,
        )
        dry_noisy = (1.0 - dry_sigmas) * dry_pixel_latents + dry_sigmas * dry_noise
        dry_concatenated = torch.cat([dry_noisy, dry_control_latents], dim=1)
        dry_packed = FluxControlPipeline._pack_latents(
            dry_concatenated,
            batch_size=dry_pixel_latents.shape[0],
            num_channels_latents=dry_concatenated.shape[1],
            height=dry_concatenated.shape[2],
            width=dry_concatenated.shape[3],
        )
    packed_dim = int(dry_packed.shape[-1])
    expected_dim = int(unwrap_model(accelerator, transformer).x_embedder.in_features)
    print(f"dry-run packed hidden_states last dim: {packed_dim}")
    print(f"transformer.x_embedder.in_features: {expected_dim}")
    if packed_dim != expected_dim:
        raise RuntimeError(
            "The loaded transformer does not appear to accept concatenated target+control latents. "
            "Do not expand channels blindly; verify model_id and Diffusers pipeline. "
            f"packed hidden dim={packed_dim}, x_embedder.in_features={expected_dim}."
        )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg.gradient_accumulation_steps)
    num_train_epochs = math.ceil(cfg.max_train_steps / num_update_steps_per_epoch)
    global_step = 0
    first_epoch = 0
    resume_checkpoint = resolve_resume_checkpoint(cfg, run_path)
    if resume_checkpoint is not None:
        print(f"Resuming from checkpoint: {resume_checkpoint}")
        accelerator.load_state(str(resume_checkpoint))
        global_step = int(Path(resume_checkpoint).name.split("-")[-1])
        first_epoch = global_step // num_update_steps_per_epoch
    elif cfg.resume_from_latest_checkpoint or cfg.resume_checkpoint_path:
        print("No checkpoint found; starting cleanly.")

    run_metadata = load_existing_metadata(run_path)
    run_metadata.update(
        {
            "optimizer": optimizer_name,
            "resume_checkpoint_path": cfg.resume_checkpoint_path,
            "resumed_checkpoint_path": str(resume_checkpoint) if resume_checkpoint else None,
            "transformer.config.in_channels": original_in_channels,
            "transformer.x_embedder.in_features": original_x_embedder_in_features,
            "vae.config.latent_channels": latent_channels,
            "original_x_embedder_shape": list(original_x_embedder_shape),
            "dry_run_packed_hidden_dim": packed_dim,
            "channel_expansion_performed": False,
            "num_train_epochs": num_train_epochs,
        }
    )
    write_json(run_path / "metadata.json", _jsonable(run_metadata))

    progress_bar = tqdm(
        range(cfg.max_train_steps),
        initial=global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )
    last_loss = None
    for _epoch in range(first_epoch, num_train_epochs):
        transformer.train()
        for batch in train_dataloader:
            with accelerator.accumulate(transformer):
                pixel_latents = encode_images(batch["pixel_values"], vae, weight_dtype)
                control_latents = encode_images(
                    batch["conditioning_pixel_values"],
                    vae,
                    weight_dtype,
                )
                bsz = pixel_latents.shape[0]
                noise = torch.randn_like(
                    pixel_latents,
                    device=accelerator.device,
                    dtype=weight_dtype,
                )
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=cfg.weighting_scheme,
                    batch_size=bsz,
                    logit_mean=cfg.logit_mean,
                    logit_std=cfg.logit_std,
                    mode_scale=cfg.mode_scale,
                )
                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(device=pixel_latents.device)
                sigmas = get_sigmas(timesteps, n_dim=pixel_latents.ndim, dtype=pixel_latents.dtype)
                noisy_model_input = (1.0 - sigmas) * pixel_latents + sigmas * noise
                concatenated = torch.cat([noisy_model_input, control_latents], dim=1)
                packed = FluxControlPipeline._pack_latents(
                    concatenated,
                    batch_size=bsz,
                    num_channels_latents=concatenated.shape[1],
                    height=concatenated.shape[2],
                    width=concatenated.shape[3],
                )
                latent_image_ids = FluxControlPipeline._prepare_latent_image_ids(
                    bsz,
                    concatenated.shape[2] // 2,
                    concatenated.shape[3] // 2,
                    accelerator.device,
                    weight_dtype,
                )
                guidance_vec = None
                if unwrap_model(accelerator, transformer).config.guidance_embeds:
                    guidance_vec = torch.full(
                        (bsz,),
                        cfg.guidance_scale,
                        device=accelerator.device,
                        dtype=weight_dtype,
                    )
                prompt_embeds, pooled_prompt_embeds = repeat_prompt_tensors(
                    prompt_embeds_base,
                    pooled_prompt_embeds_base,
                    bsz,
                    accelerator.device,
                    weight_dtype,
                )
                model_pred = transformer(
                    hidden_states=packed,
                    timestep=timesteps / 1000,
                    guidance=guidance_vec,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    return_dict=False,
                )[0]
                model_pred = FluxControlPipeline._unpack_latents(
                    model_pred,
                    height=noisy_model_input.shape[2] * vae_scale_factor,
                    width=noisy_model_input.shape[3] * vae_scale_factor,
                    vae_scale_factor=vae_scale_factor,
                )
                weighting = compute_loss_weighting_for_sd3(
                    weighting_scheme=cfg.weighting_scheme,
                    sigmas=sigmas,
                )
                target = noise - pixel_latents
                loss = torch.mean(
                    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(
                        target.shape[0],
                        -1,
                    ),
                    1,
                ).mean()
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        [param for param in transformer.parameters() if param.requires_grad],
                        cfg.max_grad_norm,
                    )
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                progress_bar.update(1)
                last_loss = loss.detach().item()
                progress_bar.set_postfix(loss=last_loss, lr=lr_scheduler.get_last_lr()[0])
                if accelerator.is_main_process and global_step % cfg.checkpointing_steps == 0:
                    save_path = run_path / f"checkpoint-{global_step}"
                    accelerator.save_state(str(save_path))
                    prune_checkpoints(run_path, cfg.checkpoints_total_limit)
                    print(f"Saved checkpoint: {save_path}")
            if global_step >= cfg.max_train_steps:
                break
        if global_step >= cfg.max_train_steps:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        transformer_to_save = unwrap_model(accelerator, transformer)
        transformer_lora_layers = get_peft_model_state_dict(transformer_to_save)
        FluxControlPipeline.save_lora_weights(
            str(run_path),
            transformer_lora_layers=transformer_lora_layers,
        )
        final_metadata = load_existing_metadata(run_path)
        final_metadata.update(
            {
                "completed_steps": global_step,
                "last_loss": last_loss,
                "final_lora_dir": str(run_path),
            }
        )
        write_json(run_path / "metadata.json", _jsonable(final_metadata))
        print(f"Saved final LoRA to {run_path}")

    accelerator.end_training()
    del transformer, vae
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    free_memory()
    return run_path


def run_pre_training_validation(cfg: FluxCannyLoraConfig, run_dir: str | Path) -> Path | None:
    """Run the configured validation smoke before expensive training starts."""

    if not cfg.run_pre_training_validation:
        return None
    if not cfg.run_inference_sanity:
        print("Pre-training validation skipped because run_inference_sanity is false.")
        return None
    if cfg.validation_backend == "lanpaint":
        from painting_inpaint.training.flux_canny_lanpaint import (
            run_flux_canny_lanpaint_validation,
        )

        return run_flux_canny_lanpaint_validation(
            cfg,
            run_dir,
            stage="validation_pre_training",
            require_lora=False,
        )
    print(f"Pre-training validation skipped for backend: {cfg.validation_backend}")
    return None


def run_inference_sanity(cfg: FluxCannyLoraConfig, run_dir: str | Path) -> Path | None:
    """Run optional post-training inference sanity check."""

    import os

    import torch
    from diffusers import FluxControlPipeline
    from huggingface_hub import get_token

    run_path = Path(run_dir)
    lora_dir = Path(cfg.inference_lora_dir) if cfg.inference_lora_dir is not None else run_path
    if not cfg.run_inference_sanity:
        return None
    if not (lora_dir / "pytorch_lora_weights.safetensors").exists():
        print(f"No final LoRA found under {lora_dir}; inference sanity skipped.")
        return None
    if cfg.validation_backend == "lanpaint":
        from painting_inpaint.training.flux_canny_lanpaint import (
            run_flux_canny_lanpaint_validation,
        )

        return run_flux_canny_lanpaint_validation(
            cfg,
            run_path,
            stage="inference_sanity",
            lora_dir=lora_dir,
            require_lora=True,
        )

    hf_token = os.environ.get("HF_TOKEN") or get_token()
    if hf_token is None:
        print(
            "HF token not found. Run huggingface-cli login or set HF_TOKEN before "
            "loading gated FLUX models."
        )

    dataset_dir = resolve_dataset_dir(cfg)
    dataset = DurerCannyDataset(
        list_image_paths(dataset_dir),
        prompt=cfg.instance_prompt,
        resolution=cfg.resolution,
        canny_low=cfg.canny_low_threshold,
        canny_high=cfg.canny_high_threshold,
        random_crop=True,
    )
    sanity_dir = (
        lora_dir / "inference_sanity"
        if cfg.inference_lora_dir
        else run_path / "inference_sanity"
    )
    sanity_dir.mkdir(parents=True, exist_ok=True)
    sample = dataset.make_sample_images(0, rng=random.Random(cfg.validation_seed))
    sample["crop"].save(sanity_dir / "target_crop.png")
    sample["canny"].save(sanity_dir / "canny.png")

    pipe = FluxControlPipeline.from_pretrained(
        cfg.model_id,
        torch_dtype=torch.bfloat16,
        token=hf_token,
    )
    pipe.load_lora_weights(str(lora_dir))
    pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=False)
    generator = (
        torch.Generator(device="cuda").manual_seed(cfg.validation_seed)
        if torch.cuda.is_available()
        else None
    )
    output = pipe(
        prompt=cfg.instance_prompt,
        control_image=sample["canny"],
        height=cfg.resolution,
        width=cfg.resolution,
        num_inference_steps=cfg.validation_num_inference_steps,
        guidance_scale=cfg.guidance_scale,
        generator=generator,
    ).images[0]
    output.save(sanity_dir / "output.png")
    comparison = Image.new("RGB", (cfg.resolution * 3, cfg.resolution), "white")
    comparison.paste(sample["crop"], (0, 0))
    comparison.paste(sample["canny"], (cfg.resolution, 0))
    comparison.paste(output, (cfg.resolution * 2, 0))
    comparison.save(sanity_dir / "comparison.png")
    print(f"Saved inference sanity images to {sanity_dir}")
    return sanity_dir


def run_flux_canny_lora_training(
    config_path: str | Path,
    *,
    run: bool = False,
    run_root: str | Path | None = None,
    run_dir: str | Path | None = None,
    resume_checkpoint: str | Path | None = None,
    resume_from_latest: bool | None = None,
    run_inference: bool | None = None,
) -> FluxCannyLoraRuntime:
    """Create a run folder and optionally execute full training."""

    cfg = load_flux_canny_lora_config(config_path)
    if resume_checkpoint is not None:
        cfg = replace(cfg, resume_checkpoint_path=str(resume_checkpoint))
    if resume_from_latest is not None:
        cfg = replace(cfg, resume_from_latest_checkpoint=resume_from_latest)
    if run_inference is not None:
        cfg = replace(cfg, run_inference_sanity=run_inference)

    runtime = create_flux_canny_lora_run_files(
        cfg,
        run_root=run_root,
        run_dir=run_dir,
        source_config=config_path,
        preflight_only=not run,
    )
    if run:
        run_pre_training_validation(cfg, runtime.run_dir)
        train_flux_canny_lora(cfg, runtime.run_dir)
        run_inference_sanity(cfg, runtime.run_dir)
    return runtime


__all__ = [
    "DurerCannyDataset",
    "FluxCannyLoraConfig",
    "FluxCannyLoraRuntime",
    "checkpoint_dirs",
    "checkpoint_step",
    "collate_fn",
    "conditional_resize",
    "config_to_yaml_payload",
    "create_flux_canny_lora_run_files",
    "crop_image",
    "export_debug_samples",
    "flux_canny_lora_config_from_dict",
    "latest_checkpoint",
    "list_image_paths",
    "load_existing_metadata",
    "load_flux_canny_lora_config",
    "make_canny_rgb",
    "make_side_by_side",
    "prune_checkpoints",
    "resolve_dataset_dir",
    "resolve_resume_checkpoint",
    "run_flux_canny_lora_training",
    "run_inference_sanity",
    "run_pre_training_validation",
    "tensor_to_pil",
    "train_flux_canny_lora",
]
