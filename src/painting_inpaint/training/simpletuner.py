"""SimpleTuner Flux LoRA helpers.

The helpers in this module generate runtime configuration files for Colab
training and create lightweight visual audits of the training image
preprocessing. They intentionally avoid importing SimpleTuner so the base
project environment stays small.
"""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps

from painting_inpaint.experiments.manifest import (
    create_run_dir,
    package_versions,
    write_json,
    write_manifest,
)
from painting_inpaint.paths import resolve_data_path, runs_root
from painting_inpaint.training.config import load_yaml
from painting_inpaint.training.datasets import scan_image_folder
from painting_inpaint.visualization import fit_to_max_side

DEFAULT_SIMPLETUNER_REPO = "https://github.com/bghira/SimpleTuner.git"
DEFAULT_SIMPLETUNER_REF = "5111eaf0e260787a80e38d0f4ba7ca0f3a471949"
SENSITIVE_KEYS = {"token", "hf_token", "hub_token", "password", "secret", "api_key"}


@dataclass(frozen=True)
class SimpleTunerRuntime:
    """Runtime file layout for one SimpleTuner training launch."""

    run_dir: Path
    config_dir: Path
    simpletuner_config_path: Path
    data_backend_config_path: Path
    trainer_command_path: Path
    manifest_path: Path
    command: list[str]
    cwd: Path
    output_dir: Path


@dataclass(frozen=True)
class AuditResult:
    """Summary of a generated dataset audit."""

    audit_dir: Path
    manifest_path: Path
    counts_path: Path
    contact_sheet_path: Path
    image_count: int
    processed_dir: Path


def load_simpletuner_config(path: str | Path) -> dict[str, Any]:
    """Load a project-level SimpleTuner training config."""

    cfg = load_yaml(path)
    if "dataset" not in cfg:
        raise ValueError("SimpleTuner config must define a dataset section.")
    if "training" not in cfg:
        raise ValueError("SimpleTuner config must define a training section.")
    return cfg


def _redact_value(value: Any) -> Any:
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            lowered = str(key).lower()
            if any(marker in lowered for marker in SENSITIVE_KEYS):
                redacted[key] = "[REDACTED]"
            else:
                redacted[key] = _redact_value(item)
        return redacted
    if isinstance(value, list):
        return [_redact_value(item) for item in value]
    return value


def _round_to_multiple(value: float | int, multiple: int) -> int:
    rounded = round(float(value) / multiple) * multiple
    return max(int(rounded), multiple)


def _aspect_ratio(width: int, height: int, rounding: int) -> float:
    return round(width / height, rounding)


def _resolve_source_dir(dataset_config: dict[str, Any]) -> Path:
    source_dir = resolve_data_path(dataset_config.get("source_dir"))
    if source_dir is None:
        raise ValueError("Dataset config must define source_dir.")
    return source_dir.resolve()


def _build_data_backend_config(
    cfg: dict[str, Any],
    *,
    run_dir: Path,
) -> list[dict[str, Any]]:
    dataset = cfg["dataset"]
    training = cfg["training"]
    source_dir = _resolve_source_dir(dataset)
    cache_root = run_dir / "simpletuner_cache"
    backend = {
        "id": dataset.get("id", "curated_durer"),
        "type": "local",
        "instance_data_dir": str(source_dir),
        "caption_strategy": dataset.get("caption_strategy", "instanceprompt"),
        "instance_prompt": dataset.get("trigger_word", dataset.get("instance_prompt", "")),
        "prepend_instance_prompt": bool(dataset.get("prepend_instance_prompt", False)),
        "only_instance_prompt": bool(dataset.get("only_instance_prompt", False)),
        "crop": bool(dataset.get("crop", True)),
        "crop_style": dataset.get("crop_style", "center"),
        "crop_aspect": dataset.get("crop_aspect", "preserve"),
        "resolution": int(dataset.get("resolution", training.get("resolution", 1024))),
        "resolution_type": dataset.get(
            "resolution_type",
            training.get("resolution_type", "pixel_area"),
        ),
        "metadata_backend": dataset.get("metadata_backend", "discovery"),
        "minimum_image_size": dataset.get("minimum_image_size", 0),
        "probability": float(dataset.get("probability", 1.0)),
        "repeats": int(dataset.get("repeats", 1)),
        "cache_dir_vae": str(cache_root / "vae"),
        "text_embeds": "curated_durer_text_embeds",
    }
    if dataset.get("maximum_image_size") is not None:
        backend["maximum_image_size"] = dataset["maximum_image_size"]
    if dataset.get("target_downsample_size") is not None:
        backend["target_downsample_size"] = dataset["target_downsample_size"]
    if dataset.get("max_upscale_threshold") is not None:
        backend["max_upscale_threshold"] = dataset["max_upscale_threshold"]

    text_backend = {
        "id": "curated_durer_text_embeds",
        "dataset_type": "text_embeds",
        "default": True,
        "type": "local",
        "cache_dir": str(cache_root / "text"),
    }
    return [backend, text_backend]


def _build_simpletuner_config(
    cfg: dict[str, Any],
    *,
    run_dir: Path,
    data_backend_config_path: Path,
) -> dict[str, Any]:
    training = cfg["training"]
    model = cfg.get("model", {})
    validation = cfg.get("validation", {})
    output_dir = run_dir / "trainer_output"

    simpletuner_config: dict[str, Any] = {
        "data_backend_config": str(data_backend_config_path),
        "output_dir": str(output_dir),
        "model_family": model.get("family", "flux"),
        "model_flavour": model.get("flavour", "dev"),
        "model_type": model.get("type", "lora"),
        "pretrained_model_name_or_path": model.get(
            "pretrained_model_name_or_path",
            "black-forest-labs/FLUX.1-dev",
        ),
        "aspect_bucket_rounding": int(training.get("aspect_bucket_rounding", 2)),
        "aspect_bucket_alignment": int(training.get("aspect_bucket_alignment", 64)),
        "base_model_precision": training.get("base_model_precision", "int8-quanto"),
        "quantize_via": training.get("quantize_via", "cpu"),
        "lora_type": training.get("lora_type", "standard"),
        "lora_rank": int(training.get("lora_rank", 32)),
        "lora_alpha": int(training.get("lora_alpha", 32)),
        "lora_dropout": float(training.get("lora_dropout", 0.05)),
        "train_batch_size": int(training.get("train_batch_size", 1)),
        "gradient_accumulation_steps": int(training.get("gradient_accumulation_steps", 4)),
        "max_train_steps": int(training.get("max_train_steps", 3000)),
        "num_train_epochs": int(training.get("num_train_epochs", 0)),
        "learning_rate": training.get("learning_rate", "1e-4"),
        "lr_scheduler": training.get("lr_scheduler", "constant"),
        "lr_warmup_steps": int(training.get("lr_warmup_steps", 100)),
        "optimizer": training.get("optimizer", "adamw_bf16"),
        "mixed_precision": training.get("mixed_precision", "bf16"),
        "gradient_checkpointing": bool(training.get("gradient_checkpointing", True)),
        "checkpoint_step_interval": int(training.get("checkpoint_step_interval", 500)),
        "checkpoints_total_limit": int(training.get("checkpoints_total_limit", 3)),
        "report_to": training.get("report_to", "tensorboard"),
        "tracker_project_name": training.get("tracker_project_name", "painting-inpaint-lora"),
        "tracker_run_name": training.get("tracker_run_name", cfg.get("name", "flux1_dev_lora")),
        "push_to_hub": bool(training.get("push_to_hub", False)),
        "push_checkpoints_to_hub": bool(training.get("push_checkpoints_to_hub", False)),
        "seed": int(training.get("seed", 20260527)),
        "vae_batch_size": int(training.get("vae_batch_size", 1)),
        "disable_bucket_pruning": bool(training.get("disable_bucket_pruning", True)),
        "validation_prompt_library": bool(validation.get("prompt_library", False)),
        "validation_steps": int(validation.get("steps", 0)),
    }

    if validation.get("disable", True):
        simpletuner_config["validation_disable"] = True
    else:
        simpletuner_config.update(
            {
                "validation_prompt": validation.get(
                    "prompt",
                    cfg["dataset"].get("trigger_word", ""),
                ),
                "validation_seed": int(validation.get("seed", training.get("seed", 20260527))),
                "validation_resolution": validation.get("resolution", "1024x1024"),
                "validation_guidance": float(validation.get("guidance", 4.0)),
                "validation_num_inference_steps": int(validation.get("num_inference_steps", 16)),
            }
        )

    for key, value in training.get("extra_args", {}).items():
        simpletuner_config[key] = value

    resume_from_checkpoint = training.get("resume_from_checkpoint")
    if resume_from_checkpoint:
        simpletuner_config["resume_from_checkpoint"] = str(resume_from_checkpoint)
        simpletuner_config["delete_invalid_checkpoints"] = bool(
            training.get("delete_invalid_checkpoints", False)
        )

    return simpletuner_config


def create_simpletuner_run_files(
    cfg: dict[str, Any],
    *,
    run_root: str | Path | None = None,
    run_dir: str | Path | None = None,
) -> SimpleTunerRuntime:
    """Create a run folder containing SimpleTuner runtime files."""

    if run_root is not None and run_dir is not None:
        raise ValueError("Pass either run_root or run_dir, not both.")

    method_name = cfg.get("name", "flux1_dev_curated_durer_simpletuner_lora")
    if run_dir is not None:
        run_dir = Path(run_dir).expanduser()
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        root = Path(run_root).expanduser() if run_root is not None else runs_root() / "training"
        run_dir = create_run_dir(method_name, root=root)
    config_dir = run_dir / "simpletuner"
    config_dir.mkdir(parents=True, exist_ok=True)

    data_backend_config_path = config_dir / "multidatabackend.json"
    simpletuner_config_path = config_dir / "config.json"

    data_backend_config = _build_data_backend_config(cfg, run_dir=run_dir)
    simpletuner_config = _build_simpletuner_config(
        cfg,
        run_dir=run_dir,
        data_backend_config_path=data_backend_config_path,
    )
    data_backend_config_path.write_text(
        json.dumps(data_backend_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_json(simpletuner_config_path, simpletuner_config)

    command = ["simpletuner", "train"]
    trainer_command = {
        "command": command,
        "redacted_command": command,
        "redacted_display": " ".join(command),
        "cwd": str(config_dir),
    }
    trainer_command_path = write_json(config_dir / "trainer_command.json", trainer_command)

    output_dir = Path(simpletuner_config["output_dir"])
    manifest = {
        "name": cfg.get("name"),
        "method": "simpletuner_flux1_dev_lora",
        "model_id": simpletuner_config["pretrained_model_name_or_path"],
        "compatible_inference_model_id": cfg.get("model", {}).get(
            "compatible_inference_model_id",
            "black-forest-labs/FLUX.1-Fill-dev",
        ),
        "dataset": cfg["dataset"].get("name"),
        "source_dir": str(_resolve_source_dir(cfg["dataset"])),
        "trigger_word": cfg["dataset"].get("trigger_word"),
        "simpletuner_repo": cfg.get("simpletuner", {}).get("repo", DEFAULT_SIMPLETUNER_REPO),
        "simpletuner_ref": cfg.get("simpletuner", {}).get("ref", DEFAULT_SIMPLETUNER_REF),
        "simpletuner_config": str(simpletuner_config_path.relative_to(run_dir)),
        "data_backend_config": str(data_backend_config_path.relative_to(run_dir)),
        "trainer_command": str(trainer_command_path.relative_to(run_dir)),
        "trainer_output_dir": str(output_dir.relative_to(run_dir)),
        "environment": {
            "package_versions": package_versions(
                [
                    "accelerate",
                    "diffusers",
                    "peft",
                    "simpletuner",
                    "torch",
                    "transformers",
                ]
            )
        },
    }
    manifest_path = write_manifest(run_dir, _redact_value(manifest))

    return SimpleTunerRuntime(
        run_dir=run_dir,
        config_dir=config_dir,
        simpletuner_config_path=simpletuner_config_path,
        data_backend_config_path=data_backend_config_path,
        trainer_command_path=trainer_command_path,
        manifest_path=manifest_path,
        command=command,
        cwd=config_dir,
        output_dir=output_dir,
    )


def estimate_processed_sample(
    image_size: tuple[int, int],
    *,
    resolution: int = 1024,
    resolution_type: str = "pixel_area",
    aspect_bucket_rounding: int = 2,
    aspect_bucket_alignment: int = 64,
) -> dict[str, Any]:
    """Estimate SimpleTuner's deterministic target/intermediary sizes.

    This mirrors the relevant pixel-area math from SimpleTuner for visual audit
    purposes. It is not a substitute for SimpleTuner's own cache metadata.
    """

    if resolution_type != "pixel_area":
        raise ValueError("Only resolution_type='pixel_area' is supported by the audit preview.")
    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size: {image_size}")

    raw_aspect = width / height
    aspect = round(raw_aspect, aspect_bucket_rounding)
    target_edge = _round_to_multiple(resolution, aspect_bucket_alignment)
    target_width = _round_to_multiple(target_edge * math.sqrt(aspect), aspect_bucket_alignment)
    target_height = _round_to_multiple(target_edge / math.sqrt(aspect), aspect_bucket_alignment)

    if target_width < target_height:
        intermediary_width = target_width
        intermediary_height = int(intermediary_width / raw_aspect)
    else:
        intermediary_height = target_height
        intermediary_width = int(intermediary_height * raw_aspect)

    if target_width > intermediary_width or target_height > intermediary_height:
        if target_width > intermediary_width:
            width_diff = target_width - intermediary_width
            height_diff = int(width_diff / raw_aspect)
        else:
            height_diff = target_height - intermediary_height
            width_diff = int(height_diff * raw_aspect)
        intermediary_width += width_diff
        intermediary_height += height_diff

    left = max(0, (intermediary_width - target_width) // 2)
    top = max(0, (intermediary_height - target_height) // 2)
    crop_box = (left, top, left + target_width, top + target_height)
    resize_scale = intermediary_width / width

    return {
        "original_size": [width, height],
        "aspect_ratio": aspect,
        "target_size": [target_width, target_height],
        "intermediary_size": [intermediary_width, intermediary_height],
        "crop_box": list(crop_box),
        "resize_scale": resize_scale,
        "upscaled": resize_scale > 1.0,
    }


def _process_preview_image(image: Image.Image, estimate: dict[str, Any]) -> Image.Image:
    intermediary_size = tuple(estimate["intermediary_size"])
    crop_box = tuple(estimate["crop_box"])
    resized = ImageOps.exif_transpose(image.convert("RGB")).resize(
        intermediary_size,
        Image.Resampling.LANCZOS,
    )
    return resized.crop(crop_box)


def _make_audit_tile(original: Image.Image, processed: Image.Image, label: str) -> Image.Image:
    panel_width = 360
    label_height = 42
    gap = 8
    original_panel = fit_to_max_side(original, panel_width)
    processed_panel = fit_to_max_side(processed, panel_width)
    height = label_height + max(original_panel.height, processed_panel.height)
    tile = Image.new("RGB", (panel_width * 2 + gap, height), (248, 248, 248))
    draw = ImageDraw.Draw(tile)
    font = ImageFont.load_default()
    draw.text((4, 4), label[:70], fill=(20, 20, 20), font=font)
    tile.paste(original_panel, (0, label_height))
    tile.paste(processed_panel, (panel_width + gap, label_height))
    return tile


def _make_grid_sheet(tiles: list[Image.Image], columns: int = 2, padding: int = 12) -> Image.Image:
    if not tiles:
        raise ValueError("At least one tile is required.")
    tile_w = max(tile.width for tile in tiles)
    tile_h = max(tile.height for tile in tiles)
    rows = math.ceil(len(tiles) / columns)
    sheet = Image.new(
        "RGB",
        (columns * tile_w + (columns + 1) * padding, rows * tile_h + (rows + 1) * padding),
        (238, 238, 238),
    )
    for index, tile in enumerate(tiles):
        row = index // columns
        col = index % columns
        x = padding + col * (tile_w + padding)
        y = padding + row * (tile_h + padding)
        sheet.paste(tile, (x, y))
    return sheet


def create_dataset_audit(
    cfg: dict[str, Any],
    *,
    run_dir: str | Path,
    max_contact_sheet_images: int = 24,
) -> AuditResult:
    """Generate visual previews of SimpleTuner-style dataset preprocessing."""

    dataset = cfg["dataset"]
    training = cfg["training"]
    source_dir = _resolve_source_dir(dataset)
    audit_dir = Path(run_dir) / "dataset_audit"
    processed_dir = audit_dir / "processed_samples"
    if audit_dir.exists():
        shutil.rmtree(audit_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    records = scan_image_folder(source_dir)
    valid_records = [record for record in records if record.valid]
    resolution = int(dataset.get("resolution", training.get("resolution", 1024)))
    resolution_type = dataset.get(
        "resolution_type",
        training.get("resolution_type", "pixel_area"),
    )
    aspect_bucket_rounding = int(training.get("aspect_bucket_rounding", 2))
    aspect_bucket_alignment = int(training.get("aspect_bucket_alignment", 64))

    manifest_rows: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    tiles: list[Image.Image] = []

    for index, record in enumerate(valid_records, start=1):
        source_path = source_dir / record.relative_path
        with Image.open(source_path) as image:
            image = ImageOps.exif_transpose(image.convert("RGB"))
            estimate = estimate_processed_sample(
                image.size,
                resolution=resolution,
                resolution_type=resolution_type,
                aspect_bucket_rounding=aspect_bucket_rounding,
                aspect_bucket_alignment=aspect_bucket_alignment,
            )
            processed = _process_preview_image(image, estimate)

        aspect_key = f"{estimate['aspect_ratio']:.{aspect_bucket_rounding}f}"
        counts[aspect_key] = counts.get(aspect_key, 0) + 1
        processed_name = f"{index:05d}_{Path(record.relative_path).stem}.jpg"
        processed_path = processed_dir / processed_name
        processed.save(processed_path, quality=92)

        row = {
            "source": record.relative_path,
            "original_size": estimate["original_size"],
            "aspect_ratio": estimate["aspect_ratio"],
            "target_size": estimate["target_size"],
            "intermediary_size": estimate["intermediary_size"],
            "crop_box": estimate["crop_box"],
            "resize_scale": estimate["resize_scale"],
            "upscaled": estimate["upscaled"],
            "processed_preview": str(processed_path.relative_to(audit_dir)),
        }
        manifest_rows.append(row)

        if len(tiles) < max_contact_sheet_images:
            label = (
                f"{record.relative_path} | {record.width}x{record.height} -> "
                f"{estimate['target_size'][0]}x{estimate['target_size'][1]}"
            )
            with Image.open(source_path) as image:
                tiles.append(_make_audit_tile(image.convert("RGB"), processed, label))

    manifest = {
        "source_dir": str(source_dir),
        "resolution": resolution,
        "resolution_type": resolution_type,
        "crop": dataset.get("crop", True),
        "crop_style": dataset.get("crop_style", "center"),
        "crop_aspect": dataset.get("crop_aspect", "preserve"),
        "aspect_bucket_rounding": aspect_bucket_rounding,
        "aspect_bucket_alignment": aspect_bucket_alignment,
        "image_count": len(valid_records),
        "invalid_count": len(records) - len(valid_records),
        "invalid_images": [record.__dict__ for record in records if not record.valid],
        "images": manifest_rows,
    }
    manifest_path = write_json(audit_dir / "dataset_audit_manifest.json", manifest)
    counts_path = write_json(audit_dir / "bucket_or_aspect_counts.json", counts)

    if tiles:
        sheet = _make_grid_sheet(tiles)
    else:
        sheet = Image.new("RGB", (512, 128), (248, 248, 248))
        ImageDraw.Draw(sheet).text((12, 12), "No valid images found.", fill=(20, 20, 20))
    contact_sheet_path = audit_dir / "contact_sheet_overall.png"
    sheet.save(contact_sheet_path)

    return AuditResult(
        audit_dir=audit_dir,
        manifest_path=manifest_path,
        counts_path=counts_path,
        contact_sheet_path=contact_sheet_path,
        image_count=len(valid_records),
        processed_dir=processed_dir,
    )
