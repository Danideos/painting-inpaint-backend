"""MetaCentrum validation sweep for trained FLUX.1-Canny LoRA checkpoints."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from painting_inpaint.experiments.manifest import package_versions, timestamp_utc, write_json
from painting_inpaint.paths import resolve_data_path, runs_root
from painting_inpaint.training.flux_canny_lanpaint import (
    clear_cuda_cache,
    inspect_flux_canny_lanpaint_pipeline,
    load_flux_canny_lanpaint_pipeline,
    load_flux_canny_lora_adapter,
    prepare_flux_canny_lanpaint_validation_inputs,
    remove_flux_canny_lora_adapter,
    run_flux_canny_lanpaint_prepared_validation,
)
from painting_inpaint.training.flux_canny_lora import (
    FluxCannyLoraConfig,
    checkpoint_dirs,
    checkpoint_step,
    load_flux_canny_lora_config,
)

LORA_WEIGHTS_NAME = "pytorch_lora_weights.safetensors"
SUPPORTED_RANKS = {16, 32, 64}


@dataclass(frozen=True)
class LoraValidationCandidate:
    """One checkpoint or final LoRA to validate."""

    label: str
    checkpoint: str
    step: int
    lora_dir: Path
    weights_path: Path

    def to_manifest(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "checkpoint": self.checkpoint,
            "step": self.step,
            "lora_dir": str(self.lora_dir),
            "weights_path": str(self.weights_path),
        }


def noise_strength_slug(strength: float) -> str:
    """Return a stable folder slug for a partial-noise strength."""

    if not (0.0 < float(strength) <= 1.0):
        raise ValueError("partial-noise strength must be in the interval (0.0, 1.0].")
    return f"noise_{round(float(strength) * 100):03d}"


def discover_lora_candidates(run_dir: str | Path) -> list[LoraValidationCandidate]:
    """Return checkpoint LoRAs sorted numerically, followed by the final LoRA."""

    run_path = Path(run_dir).expanduser().resolve()
    candidates: list[LoraValidationCandidate] = []
    for checkpoint_dir in checkpoint_dirs(run_path):
        weights_path = checkpoint_dir / LORA_WEIGHTS_NAME
        step = checkpoint_step(checkpoint_dir)
        if step is None or not weights_path.exists():
            continue
        candidates.append(
            LoraValidationCandidate(
                label=f"checkpoint_{step}",
                checkpoint=checkpoint_dir.name,
                step=step,
                lora_dir=checkpoint_dir,
                weights_path=weights_path,
            )
        )

    final_weights = run_path / LORA_WEIGHTS_NAME
    if final_weights.exists():
        candidates.append(
            LoraValidationCandidate(
                label="final",
                checkpoint="final",
                step=999_999_999,
                lora_dir=run_path,
                weights_path=final_weights,
            )
        )
    return candidates


def discover_latest_complete_rank_run(training_root: str | Path, rank: int) -> Path:
    """Find the newest complete training run for a LoRA rank."""

    root = Path(training_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Missing training root: {root}")
    pattern = f"*flux1_canny_dev_curated_durer_lora_rank{rank}"
    candidates = [
        path
        for path in root.glob(pattern)
        if path.is_dir()
        and (path / LORA_WEIGHTS_NAME).exists()
        and discover_lora_candidates(path)
    ]
    if not candidates:
        raise FileNotFoundError(f"No complete rank-{rank} FLUX-Canny LoRA run found under {root}")
    return sorted(candidates, key=lambda path: path.name)[-1].resolve()


def _load_raw_yaml(path: str | Path) -> dict[str, Any]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Validation sweep config must be a mapping: {path}")
    return raw


def load_flux_canny_validation_sweep_config(path: str | Path, *, rank: int) -> FluxCannyLoraConfig:
    """Load the base validation config and apply the requested rank."""

    if int(rank) not in SUPPORTED_RANKS:
        raise ValueError(f"Unsupported rank {rank}; expected one of {sorted(SUPPORTED_RANKS)}")
    return replace(load_flux_canny_lora_config(path), rank=int(rank))


def partial_noise_strengths_from_config(path: str | Path) -> tuple[float, ...]:
    """Read the sweep partial-noise strengths from YAML."""

    raw = _load_raw_yaml(path)
    sweep = raw.get("sweep", {})
    validation = raw.get("validation", {})
    strengths = sweep.get("partial_noise_strengths", validation.get("partial_noise_strengths"))
    if strengths is None:
        strengths = [validation.get("partial_noise_strength", 1.0)]
    if not isinstance(strengths, list | tuple) or not strengths:
        raise ValueError("sweep.partial_noise_strengths must be a non-empty list.")
    parsed = tuple(float(value) for value in strengths)
    for strength in parsed:
        noise_strength_slug(strength)
    return parsed


def required_rosary_files_from_config(path: str | Path) -> tuple[str, ...]:
    """Read required Rosary files from YAML."""

    raw = _load_raw_yaml(path)
    rosary = raw.get("rosary", {})
    required = rosary.get("required_files", ())
    if not isinstance(required, list | tuple):
        raise ValueError("rosary.required_files must be a list.")
    return tuple(str(item) for item in required)


def verify_required_rosary_files(required_files: tuple[str, ...]) -> list[dict[str, Any]]:
    """Resolve required Rosary data files and fail with a clear missing-file message."""

    records: list[dict[str, Any]] = []
    missing: list[str] = []
    for raw_path in required_files:
        resolved = resolve_data_path(raw_path)
        if resolved is None:
            missing.append(raw_path)
            continue
        resolved = resolved.resolve()
        if not resolved.exists():
            missing.append(str(resolved))
            continue
        records.append(
            {
                "configured_path": raw_path,
                "resolved_path": str(resolved),
                "size_bytes": resolved.stat().st_size,
            }
        )
    if missing:
        raise FileNotFoundError("Missing required Rosary file(s):\n" + "\n".join(missing))
    return records


def planned_validation_records(
    *,
    rank: int,
    loras: list[LoraValidationCandidate],
    strengths: tuple[float, ...],
    windows: tuple[dict[str, Any], ...],
) -> list[dict[str, Any]]:
    """Return the deterministic validation matrix without loading any models."""

    records: list[dict[str, Any]] = []
    for lora in loras:
        for strength in strengths:
            noise_slug = noise_strength_slug(strength)
            for window in windows:
                window_id = str(window["window_id"])
                records.append(
                    {
                        "rank": int(rank),
                        "checkpoint": lora.checkpoint,
                        "lora_label": lora.label,
                        "lora_dir": str(lora.lora_dir),
                        "weights_path": str(lora.weights_path),
                        "partial_noise_strength": float(strength),
                        "noise_slug": noise_slug,
                        "window_id": window_id,
                        "relative_output_dir": str(Path(lora.label) / noise_slug / window_id),
                    }
                )
    return records


def make_validation_run_dir(run_root: str | Path | None, *, rank: int) -> Path:
    """Create the rank validation run directory."""

    root = Path(run_root).expanduser() if run_root is not None else runs_root() / "validation"
    run_dir = root / f"flux1_canny_rosary_lanpaint_rank{rank}_{timestamp_utc()}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _base_manifest(
    *,
    cfg: FluxCannyLoraConfig,
    config_path: Path,
    rank_run_dir: Path,
    rank: int,
    loras: list[LoraValidationCandidate],
    strengths: tuple[float, ...],
    rosary_files: list[dict[str, Any]],
) -> dict[str, Any]:
    planned_records = planned_validation_records(
        rank=rank,
        loras=loras,
        strengths=strengths,
        windows=cfg.validation_windows,
    )
    return {
        "completed": False,
        "config_path": str(config_path),
        "rank": int(rank),
        "rank_run_dir": str(rank_run_dir),
        "model_id": cfg.model_id,
        "prompt": cfg.validation_prompt,
        "canny_low_threshold": cfg.canny_low_threshold,
        "canny_high_threshold": cfg.canny_high_threshold,
        "num_inference_steps": cfg.validation_num_inference_steps,
        "partial_noise_strengths": [float(value) for value in strengths],
        "windows": [dict(window) for window in cfg.validation_windows],
        "lora_candidates": [candidate.to_manifest() for candidate in loras],
        "rosary_files": rosary_files,
        "planned_record_count": len(planned_records),
        "planned_records": planned_records,
        "completed_records": [],
        "adapter_events": [],
        "pipeline_reload_count": 0,
        "package_versions": package_versions(
            ["torch", "diffusers", "transformers", "accelerate", "peft", "LanPaint"]
        ),
    }


def plan_flux_canny_lora_validation_sweep(
    *,
    config_path: str | Path,
    rank: int,
    training_root: str | Path | None = None,
    rank_run_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Build and validate the sweep plan without loading FLUX."""

    config = Path(config_path)
    cfg = load_flux_canny_validation_sweep_config(config, rank=rank)
    strengths = partial_noise_strengths_from_config(config)
    rosary_files = verify_required_rosary_files(required_rosary_files_from_config(config))
    root = (
        Path(training_root).expanduser()
        if training_root is not None
        else runs_root() / "training"
    )
    run_dir = (
        Path(rank_run_dir).expanduser().resolve()
        if rank_run_dir is not None
        else discover_latest_complete_rank_run(root, rank)
    )
    loras = discover_lora_candidates(run_dir)
    if not loras:
        raise FileNotFoundError(f"No LoRA weights found under rank run: {run_dir}")
    if not (run_dir / LORA_WEIGHTS_NAME).exists():
        raise FileNotFoundError(f"Missing final LoRA weights under rank run: {run_dir}")
    return _base_manifest(
        cfg=cfg,
        config_path=config,
        rank_run_dir=run_dir,
        rank=rank,
        loras=loras,
        strengths=strengths,
        rosary_files=rosary_files,
    )


def run_flux_canny_lora_validation_sweep(
    *,
    config_path: str | Path,
    rank: int,
    training_root: str | Path | None = None,
    rank_run_dir: str | Path | None = None,
    run_root: str | Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run or dry-run the full rank/checkpoint/noise/window validation sweep."""

    plan = plan_flux_canny_lora_validation_sweep(
        config_path=config_path,
        rank=rank,
        training_root=training_root,
        rank_run_dir=rank_run_dir,
    )
    if dry_run:
        return {**plan, "dry_run": True}

    config = Path(config_path)
    cfg = load_flux_canny_validation_sweep_config(config, rank=rank)
    strengths = partial_noise_strengths_from_config(config)
    loras = [
        LoraValidationCandidate(
            label=str(item["label"]),
            checkpoint=str(item["checkpoint"]),
            step=int(item["step"]),
            lora_dir=Path(item["lora_dir"]),
            weights_path=Path(item["weights_path"]),
        )
        for item in plan["lora_candidates"]
    ]
    run_dir = make_validation_run_dir(run_root, rank=rank)
    manifest = {
        **plan,
        "dry_run": False,
        "validation_run_dir": str(run_dir),
        "inputs_dir": str(run_dir / "inputs"),
    }
    manifest_path = run_dir / "validation_manifest.json"
    write_json(manifest_path, manifest)

    prepared_windows = prepare_flux_canny_lanpaint_validation_inputs(cfg, run_dir / "inputs")
    pipe, torch_dtype = load_flux_canny_lanpaint_pipeline(cfg)
    inspect_flux_canny_lanpaint_pipeline(pipe, cfg, run_dir, lora_enabled=False)

    try:
        for lora_index, candidate in enumerate(loras):
            print(f"Loading LoRA {candidate.label}: {candidate.lora_dir}")
            load_info = load_flux_canny_lora_adapter(
                pipe,
                candidate.lora_dir,
                adapter_name=candidate.label,
            )
            manifest["adapter_events"].append({"event": "load", **load_info})
            write_json(manifest_path, manifest)

            for strength in strengths:
                cfg_for_strength = replace(cfg, validation_partial_noise_strength=float(strength))
                noise_slug = noise_strength_slug(strength)
                stage = f"{candidate.label}/{noise_slug}"
                stage_dir, window_results = run_flux_canny_lanpaint_prepared_validation(
                    pipe,
                    cfg_for_strength,
                    run_dir,
                    stage=stage,
                    prepared_windows=prepared_windows,
                    lora_dir=candidate.lora_dir,
                    lora_enabled=True,
                    torch_dtype=torch_dtype,
                    update_run_metadata=False,
                )
                for window_result in window_results:
                    manifest["completed_records"].append(
                        {
                            "rank": int(rank),
                            "checkpoint": candidate.checkpoint,
                            "lora_label": candidate.label,
                            "lora_dir": str(candidate.lora_dir),
                            "weights_path": str(candidate.weights_path),
                            "partial_noise_strength": float(strength),
                            "noise_slug": noise_slug,
                            "window_id": window_result["window_id"],
                            "stage_dir": str(stage_dir),
                            **window_result,
                        }
                    )
                write_json(manifest_path, manifest)

            cleanup = remove_flux_canny_lora_adapter(
                pipe,
                adapter_name=str(load_info["adapter_name"]),
                named_adapter=bool(load_info["named_adapter"]),
            )
            manifest["adapter_events"].append({"event": "cleanup", **cleanup})
            write_json(manifest_path, manifest)

            if not cleanup["removed"] and lora_index < len(loras) - 1:
                print("LoRA adapter cleanup unavailable; reloading FLUX pipeline before next LoRA.")
                del pipe
                clear_cuda_cache()
                pipe, torch_dtype = load_flux_canny_lanpaint_pipeline(cfg)
                inspect_flux_canny_lanpaint_pipeline(pipe, cfg, run_dir, lora_enabled=False)
                manifest["pipeline_reload_count"] = int(manifest["pipeline_reload_count"]) + 1
                write_json(manifest_path, manifest)

        manifest["completed"] = True
        manifest["completed_record_count"] = len(manifest["completed_records"])
        write_json(manifest_path, manifest)
        return manifest
    finally:
        try:
            del pipe
        except UnboundLocalError:
            pass
        clear_cuda_cache()


__all__ = [
    "LoraValidationCandidate",
    "discover_latest_complete_rank_run",
    "discover_lora_candidates",
    "load_flux_canny_validation_sweep_config",
    "noise_strength_slug",
    "partial_noise_strengths_from_config",
    "plan_flux_canny_lora_validation_sweep",
    "planned_validation_records",
    "required_rosary_files_from_config",
    "run_flux_canny_lora_validation_sweep",
    "verify_required_rosary_files",
]
