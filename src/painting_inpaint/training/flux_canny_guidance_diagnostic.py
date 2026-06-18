"""One-window final-LoRA guidance diagnostic for FLUX.1-Canny LanPaint."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from painting_inpaint.experiments.manifest import package_versions, timestamp_utc, write_json
from painting_inpaint.paths import runs_root
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
    load_flux_canny_lora_config,
)
from painting_inpaint.training.flux_canny_validation_sweep import (
    LORA_WEIGHTS_NAME,
    SUPPORTED_RANKS,
    discover_latest_complete_rank_run,
    noise_strength_slug,
    required_rosary_files_from_config,
    verify_required_rosary_files,
)

DEFAULT_DIAGNOSTIC_RANKS = (16, 32, 64)


@dataclass(frozen=True)
class FinalLoraDiagnosticCandidate:
    """One final LoRA selected for the guidance diagnostic."""

    rank: int
    label: str
    run_dir: Path
    lora_dir: Path
    weights_path: Path

    def to_manifest(self) -> dict[str, Any]:
        return {
            "rank": int(self.rank),
            "label": self.label,
            "checkpoint": "final",
            "run_dir": str(self.run_dir),
            "lora_dir": str(self.lora_dir),
            "weights_path": str(self.weights_path),
        }


def guidance_scale_slug(guidance_scale: float) -> str:
    """Return a stable folder slug for a guidance scale."""

    value = float(guidance_scale)
    if value < 0:
        raise ValueError("guidance_scale must be non-negative.")
    text = f"{value:g}".replace(".", "p")
    return f"guidance_{text}"


def _load_raw_yaml(path: str | Path) -> dict[str, Any]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Diagnostic config must be a mapping: {path}")
    return raw


def diagnostic_ranks_from_config(path: str | Path) -> tuple[int, ...]:
    """Read diagnostic ranks from YAML."""

    raw = _load_raw_yaml(path)
    diagnostic = raw.get("diagnostic", {})
    ranks = diagnostic.get("ranks", DEFAULT_DIAGNOSTIC_RANKS)
    if not isinstance(ranks, list | tuple) or not ranks:
        raise ValueError("diagnostic.ranks must be a non-empty list.")
    parsed = tuple(int(rank) for rank in ranks)
    unsupported = sorted(set(parsed) - SUPPORTED_RANKS)
    if unsupported:
        raise ValueError(f"Unsupported diagnostic rank(s): {unsupported}")
    return parsed


def load_final_guidance_diagnostic_config(path: str | Path) -> FluxCannyLoraConfig:
    """Load and validate the one-window diagnostic config."""

    cfg = load_flux_canny_lora_config(path)
    if len(cfg.validation_windows) != 1:
        raise ValueError("Final guidance diagnostic expects exactly one validation window.")
    noise_strength_slug(cfg.validation_partial_noise_strength)
    guidance_scale_slug(cfg.validation_guidance_scale)
    return cfg


def discover_final_lora_candidate(
    training_root: str | Path,
    rank: int,
) -> FinalLoraDiagnosticCandidate:
    """Find the newest complete run and final LoRA for one rank."""

    run_dir = discover_latest_complete_rank_run(training_root, rank)
    weights_path = run_dir / LORA_WEIGHTS_NAME
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing final LoRA weights: {weights_path}")
    return FinalLoraDiagnosticCandidate(
        rank=int(rank),
        label=f"rank{int(rank)}_final",
        run_dir=run_dir,
        lora_dir=run_dir,
        weights_path=weights_path,
    )


def make_final_guidance_diagnostic_run_dir(run_root: str | Path | None) -> Path:
    """Create the diagnostic run directory."""

    root = Path(run_root).expanduser() if run_root is not None else runs_root() / "validation"
    run_dir = root / f"flux1_canny_rosary_lanpaint_final_guidance15_left_{timestamp_utc()}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def planned_final_guidance_records(
    *,
    candidates: list[FinalLoraDiagnosticCandidate],
    cfg: FluxCannyLoraConfig,
) -> list[dict[str, Any]]:
    """Return the deterministic three-rank final-LoRA diagnostic matrix."""

    strength = float(cfg.validation_partial_noise_strength)
    noise_slug = noise_strength_slug(strength)
    guidance_scale = float(cfg.validation_guidance_scale)
    guidance_slug = guidance_scale_slug(guidance_scale)
    window_id = str(cfg.validation_windows[0]["window_id"])
    records: list[dict[str, Any]] = []
    for candidate in candidates:
        stage = Path(f"rank{candidate.rank}") / "final" / guidance_slug / noise_slug / window_id
        records.append(
            {
                "rank": int(candidate.rank),
                "checkpoint": "final",
                "lora_label": candidate.label,
                "lora_dir": str(candidate.lora_dir),
                "weights_path": str(candidate.weights_path),
                "partial_noise_strength": strength,
                "noise_slug": noise_slug,
                "guidance_scale": guidance_scale,
                "guidance_slug": guidance_slug,
                "window_id": window_id,
                "relative_output_dir": str(stage),
            }
        )
    return records


def plan_flux_canny_final_guidance_diagnostic(
    *,
    config_path: str | Path,
    training_root: str | Path | None = None,
) -> dict[str, Any]:
    """Build the diagnostic plan without loading FLUX."""

    config = Path(config_path)
    cfg = load_final_guidance_diagnostic_config(config)
    ranks = diagnostic_ranks_from_config(config)
    rosary_files = verify_required_rosary_files(required_rosary_files_from_config(config))
    root = (
        Path(training_root).expanduser()
        if training_root is not None
        else runs_root() / "training"
    )
    candidates = [discover_final_lora_candidate(root, rank) for rank in ranks]
    planned_records = planned_final_guidance_records(candidates=candidates, cfg=cfg)
    return {
        "completed": False,
        "config_path": str(config),
        "model_id": cfg.model_id,
        "prompt": cfg.validation_prompt,
        "backend": "lanpaint",
        "diagnostic": "final_loras_guidance15_left_window",
        "ranks": [int(rank) for rank in ranks],
        "canny_low_threshold": cfg.canny_low_threshold,
        "canny_high_threshold": cfg.canny_high_threshold,
        "num_inference_steps": cfg.validation_num_inference_steps,
        "partial_noise_strength": float(cfg.validation_partial_noise_strength),
        "noise_slug": noise_strength_slug(cfg.validation_partial_noise_strength),
        "guidance_scale": float(cfg.validation_guidance_scale),
        "guidance_slug": guidance_scale_slug(cfg.validation_guidance_scale),
        "windows": [dict(window) for window in cfg.validation_windows],
        "rosary_files": rosary_files,
        "lora_candidates": [candidate.to_manifest() for candidate in candidates],
        "planned_record_count": len(planned_records),
        "planned_records": planned_records,
        "completed_records": [],
        "adapter_events": [],
        "pipeline_reload_count": 0,
        "package_versions": package_versions(
            ["torch", "diffusers", "transformers", "accelerate", "peft", "LanPaint"]
        ),
    }


def run_flux_canny_final_guidance_diagnostic(
    *,
    config_path: str | Path,
    training_root: str | Path | None = None,
    run_root: str | Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run or dry-run the final LoRA one-window guidance diagnostic."""

    plan = plan_flux_canny_final_guidance_diagnostic(
        config_path=config_path,
        training_root=training_root,
    )
    if dry_run:
        return {**plan, "dry_run": True}

    cfg = load_final_guidance_diagnostic_config(config_path)
    candidates = [
        FinalLoraDiagnosticCandidate(
            rank=int(item["rank"]),
            label=str(item["label"]),
            run_dir=Path(item["run_dir"]),
            lora_dir=Path(item["lora_dir"]),
            weights_path=Path(item["weights_path"]),
        )
        for item in plan["lora_candidates"]
    ]
    run_dir = make_final_guidance_diagnostic_run_dir(run_root)
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

    noise_slug = noise_strength_slug(cfg.validation_partial_noise_strength)
    guidance_slug = guidance_scale_slug(cfg.validation_guidance_scale)

    try:
        for index, candidate in enumerate(candidates):
            adapter_name = f"rank{candidate.rank}_final"
            print(f"Loading final LoRA for rank {candidate.rank}: {candidate.lora_dir}")
            load_info = load_flux_canny_lora_adapter(
                pipe,
                candidate.lora_dir,
                adapter_name=adapter_name,
            )
            manifest["adapter_events"].append(
                {"event": "load", "rank": candidate.rank, **load_info}
            )
            write_json(manifest_path, manifest)

            cfg_for_rank = replace(cfg, rank=int(candidate.rank))
            stage = f"rank{candidate.rank}/final/{guidance_slug}/{noise_slug}"
            stage_dir, window_results = run_flux_canny_lanpaint_prepared_validation(
                pipe,
                cfg_for_rank,
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
                        "rank": int(candidate.rank),
                        "checkpoint": "final",
                        "lora_label": candidate.label,
                        "lora_dir": str(candidate.lora_dir),
                        "weights_path": str(candidate.weights_path),
                        "partial_noise_strength": float(cfg.validation_partial_noise_strength),
                        "noise_slug": noise_slug,
                        "guidance_scale": float(cfg.validation_guidance_scale),
                        "guidance_slug": guidance_slug,
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
            manifest["adapter_events"].append(
                {"event": "cleanup", "rank": candidate.rank, **cleanup}
            )
            write_json(manifest_path, manifest)

            if not cleanup["removed"] and index < len(candidates) - 1:
                print("LoRA adapter cleanup unavailable; reloading FLUX pipeline before next rank.")
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
    "DEFAULT_DIAGNOSTIC_RANKS",
    "FinalLoraDiagnosticCandidate",
    "diagnostic_ranks_from_config",
    "discover_final_lora_candidate",
    "guidance_scale_slug",
    "load_final_guidance_diagnostic_config",
    "plan_flux_canny_final_guidance_diagnostic",
    "planned_final_guidance_records",
    "run_flux_canny_final_guidance_diagnostic",
]
