"""One-window final-LoRA diagnostic using Diffusers FluxControlInpaintPipeline."""

from __future__ import annotations

import gc
import inspect
import os
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from PIL import Image

from painting_inpaint.compositing import hard_composite, outside_mask_changed
from painting_inpaint.experiments.manifest import package_versions, timestamp_utc, write_json
from painting_inpaint.masks import make_mask_overlay
from painting_inpaint.paths import runs_root
from painting_inpaint.training.flux_canny_guidance_diagnostic import (
    FinalLoraDiagnosticCandidate,
    diagnostic_ranks_from_config,
    discover_final_lora_candidate,
    guidance_scale_slug,
    load_final_guidance_diagnostic_config,
    planned_final_guidance_records,
)
from painting_inpaint.training.flux_canny_lanpaint import (
    prepare_flux_canny_lanpaint_validation_inputs,
)
from painting_inpaint.training.flux_canny_validation_sweep import (
    noise_strength_slug,
    required_rosary_files_from_config,
    verify_required_rosary_files,
)
from painting_inpaint.visualization import make_comparison_sheet


def clear_cuda_cache() -> None:
    """Clear Python and CUDA caches if CUDA is available."""

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def callable_accepts(callable_obj: Any, parameter_name: str) -> bool:
    """Return whether a callable accepts a parameter or arbitrary kwargs."""

    try:
        signature = inspect.signature(callable_obj)
    except Exception:
        return False
    if parameter_name in signature.parameters:
        return True
    return any(
        param.kind is inspect.Parameter.VAR_KEYWORD
        for param in signature.parameters.values()
    )


def native_strength_schedule_estimate(num_inference_steps: int, strength: float) -> dict[str, Any]:
    """Approximate Diffusers inpaint strength schedule length."""

    steps = int(num_inference_steps)
    value = float(strength)
    init_timestep = min(round(steps * value), steps)
    t_start = max(steps - init_timestep, 0)
    return {
        "source": "Diffusers native strength parameter",
        "requested_num_inference_steps": steps,
        "strength": value,
        "init_timestep": init_timestep,
        "t_start": t_start,
        "estimated_effective_num_steps": steps - t_start,
        "interpretation": (
            "higher strength adds more noise and runs more denoising steps; "
            "strength=1.0 uses the full schedule"
        ),
    }


def make_final_inpaint_diagnostic_run_dir(run_root: str | Path | None) -> Path:
    """Create the Diffusers inpaint diagnostic run directory."""

    root = Path(run_root).expanduser() if run_root is not None else runs_root() / "validation"
    run_dir = root / f"flux1_canny_rosary_inpaint_final_guidance15_left_{timestamp_utc()}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def plan_flux_canny_final_inpaint_diagnostic(
    *,
    config_path: str | Path,
    training_root: str | Path | None = None,
) -> dict[str, Any]:
    """Build the Diffusers inpaint diagnostic plan without loading FLUX."""

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
    for record in planned_records:
        record["relative_output_dir"] = str(
            Path(f"rank{record['rank']}")
            / "final"
            / record["guidance_slug"]
            / record["noise_slug"]
            / record["window_id"]
        )
    return {
        "completed": False,
        "config_path": str(config),
        "model_id": cfg.model_id,
        "prompt": cfg.validation_prompt,
        "backend": "diffusers_inpaint",
        "diagnostic": "final_loras_guidance15_left_window_inpaint_pipeline",
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
            ["torch", "diffusers", "transformers", "accelerate", "peft"]
        ),
    }


def load_flux_canny_inpaint_pipeline(cfg: Any) -> tuple[Any, Any]:
    """Load the native Diffusers FluxControlInpaintPipeline."""

    import torch
    from diffusers import FluxControlInpaintPipeline
    from huggingface_hub import get_token, login

    if not torch.cuda.is_available():
        raise RuntimeError("FluxControlInpaintPipeline diagnostic requires a CUDA GPU.")

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN") or get_token()
    if not hf_token:
        raise RuntimeError("Missing HF_TOKEN for gated FLUX.1-Canny-dev validation.")
    login(token=hf_token, add_to_git_credential=False)

    torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    clear_cuda_cache()
    pipe = FluxControlInpaintPipeline.from_pretrained(
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


def load_inpaint_lora_adapter(pipe: Any, candidate: FinalLoraDiagnosticCandidate) -> dict[str, Any]:
    """Load one final LoRA into the Diffusers inpaint pipeline."""

    adapter_name = f"rank{candidate.rank}_final"
    if not hasattr(pipe, "load_lora_weights"):
        raise RuntimeError(f"{type(pipe).__name__} does not expose load_lora_weights.")
    pipe.load_lora_weights(
        str(candidate.lora_dir),
        weight_name=candidate.weights_path.name,
        adapter_name=adapter_name,
    )
    strength_mode = "loaded_default_weight"
    if hasattr(pipe, "set_adapters"):
        try:
            pipe.set_adapters([adapter_name], adapter_weights=[1.0])
            strength_mode = "set_adapters"
        except TypeError:
            pipe.set_adapters([adapter_name])
            strength_mode = "set_adapters_no_weights"
    return {
        "adapter_name": adapter_name,
        "lora_dir": str(candidate.lora_dir),
        "weights_path": str(candidate.weights_path),
        "load_method": "load_lora_weights",
        "strength_mode": strength_mode,
    }


def remove_inpaint_lora_adapter(pipe: Any, adapter_name: str) -> dict[str, Any]:
    """Remove a LoRA adapter when supported by the installed Diffusers runtime."""

    if hasattr(pipe, "delete_adapters"):
        try:
            pipe.delete_adapters(adapter_name)
            clear_cuda_cache()
            return {"removed": True, "method": "delete_adapters", "adapter_name": adapter_name}
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
            return {"removed": True, "method": "unload_lora_weights", "adapter_name": adapter_name}
        except Exception as exc:
            return {
                "removed": False,
                "method": "unload_lora_weights",
                "adapter_name": adapter_name,
                "error": repr(exc),
            }
    return {"removed": False, "method": "none_available", "adapter_name": adapter_name}


def run_one_inpaint_candidate(
    *,
    pipe: Any,
    cfg: Any,
    candidate: FinalLoraDiagnosticCandidate,
    prepared_window: dict[str, Any],
    run_dir: Path,
    torch_dtype: Any,
) -> dict[str, Any]:
    """Run one rank final LoRA through FluxControlInpaintPipeline."""

    import torch

    noise_slug = noise_strength_slug(cfg.validation_partial_noise_strength)
    guidance_slug = guidance_scale_slug(cfg.validation_guidance_scale)
    window_id = str(prepared_window["window_record"]["window_id"])
    stage_dir = (
        run_dir
        / f"rank{candidate.rank}"
        / "final"
        / guidance_slug
        / noise_slug
        / window_id
    )
    stage_dir.mkdir(parents=True, exist_ok=True)

    input_crop = prepared_window["input_crop"]
    mask_crop = prepared_window["mask_crop"]
    canny_control = prepared_window["canny_control"]
    raw_output_path = stage_dir / "raw_output.png"
    composite_path = stage_dir / "composite.png"
    comparison_path = stage_dir / "comparison.png"
    metadata_path = stage_dir / "window_metadata.json"

    step_trace: list[dict[str, Any]] = []

    def record_step_trace(_pipeline: Any, step_index: int, timestep: Any, callback_kwargs: Any):
        value = timestep
        if hasattr(value, "detach"):
            value = value.detach().float().cpu()
            if value.numel() == 1:
                value = float(value.item())
            else:
                value = [float(item) for item in value.flatten().tolist()]
        step_trace.append({"step_index": int(step_index), "timestep": value})
        return callback_kwargs

    generator = torch.Generator(device="cpu").manual_seed(int(cfg.validation_seed))
    call_kwargs = {
        "prompt": cfg.validation_prompt,
        "image": input_crop,
        "control_image": canny_control,
        "mask_image": mask_crop,
        "height": int(prepared_window["window_size"]),
        "width": int(prepared_window["window_size"]),
        "strength": float(cfg.validation_partial_noise_strength),
        "num_inference_steps": int(cfg.validation_num_inference_steps),
        "guidance_scale": float(cfg.validation_guidance_scale),
        "generator": generator,
    }
    if callable_accepts(pipe.__call__, "max_sequence_length"):
        call_kwargs["max_sequence_length"] = int(cfg.validation_max_sequence_length)
    if callable_accepts(pipe.__call__, "callback_on_step_end"):
        call_kwargs["callback_on_step_end"] = record_step_trace
    if callable_accepts(pipe.__call__, "callback_on_step_end_tensor_inputs"):
        call_kwargs["callback_on_step_end_tensor_inputs"] = ["latents"]

    print(
        f"Running rank{candidate.rank}/final/{guidance_slug}/{noise_slug}/{window_id}: "
        f"box=({prepared_window['window_record']['x']}, {prepared_window['window_record']['y']}, "
        f"{prepared_window['window_record']['x'] + prepared_window['window_record']['width']}, "
        f"{prepared_window['window_record']['y'] + prepared_window['window_record']['height']}), "
        f"strength={cfg.validation_partial_noise_strength}, "
        f"guidance={cfg.validation_guidance_scale}, seed={cfg.validation_seed}"
    )
    started = time.perf_counter()
    with torch.inference_mode():
        raw_output = pipe(**call_kwargs).images[0].convert("RGB")
    runtime_seconds = time.perf_counter() - started

    if raw_output.size != input_crop.size:
        raw_output = raw_output.resize(input_crop.size, Image.Resampling.LANCZOS)
    raw_output.save(raw_output_path)

    composite = hard_composite(input_crop, raw_output, mask_crop)
    changed_outside_mask = outside_mask_changed(input_crop, composite, mask_crop)
    if changed_outside_mask:
        raise AssertionError(f"{window_id}: hard composite changed pixels outside the mask")
    composite.save(composite_path)

    make_comparison_sheet(
        [
            input_crop,
            make_mask_overlay(input_crop, mask_crop, color=(255, 40, 40), alpha=110 / 255),
            canny_control,
            raw_output,
            composite,
        ],
        ["input crop", "mask overlay", "canny control", "Diffusers raw", "hard composite"],
        output_path=comparison_path,
        max_panel_side=512,
        label_height=28,
    )

    metadata = {
        **prepared_window["window_record"],
        "backend": "diffusers_inpaint",
        "pipeline_class": type(pipe).__name__,
        "pipeline_call_used": True,
        "model_id": cfg.model_id,
        "rank": int(candidate.rank),
        "checkpoint": "final",
        "lora_dir": str(candidate.lora_dir),
        "weights_path": str(candidate.weights_path),
        "torch_dtype": str(torch_dtype),
        "prompt": cfg.validation_prompt,
        "seed": int(cfg.validation_seed),
        "elapsed_seconds": runtime_seconds,
        "control_source": prepared_window["resolved_control_source"],
        "inference_settings": {
            "strength": float(cfg.validation_partial_noise_strength),
            "partial_noise_control": "diffusers_native_strength_argument",
            "num_inference_steps": int(cfg.validation_num_inference_steps),
            "guidance_scale": float(cfg.validation_guidance_scale),
            "height": int(prepared_window["window_size"]),
            "width": int(prepared_window["window_size"]),
            "max_sequence_length": int(cfg.validation_max_sequence_length),
        },
        "native_strength_schedule_estimate": native_strength_schedule_estimate(
            cfg.validation_num_inference_steps,
            cfg.validation_partial_noise_strength,
        ),
        "actual_schedule_trace": {
            "step_trace": step_trace,
            "actual_recorded_num_steps": len(step_trace),
            "first_recorded_timestep": step_trace[0]["timestep"] if step_trace else None,
            "last_recorded_timestep": step_trace[-1]["timestep"] if step_trace else None,
        },
        "canny_settings": {
            "source": prepared_window["resolved_control_source"],
            "low_threshold": cfg.canny_low_threshold,
            "high_threshold": cfg.canny_high_threshold,
            "blur_radius": cfg.validation_canny_blur_radius,
        },
        "mask_conventions": {
            "project_mask_edit": "white = edit, black = preserve",
            "diffusers_mask": "white = repaint/edit, black = preserve",
        },
        "paths": {
            **{key: str(path) for key, path in prepared_window["paths"].items()},
            "raw_output": str(raw_output_path),
            "composite": str(composite_path),
            "comparison": str(comparison_path),
            "source_base_image": str(prepared_window["base_path"]),
            "source_mask": str(prepared_window["mask_path"]),
            "source_conditioning_prior": str(prepared_window["prior_path"])
            if prepared_window["prior_path"] is not None
            else None,
            "source_control_source": str(prepared_window["control_source_path"]),
        },
        "outside_mask_changed_after_hard_composite": changed_outside_mask,
    }
    write_json(metadata_path, metadata)

    print(f"Saved rank{candidate.rank} Diffusers inpaint diagnostic images to {stage_dir}")
    return {
        "rank": int(candidate.rank),
        "checkpoint": "final",
        "lora_label": candidate.label,
        "lora_dir": str(candidate.lora_dir),
        "weights_path": str(candidate.weights_path),
        "partial_noise_strength": float(cfg.validation_partial_noise_strength),
        "noise_slug": noise_slug,
        "guidance_scale": float(cfg.validation_guidance_scale),
        "guidance_slug": guidance_slug,
        "window_id": window_id,
        "stage_dir": str(stage_dir),
        "window_metadata": str(metadata_path),
        "comparison": str(comparison_path),
        "composite": str(composite_path),
        "raw_output": str(raw_output_path),
        "seed": int(cfg.validation_seed),
        "elapsed_seconds": runtime_seconds,
        "completed": True,
    }


def run_flux_canny_final_inpaint_diagnostic(
    *,
    config_path: str | Path,
    training_root: str | Path | None = None,
    run_root: str | Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run or dry-run the final LoRA Diffusers inpaint diagnostic."""

    plan = plan_flux_canny_final_inpaint_diagnostic(
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
    run_dir = make_final_inpaint_diagnostic_run_dir(run_root)
    manifest = {
        **plan,
        "dry_run": False,
        "validation_run_dir": str(run_dir),
        "inputs_dir": str(run_dir / "inputs"),
    }
    manifest_path = run_dir / "validation_manifest.json"
    write_json(manifest_path, manifest)

    prepared_windows = prepare_flux_canny_lanpaint_validation_inputs(cfg, run_dir / "inputs")
    prepared_window = prepared_windows[0]
    pipe, torch_dtype = load_flux_canny_inpaint_pipeline(cfg)
    api_report = {
        "backend": "diffusers_inpaint",
        "model_id": cfg.model_id,
        "pipeline_class": type(pipe).__name__,
        "call_signature": str(inspect.signature(pipe.__call__)),
        "accepts_image": callable_accepts(pipe.__call__, "image"),
        "accepts_control_image": callable_accepts(pipe.__call__, "control_image"),
        "accepts_mask_image": callable_accepts(pipe.__call__, "mask_image"),
        "accepts_strength": callable_accepts(pipe.__call__, "strength"),
        "accepts_callback_on_step_end": callable_accepts(pipe.__call__, "callback_on_step_end"),
        "accepts_max_sequence_length": callable_accepts(pipe.__call__, "max_sequence_length"),
        "torch_dtype": str(torch_dtype),
        "pipeline_call_used": True,
    }
    write_json(run_dir / "api_report.json", api_report)
    missing = [
        name
        for name in ["image", "control_image", "mask_image", "strength"]
        if not callable_accepts(pipe.__call__, name)
    ]
    if missing:
        raise RuntimeError(f"Pipeline call signature is missing expected parameters: {missing}")

    try:
        for index, candidate in enumerate(candidates):
            print(f"Loading final LoRA for rank {candidate.rank}: {candidate.lora_dir}")
            load_info = load_inpaint_lora_adapter(pipe, candidate)
            manifest["adapter_events"].append(
                {"event": "load", "rank": candidate.rank, **load_info}
            )
            write_json(manifest_path, manifest)

            cfg_for_rank = replace(cfg, rank=int(candidate.rank))
            record = run_one_inpaint_candidate(
                pipe=pipe,
                cfg=cfg_for_rank,
                candidate=candidate,
                prepared_window=prepared_window,
                run_dir=run_dir,
                torch_dtype=torch_dtype,
            )
            manifest["completed_records"].append(record)
            write_json(manifest_path, manifest)

            cleanup = remove_inpaint_lora_adapter(pipe, str(load_info["adapter_name"]))
            manifest["adapter_events"].append(
                {"event": "cleanup", "rank": candidate.rank, **cleanup}
            )
            write_json(manifest_path, manifest)

            if not cleanup["removed"] and index < len(candidates) - 1:
                print("LoRA adapter cleanup unavailable; reloading pipeline before next rank.")
                del pipe
                clear_cuda_cache()
                pipe, torch_dtype = load_flux_canny_inpaint_pipeline(cfg)
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
    "callable_accepts",
    "clear_cuda_cache",
    "load_flux_canny_inpaint_pipeline",
    "native_strength_schedule_estimate",
    "plan_flux_canny_final_inpaint_diagnostic",
    "run_flux_canny_final_inpaint_diagnostic",
]
