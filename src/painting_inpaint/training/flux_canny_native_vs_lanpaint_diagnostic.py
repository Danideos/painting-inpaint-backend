"""Tiny native-vs-LanPaint diagnostic for FLUX.1-Canny LoRA validation."""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

from painting_inpaint.compositing import hard_composite, outside_mask_changed
from painting_inpaint.experiments.manifest import package_versions, timestamp_utc, write_json
from painting_inpaint.masks import make_mask_overlay
from painting_inpaint.paths import runs_root
from painting_inpaint.training.flux_canny_guidance_diagnostic import guidance_scale_slug
from painting_inpaint.training.flux_canny_inpaint_diagnostic import (
    callable_accepts,
    clear_cuda_cache,
    load_flux_canny_inpaint_pipeline,
    native_strength_schedule_estimate,
    remove_inpaint_lora_adapter,
)
from painting_inpaint.training.flux_canny_lanpaint import (
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
    discover_latest_complete_rank_run,
    noise_strength_slug,
    required_rosary_files_from_config,
    verify_required_rosary_files,
)
from painting_inpaint.visualization import make_comparison_sheet

BIG_BRANCH_MODES = ("identical", "approx_cfg_big_1")


@dataclass(frozen=True)
class TinyDiagnosticLoraCandidate:
    """The single conservative checkpoint LoRA used by the tiny diagnostic."""

    rank: int
    checkpoint_step: int
    adapter_weight: float
    run_dir: Path
    lora_dir: Path
    weights_path: Path

    @property
    def checkpoint(self) -> str:
        return f"checkpoint-{self.checkpoint_step}"

    @property
    def label(self) -> str:
        return (
            f"rank{self.rank}_checkpoint{self.checkpoint_step}_"
            f"{lora_scale_slug(self.adapter_weight)}"
        )

    def to_manifest(self) -> dict[str, Any]:
        return {
            "rank": int(self.rank),
            "checkpoint": self.checkpoint,
            "checkpoint_step": int(self.checkpoint_step),
            "adapter_weight": float(self.adapter_weight),
            "scale_slug": lora_scale_slug(self.adapter_weight),
            "label": self.label,
            "run_dir": str(self.run_dir),
            "lora_dir": str(self.lora_dir),
            "weights_path": str(self.weights_path),
        }


def lora_scale_slug(scale: float) -> str:
    """Return a stable folder slug for a LoRA adapter scale."""

    value = float(scale)
    if value < 0:
        raise ValueError("LoRA adapter scale must be non-negative.")
    return f"scale{round(value * 100):03d}"


def validate_big_branch_mode(mode: str) -> str:
    """Validate and normalize a tiny-diagnostic LanPaint BiG branch mode."""

    value = str(mode)
    if value not in BIG_BRANCH_MODES:
        raise ValueError(
            f"Unsupported BiG branch mode {value!r}; expected one of {list(BIG_BRANCH_MODES)}"
        )
    return value


def flux_time_record(flow_t: float) -> dict[str, float]:
    """Return FLUX/LanPaint time values for one flow sigma."""

    flow = float(flow_t)
    abt = (1.0 - flow) ** 2 / ((1.0 - flow) ** 2 + flow**2)
    ve_sigma = flow / max(1.0 - flow, 1e-6)
    return {"flow_t": flow, "abt": float(abt), "ve_sigma": float(ve_sigma)}


def flux_schedule_records(flow_values: list[float]) -> list[dict[str, float]]:
    """Return FLUX/LanPaint time values for a flow sigma list."""

    return [flux_time_record(value) for value in flow_values]


def _load_raw_yaml(path: str | Path) -> dict[str, Any]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Tiny diagnostic config must be a mapping: {path}")
    return raw


def _diagnostic_settings(path: str | Path) -> dict[str, Any]:
    raw = _load_raw_yaml(path)
    diagnostic = raw.get("diagnostic", {})
    if not isinstance(diagnostic, dict):
        raise ValueError("diagnostic must be a mapping.")
    modes = diagnostic.get("big_branch_modes", list(BIG_BRANCH_MODES))
    if not isinstance(modes, list | tuple) or not modes:
        raise ValueError("diagnostic.big_branch_modes must be a non-empty list.")
    return {
        "rank": int(diagnostic.get("lora_rank", 16)),
        "checkpoint_step": int(diagnostic.get("checkpoint_step", 1000)),
        "adapter_weight": float(diagnostic.get("lora_adapter_weight", 0.5)),
        "big_branch_modes": [validate_big_branch_mode(mode) for mode in modes],
    }


def load_tiny_native_vs_lanpaint_config(path: str | Path) -> FluxCannyLoraConfig:
    """Load and validate the tiny diagnostic config."""

    cfg = load_flux_canny_lora_config(path)
    if len(cfg.validation_windows) != 1:
        raise ValueError("Tiny native-vs-LanPaint diagnostic expects exactly one window.")
    if cfg.validation_num_inference_steps != 30:
        raise ValueError("Tiny native-vs-LanPaint diagnostic expects 30 inference steps.")
    if float(cfg.validation_guidance_scale) != 3.5:
        raise ValueError("Tiny native-vs-LanPaint diagnostic expects guidance_scale=3.5.")
    if float(cfg.validation_partial_noise_strength) != 1.0:
        raise ValueError("Tiny native-vs-LanPaint diagnostic expects partial_noise_strength=1.0.")
    noise_strength_slug(cfg.validation_partial_noise_strength)
    guidance_scale_slug(cfg.validation_guidance_scale)
    return cfg


def discover_tiny_diagnostic_lora(
    training_root: str | Path,
    *,
    rank: int,
    checkpoint_step: int,
    adapter_weight: float,
) -> TinyDiagnosticLoraCandidate:
    """Find the configured conservative checkpoint LoRA."""

    run_dir = discover_latest_complete_rank_run(training_root, rank)
    lora_dir = run_dir / f"checkpoint-{int(checkpoint_step)}"
    weights_path = lora_dir / LORA_WEIGHTS_NAME
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Missing rank-{rank} checkpoint-{checkpoint_step} LoRA weights: {weights_path}"
        )
    return TinyDiagnosticLoraCandidate(
        rank=int(rank),
        checkpoint_step=int(checkpoint_step),
        adapter_weight=float(adapter_weight),
        run_dir=run_dir,
        lora_dir=lora_dir,
        weights_path=weights_path,
    )


def planned_tiny_diagnostic_records(
    *,
    cfg: FluxCannyLoraConfig,
    lora: TinyDiagnosticLoraCandidate,
    big_branch_modes: list[str],
) -> list[dict[str, Any]]:
    """Return the fixed native/LanPaint tiny diagnostic matrix."""

    window_id = str(cfg.validation_windows[0]["window_id"])
    guidance_slug = guidance_scale_slug(cfg.validation_guidance_scale)
    noise_slug = noise_strength_slug(cfg.validation_partial_noise_strength)
    scale_slug = lora_scale_slug(lora.adapter_weight)
    records: list[dict[str, Any]] = [
        {
            "case_id": "A_native_no_lora",
            "backend": "native_diffusers_inpaint",
            "lora_enabled": False,
            "lora_scale": None,
            "big_branch_mode": None,
        }
    ]
    records.extend(
        {
            "case_id": f"B_lanpaint_no_lora_big_{mode}",
            "backend": "lanpaint_adapter",
            "lora_enabled": False,
            "lora_scale": None,
            "big_branch_mode": mode,
        }
        for mode in big_branch_modes
    )
    records.append(
        {
            "case_id": f"C_native_rank{lora.rank}_checkpoint{lora.checkpoint_step}_{scale_slug}",
            "backend": "native_diffusers_inpaint",
            "lora_enabled": True,
            "lora_scale": float(lora.adapter_weight),
            "lora_dir": str(lora.lora_dir),
            "weights_path": str(lora.weights_path),
            "checkpoint": lora.checkpoint,
            "big_branch_mode": None,
        }
    )
    records.extend(
        {
            "case_id": (
                f"D_lanpaint_rank{lora.rank}_checkpoint{lora.checkpoint_step}_"
                f"{scale_slug}_big_{mode}"
            ),
            "backend": "lanpaint_adapter",
            "lora_enabled": True,
            "lora_scale": float(lora.adapter_weight),
            "lora_dir": str(lora.lora_dir),
            "weights_path": str(lora.weights_path),
            "checkpoint": lora.checkpoint,
            "big_branch_mode": mode,
        }
        for mode in big_branch_modes
    )
    for record in records:
        record["guidance_scale"] = float(cfg.validation_guidance_scale)
        record["guidance_slug"] = guidance_slug
        record["partial_noise_strength"] = float(cfg.validation_partial_noise_strength)
        record["noise_slug"] = noise_slug
        record["window_id"] = window_id
        record["relative_output_dir"] = str(Path(record["case_id"]) / window_id)
    return records


def plan_flux_canny_native_vs_lanpaint_tiny_diagnostic(
    *,
    config_path: str | Path,
    training_root: str | Path | None = None,
) -> dict[str, Any]:
    """Build the tiny diagnostic plan without loading FLUX."""

    config = Path(config_path)
    cfg = load_tiny_native_vs_lanpaint_config(config)
    settings = _diagnostic_settings(config)
    rosary_files = verify_required_rosary_files(required_rosary_files_from_config(config))
    root = (
        Path(training_root).expanduser()
        if training_root is not None
        else runs_root() / "training"
    )
    lora = discover_tiny_diagnostic_lora(
        root,
        rank=settings["rank"],
        checkpoint_step=settings["checkpoint_step"],
        adapter_weight=settings["adapter_weight"],
    )
    planned_records = planned_tiny_diagnostic_records(
        cfg=cfg,
        lora=lora,
        big_branch_modes=settings["big_branch_modes"],
    )
    return {
        "completed": False,
        "config_path": str(config),
        "model_id": cfg.model_id,
        "prompt": cfg.validation_prompt,
        "diagnostic": "native_vs_lanpaint_tiny",
        "canny_low_threshold": cfg.canny_low_threshold,
        "canny_high_threshold": cfg.canny_high_threshold,
        "num_inference_steps": int(cfg.validation_num_inference_steps),
        "partial_noise_strength": float(cfg.validation_partial_noise_strength),
        "noise_slug": noise_strength_slug(cfg.validation_partial_noise_strength),
        "guidance_scale": float(cfg.validation_guidance_scale),
        "guidance_slug": guidance_scale_slug(cfg.validation_guidance_scale),
        "windows": [dict(window) for window in cfg.validation_windows],
        "rosary_files": rosary_files,
        "lora_candidate": lora.to_manifest(),
        "big_branch_modes": list(settings["big_branch_modes"]),
        "planned_record_count": len(planned_records),
        "planned_records": planned_records,
        "completed_records": [],
        "adapter_events": [],
        "package_versions": package_versions(
            ["torch", "diffusers", "transformers", "accelerate", "peft", "LanPaint"]
        ),
    }


def make_tiny_diagnostic_run_dir(run_root: str | Path | None) -> Path:
    """Create the tiny diagnostic run directory."""

    root = Path(run_root).expanduser() if run_root is not None else runs_root() / "validation"
    run_dir = root / f"flux1_canny_native_vs_lanpaint_tiny_{timestamp_utc()}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _tensor_to_float_list(value: Any) -> list[float]:
    if value is None:
        return []
    try:
        import torch

        if torch.is_tensor(value):
            return [float(item) for item in value.detach().float().cpu().flatten().tolist()]
    except Exception:
        pass
    if isinstance(value, list | tuple):
        return [float(item) for item in value]
    return []


def _scheduler_report(pipe: Any, step_trace: list[dict[str, Any]]) -> dict[str, Any]:
    scheduler = getattr(pipe, "scheduler", None)
    timesteps = _tensor_to_float_list(getattr(scheduler, "timesteps", None))
    sigmas = _tensor_to_float_list(getattr(scheduler, "sigmas", None))
    flow_values = sigmas[: len(step_trace)] if sigmas else []
    return {
        "scheduler_class": type(scheduler).__name__ if scheduler is not None else None,
        "scheduler_config": dict(getattr(scheduler, "config", {}) or {})
        if scheduler is not None
        else {},
        "callback_step_trace": step_trace,
        "actual_recorded_num_steps": len(step_trace),
        "scheduler_timesteps": timesteps,
        "scheduler_sigmas": sigmas,
        "flow_time_records": flux_schedule_records(flow_values),
    }


def _load_native_lora_adapter(pipe: Any, lora: TinyDiagnosticLoraCandidate) -> dict[str, Any]:
    adapter_name = lora.label
    if not hasattr(pipe, "load_lora_weights"):
        raise RuntimeError(f"{type(pipe).__name__} does not expose load_lora_weights.")
    pipe.load_lora_weights(
        str(lora.lora_dir),
        weight_name=lora.weights_path.name,
        adapter_name=adapter_name,
    )
    strength_mode = "loaded_default_weight"
    if hasattr(pipe, "set_adapters"):
        try:
            pipe.set_adapters([adapter_name], adapter_weights=[float(lora.adapter_weight)])
            strength_mode = "set_adapters"
        except TypeError:
            pipe.set_adapters([adapter_name])
            strength_mode = "set_adapters_no_weights"
    return {
        "adapter_name": adapter_name,
        "adapter_weight": float(lora.adapter_weight),
        "lora_dir": str(lora.lora_dir),
        "weights_path": str(lora.weights_path),
        "load_method": "load_lora_weights",
        "strength_mode": strength_mode,
    }


def run_native_tiny_case(
    *,
    pipe: Any,
    cfg: FluxCannyLoraConfig,
    prepared_window: dict[str, Any],
    run_dir: Path,
    case_id: str,
    torch_dtype: Any,
    lora: TinyDiagnosticLoraCandidate | None,
) -> dict[str, Any]:
    """Run one native Diffusers inpaint tiny diagnostic case."""

    import torch

    window_record = dict(prepared_window["window_record"])
    window_id = str(window_record["window_id"])
    case_dir = run_dir / case_id / window_id
    case_dir.mkdir(parents=True, exist_ok=True)

    input_crop = prepared_window["input_crop"]
    mask_crop = prepared_window["mask_crop"]
    canny_control = prepared_window["canny_control"]
    raw_output_path = case_dir / "raw_output.png"
    composite_path = case_dir / "composite.png"
    comparison_path = case_dir / "comparison.png"
    metadata_path = case_dir / "window_metadata.json"

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
        f"Running {case_id}/{window_id}: backend=native_diffusers_inpaint, "
        f"lora_enabled={lora is not None}, guidance={cfg.validation_guidance_scale}, "
        f"strength={cfg.validation_partial_noise_strength}, seed={cfg.validation_seed}"
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
        raise AssertionError(f"{case_id}/{window_id}: hard composite changed outside mask")
    composite.save(composite_path)

    make_comparison_sheet(
        [
            input_crop,
            make_mask_overlay(input_crop, mask_crop, color=(255, 40, 40), alpha=110 / 255),
            canny_control,
            raw_output,
            composite,
        ],
        ["input crop", "mask overlay", "canny control", "native raw", "hard composite"],
        output_path=comparison_path,
        max_panel_side=512,
        label_height=28,
    )

    schedule_report = _scheduler_report(pipe, step_trace)
    metadata = {
        **window_record,
        "case_id": case_id,
        "backend": "native_diffusers_inpaint",
        "pipeline_class": type(pipe).__name__,
        "pipeline_call_used": True,
        "model_id": cfg.model_id,
        "torch_dtype": str(torch_dtype),
        "prompt": cfg.validation_prompt,
        "seed": int(cfg.validation_seed),
        "elapsed_seconds": runtime_seconds,
        "control_source": prepared_window["resolved_control_source"],
        "lora_enabled": lora is not None,
        "lora": lora.to_manifest() if lora is not None else None,
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
        "actual_schedule_trace": schedule_report,
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
    print(f"Saved native diagnostic case to {case_dir}")
    return {
        "case_id": case_id,
        "backend": "native_diffusers_inpaint",
        "window_id": window_id,
        "lora_enabled": lora is not None,
        "lora": lora.to_manifest() if lora is not None else None,
        "window_metadata": str(metadata_path),
        "comparison": str(comparison_path),
        "composite": str(composite_path),
        "raw_output": str(raw_output_path),
        "seed": int(cfg.validation_seed),
        "elapsed_seconds": runtime_seconds,
        "completed": True,
    }


def run_flux_canny_native_vs_lanpaint_tiny_diagnostic(
    *,
    config_path: str | Path,
    training_root: str | Path | None = None,
    run_root: str | Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run or dry-run the tiny native-vs-LanPaint diagnostic."""

    plan = plan_flux_canny_native_vs_lanpaint_tiny_diagnostic(
        config_path=config_path,
        training_root=training_root,
    )
    if dry_run:
        return {**plan, "dry_run": True}

    cfg = load_tiny_native_vs_lanpaint_config(config_path)
    lora = TinyDiagnosticLoraCandidate(
        rank=int(plan["lora_candidate"]["rank"]),
        checkpoint_step=int(plan["lora_candidate"]["checkpoint_step"]),
        adapter_weight=float(plan["lora_candidate"]["adapter_weight"]),
        run_dir=Path(plan["lora_candidate"]["run_dir"]),
        lora_dir=Path(plan["lora_candidate"]["lora_dir"]),
        weights_path=Path(plan["lora_candidate"]["weights_path"]),
    )
    run_dir = make_tiny_diagnostic_run_dir(run_root)
    manifest = {
        **plan,
        "dry_run": False,
        "validation_run_dir": str(run_dir),
        "inputs_dir": str(run_dir / "inputs"),
    }
    manifest_path = run_dir / "diagnostic_manifest.json"
    write_json(manifest_path, manifest)

    prepared_windows = prepare_flux_canny_lanpaint_validation_inputs(cfg, run_dir / "inputs")
    prepared_window = prepared_windows[0]

    native_pipe, native_dtype = load_flux_canny_inpaint_pipeline(cfg)
    try:
        record = run_native_tiny_case(
            pipe=native_pipe,
            cfg=cfg,
            prepared_window=prepared_window,
            run_dir=run_dir,
            case_id="A_native_no_lora",
            torch_dtype=native_dtype,
            lora=None,
        )
        manifest["completed_records"].append(record)
        write_json(manifest_path, manifest)

        load_info = _load_native_lora_adapter(native_pipe, lora)
        manifest["adapter_events"].append(
            {"event": "load", "backend": "native_diffusers_inpaint", **load_info}
        )
        write_json(manifest_path, manifest)
        record = run_native_tiny_case(
            pipe=native_pipe,
            cfg=cfg,
            prepared_window=prepared_window,
            run_dir=run_dir,
            case_id=f"C_native_rank{lora.rank}_checkpoint{lora.checkpoint_step}_{lora_scale_slug(lora.adapter_weight)}",
            torch_dtype=native_dtype,
            lora=lora,
        )
        manifest["completed_records"].append(record)
        cleanup = remove_inpaint_lora_adapter(native_pipe, str(load_info["adapter_name"]))
        manifest["adapter_events"].append(
            {"event": "cleanup", "backend": "native_diffusers_inpaint", **cleanup}
        )
        write_json(manifest_path, manifest)
    finally:
        del native_pipe
        clear_cuda_cache()

    lanpaint_pipe, lanpaint_dtype = load_flux_canny_lanpaint_pipeline(cfg)
    inspect_flux_canny_lanpaint_pipeline(lanpaint_pipe, cfg, run_dir, lora_enabled=False)
    try:
        for mode in plan["big_branch_modes"]:
            stage = f"B_lanpaint_no_lora_big_{mode}"
            stage_dir, window_results = run_flux_canny_lanpaint_prepared_validation(
                lanpaint_pipe,
                cfg,
                run_dir,
                stage=stage,
                prepared_windows=prepared_windows,
                lora_dir=None,
                lora_enabled=False,
                torch_dtype=lanpaint_dtype,
                update_run_metadata=False,
                big_branch_mode=mode,
                big_guidance_scale=1.0,
            )
            for window_result in window_results:
                manifest["completed_records"].append(
                    {
                        "case_id": stage,
                        "backend": "lanpaint_adapter",
                        "big_branch_mode": mode,
                        "lora_enabled": False,
                        "stage_dir": str(stage_dir),
                        **window_result,
                    }
                )
            write_json(manifest_path, manifest)

        load_info = load_flux_canny_lora_adapter(
            lanpaint_pipe,
            lora.lora_dir,
            adapter_name=lora.label,
            adapter_weight=lora.adapter_weight,
        )
        manifest["adapter_events"].append(
            {"event": "load", "backend": "lanpaint_adapter", **load_info}
        )
        write_json(manifest_path, manifest)
        for mode in plan["big_branch_modes"]:
            stage = (
                f"D_lanpaint_rank{lora.rank}_checkpoint{lora.checkpoint_step}_"
                f"{lora_scale_slug(lora.adapter_weight)}_big_{mode}"
            )
            cfg_for_lora = replace(cfg, rank=int(lora.rank))
            stage_dir, window_results = run_flux_canny_lanpaint_prepared_validation(
                lanpaint_pipe,
                cfg_for_lora,
                run_dir,
                stage=stage,
                prepared_windows=prepared_windows,
                lora_dir=lora.lora_dir,
                lora_enabled=True,
                torch_dtype=lanpaint_dtype,
                update_run_metadata=False,
                big_branch_mode=mode,
                big_guidance_scale=1.0,
            )
            for window_result in window_results:
                manifest["completed_records"].append(
                    {
                        "case_id": stage,
                        "backend": "lanpaint_adapter",
                        "big_branch_mode": mode,
                        "lora_enabled": True,
                        "lora": lora.to_manifest(),
                        "stage_dir": str(stage_dir),
                        **window_result,
                    }
                )
            write_json(manifest_path, manifest)

        cleanup = remove_flux_canny_lora_adapter(
            lanpaint_pipe,
            adapter_name=str(load_info["adapter_name"]),
            named_adapter=bool(load_info["named_adapter"]),
        )
        manifest["adapter_events"].append(
            {"event": "cleanup", "backend": "lanpaint_adapter", **cleanup}
        )
        manifest["completed"] = True
        manifest["completed_record_count"] = len(manifest["completed_records"])
        write_json(manifest_path, manifest)
        return manifest
    finally:
        del lanpaint_pipe
        clear_cuda_cache()


__all__ = [
    "BIG_BRANCH_MODES",
    "TinyDiagnosticLoraCandidate",
    "discover_tiny_diagnostic_lora",
    "flux_schedule_records",
    "flux_time_record",
    "load_tiny_native_vs_lanpaint_config",
    "lora_scale_slug",
    "plan_flux_canny_native_vs_lanpaint_tiny_diagnostic",
    "planned_tiny_diagnostic_records",
    "run_flux_canny_native_vs_lanpaint_tiny_diagnostic",
    "validate_big_branch_mode",
]
