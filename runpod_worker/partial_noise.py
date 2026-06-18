"""FLUX Fill pipeline variant with explicit partial-noise schedule slicing."""

from __future__ import annotations

from typing import Any

try:
    from diffusers import FluxFillPipeline
except Exception:  # pragma: no cover - exercised only in minimal import environments.

    class FluxFillPipeline:  # type: ignore[no-redef]
        """Placeholder so this module remains importable without optional dependencies."""


def normalize_partial_noise(value: Any) -> float:
    """Return a validated partial-noise fraction, defaulting to the full schedule."""

    if value is None:
        return 1.0
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("partial_noise must be a number in the interval (0.0, 1.0].") from exc
    if not 0.0 < normalized <= 1.0:
        raise ValueError("partial_noise must be in the interval (0.0, 1.0].")
    return normalized


def partial_noise_start_index(schedule_length: int, partial_noise: Any) -> int:
    """Return the scheduler start index used by the experimental notebooks."""

    length = int(schedule_length)
    if length <= 0:
        raise ValueError("schedule_length must be positive.")
    fraction = normalize_partial_noise(partial_noise)
    begin_index = int(round(length * (1.0 - fraction)))
    return max(0, min(begin_index, length - 1))


def _is_torch_tensor(value: Any) -> bool:
    try:
        import torch

        return bool(torch.is_tensor(value))
    except Exception:
        return False


def _scalar_float(value: Any) -> float | list[float] | None:
    if value is None:
        return None
    if _is_torch_tensor(value):
        tensor = value.detach().float().cpu()
        if tensor.numel() != 1:
            return [float(item) for item in tensor.flatten().tolist()]
        return float(tensor.item())
    return float(value)


def _as_float_list(values: Any) -> list[float] | None:
    if values is None:
        return None
    if _is_torch_tensor(values):
        return [float(item) for item in values.detach().float().cpu().flatten().tolist()]
    return [float(item) for item in values]


def build_schedule_rows(
    full_timesteps: Any,
    full_sigmas: Any,
    start_index: int,
) -> list[dict[str, Any]]:
    """Build JSON-safe schedule rows matching the notebook debug metadata."""

    timestep_values = _as_float_list(full_timesteps) or []
    sigma_values = _as_float_list(full_sigmas)
    rows: list[dict[str, Any]] = []
    for idx, timestep in enumerate(timestep_values):
        if idx < start_index:
            status = "skip"
        elif idx == start_index:
            status = "START"
        else:
            status = "run"
        sigma = sigma_values[idx] if sigma_values is not None and idx < len(sigma_values) else None
        rows.append(
            {
                "index": int(idx),
                "timestep": timestep,
                "sigma": sigma,
                "selected_start": bool(idx == start_index),
                "will_run": bool(idx >= start_index),
                "status": status,
            }
        )
    return rows


class FluxFillPartialNoisePipeline(FluxFillPipeline):
    """Flux Fill pipeline with the notebook-local partial-noise schedule control."""

    print_timestep_schedule = False
    partial_noise = 1.0
    partial_noise_fraction = 1.0

    def _requested_partial_noise(self) -> float:
        value = getattr(self, "partial_noise", None)
        if value is None:
            value = getattr(self, "partial_noise_fraction", 1.0)
        return normalize_partial_noise(value)

    def _print_schedule(self, rows: list[dict[str, Any]]) -> None:
        print("Flux Fill scheduler schedule; higher sigma/flow means more initial noise:")
        print(" idx | timestep | sigma/flow | status")
        print("-----+----------+------------+--------")
        for row in rows:
            timestep_text = "?" if row["timestep"] is None else f"{row['timestep']:8.3f}"
            sigma_text = "?" if row["sigma"] is None else f"{row['sigma']:10.6f}"
            print(f"{row['index']:4d} | {timestep_text} | {sigma_text} | {row['status']}")

    def get_timesteps(self, num_inference_steps: int, strength: float, device: Any):
        full_timesteps = self.scheduler.timesteps.detach().clone()
        scheduler_sigmas = getattr(self.scheduler, "sigmas", None)
        full_sigmas = (
            scheduler_sigmas.detach().clone()
            if _is_torch_tensor(scheduler_sigmas)
            else scheduler_sigmas
        )

        requested_fraction = self._requested_partial_noise()
        full_num_steps = int(len(full_timesteps))
        begin_index = partial_noise_start_index(full_num_steps, requested_fraction)
        timesteps = self.scheduler.timesteps[begin_index:]
        if hasattr(self.scheduler, "set_begin_index"):
            self.scheduler.set_begin_index(begin_index)

        effective_num_steps = int(len(timesteps))
        schedule_rows = build_schedule_rows(full_timesteps, full_sigmas, begin_index)
        selected_row = schedule_rows[begin_index] if 0 <= begin_index < len(schedule_rows) else None
        self._last_schedule_debug = {
            "partial_noise": requested_fraction,
            "partial_noise_fraction": requested_fraction,
            "requested_partial_noise": requested_fraction,
            "requested_partial_noise_fraction": requested_fraction,
            "diffusers_strength_argument_ignored": float(strength),
            "requested_num_inference_steps": int(num_inference_steps),
            "start_selection": "custom_fractional_schedule_slice",
            "t_start": int(begin_index),
            "scheduler_order": int(getattr(self.scheduler, "order", 1)),
            "scheduler_begin_index": int(begin_index),
            "scheduler_begin_index_set": hasattr(self.scheduler, "set_begin_index"),
            "full_num_steps": full_num_steps,
            "effective_num_steps": effective_num_steps,
            "first_selected_timestep": selected_row.get("timestep") if selected_row else None,
            "first_selected_sigma": selected_row.get("sigma") if selected_row else None,
            "schedule": schedule_rows,
            "sliced_schedule": schedule_rows[begin_index:],
        }
        if self.print_timestep_schedule:
            self._print_schedule(schedule_rows)
            print(
                "Schedule slice: "
                f"partial_noise={requested_fraction:.3f}, "
                f"start_idx={begin_index}/{len(schedule_rows) - 1}, "
                f"steps={effective_num_steps}/{full_num_steps}."
            )
        return timesteps, effective_num_steps

    def prepare_latents(self, *args: Any, **kwargs: Any):
        timestep = args[1] if len(args) > 1 else kwargs.get("timestep")
        timestep_for_debug = (
            timestep[:1] if _is_torch_tensor(timestep) and timestep.numel() else timestep
        )
        height = args[4] if len(args) > 4 else kwargs.get("height")
        width = args[5] if len(args) > 5 else kwargs.get("width")
        batch_size = args[2] if len(args) > 2 else kwargs.get("batch_size")
        channels = args[3] if len(args) > 3 else kwargs.get("num_channels_latents")
        dtype = args[6] if len(args) > 6 else kwargs.get("dtype")
        device = args[7] if len(args) > 7 else kwargs.get("device")
        latents = args[9] if len(args) > 9 else kwargs.get("latents")
        self._last_latent_init_debug = {
            "initialization": "source_image_noised_to_selected_timestep_by_scheduler.scale_noise",
            "latents_provided": latents is not None,
            "latent_timestep": _scalar_float(timestep_for_debug),
            "batch_size": int(batch_size) if batch_size is not None else None,
            "num_channels_latents": int(channels) if channels is not None else None,
            "height": int(height) if height is not None else None,
            "width": int(width) if width is not None else None,
            "dtype": str(dtype),
            "device": str(device),
        }
        return super().prepare_latents(*args, **kwargs)


def set_partial_noise(pipe: Any, value: Any) -> float:
    """Set both historical and request-facing partial-noise attributes on a pipeline."""

    normalized = normalize_partial_noise(value)
    pipe.partial_noise = normalized
    pipe.partial_noise_fraction = normalized
    return normalized
