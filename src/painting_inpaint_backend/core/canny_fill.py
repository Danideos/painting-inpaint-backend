"""Hybrid FLUX-Canny/LanPaint followed by FLUX Fill refinement."""

from __future__ import annotations

import math
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from .canny_lanpaint import (
    CANNY_BLUR_RADIUS,
    CANNY_HIGH_THRESHOLD,
    CANNY_LOW_THRESHOLD,
    LANPAINT_BETA,
    LANPAINT_FINAL_OUTER_STEPS_WITHOUT_INNER,
    LANPAINT_FRICTION,
    LANPAINT_INNER_STEPS,
    LANPAINT_LAMBDA,
    LANPAINT_STEP_SIZE,
    FluxCannyLanPaintService,
)
from .image_io import image_from_base64, image_to_base64
from .inference import InferenceService, WorkerInputError
from .methods import (
    FLUX_CANNY_FILL_METHOD,
    FLUX_CANNY_LANPAINT_METHOD,
    FLUX_CANNY_LANPAINT_NATIVE_METHOD,
    FLUX_FILL_CANNY_FILL_METHOD,
    FLUX_FILL_CANNY_NATIVE_METHOD,
    FLUX_FILL_METHOD,
    normalize_method,
)
from .partial_noise import normalize_partial_noise
from .progress import ProgressReporter

CANNY_FILL_DEFAULT_CANNY_GUIDANCE_SCALE = 7.0
CANNY_FILL_DEFAULT_CANNY_PARTIAL_NOISE = 1.0
CANNY_FILL_DEFAULT_CANNY_STEPS = 30
CANNY_FILL_DEFAULT_FILL_GUIDANCE_SCALE = 30.0
CANNY_FILL_DEFAULT_FILL_PARTIAL_NOISE = 0.4
CANNY_FILL_DEFAULT_FILL_STEPS = 50

_REMOTE_LORA_ENV_KEYS = (
    "LORA_REPO_ID",
    "LORA_FILENAME",
    "LORA_REVISION",
    "LORA_CACHE_DIR",
    "LORA_WEIGHT_NAME",
)


@dataclass(frozen=True)
class CannyFillRequestSettings:
    """Validated controls for the two-stage Canny/LanPaint then Fill workflow."""

    prompt: str
    seed: int | None
    output_format: str
    canny_prompt: str
    canny_partial_noise: float
    canny_guidance_scale: float
    canny_num_inference_steps: int
    canny_lora_scale: float
    canny_max_sequence_length: int
    fill_prompt: str
    fill_negative_prompt: str
    fill_partial_noise: float
    fill_guidance_scale: float
    fill_num_inference_steps: int
    fill_lora_scale: float
    fill_max_sequence_length: int
    # Per-stage overrides for flux_fill_canny_fill (fall back to fill_* if not set)
    fill_pre_prompt: str
    fill_pre_negative_prompt: str
    fill_pre_partial_noise: float
    fill_pre_guidance_scale: float
    fill_pre_num_inference_steps: int
    fill_pre_lora_scale: float
    fill_post_prompt: str
    fill_post_negative_prompt: str
    fill_post_partial_noise: float
    fill_post_guidance_scale: float
    fill_post_num_inference_steps: int
    fill_post_lora_scale: float
    canny_low_threshold: int
    canny_high_threshold: int
    canny_blur_radius: float
    canny_control_strategy: str | None
    skip_fill_pre: bool
    lanpaint_inner_steps: int
    lanpaint_friction: float
    lanpaint_lambda: float
    lanpaint_beta: float
    lanpaint_step_size: float
    lanpaint_final_outer_steps_without_inner: int


def _payload_float(payload: dict[str, Any], key: str, default: float) -> float:
    value = payload.get(key, default)
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise WorkerInputError(f"{key} must be a number.") from exc
    if not math.isfinite(number):
        raise WorkerInputError(f"{key} must be finite.")
    return number


def _payload_int(payload: dict[str, Any], key: str, default: int) -> int:
    value = payload.get(key, default)
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise WorkerInputError(f"{key} must be an integer.") from exc
    return parsed


def _optional_prompt(
    payload: dict[str, Any],
    key: str,
    default: str,
    *,
    allow_empty: bool = False,
) -> str:
    value = payload.get(key, default)
    if not isinstance(value, str):
        raise WorkerInputError(f"{key} must be a string.")
    if not allow_empty and not value.strip():
        raise WorkerInputError(f"{key} must not be empty.")
    return value


def _optional_non_negative_float(payload: dict[str, Any], key: str, default: float) -> float:
    value = _payload_float(payload, key, default)
    if value < 0:
        raise WorkerInputError(f"{key} must be non-negative.")
    return value


def _optional_positive_int(payload: dict[str, Any], key: str, default: int) -> int:
    value = _payload_int(payload, key, default)
    if value <= 0:
        raise WorkerInputError(f"{key} must be positive.")
    return value


def _parse_seed(payload: dict[str, Any]) -> int | None:
    value = payload.get("seed")
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise WorkerInputError("seed must be an integer when provided.") from exc


def _parse_output_format(payload: dict[str, Any]) -> str:
    from .image_io import normalize_output_format

    try:
        return normalize_output_format(payload.get("output_format", "png"))
    except ValueError as exc:
        raise WorkerInputError(str(exc)) from exc


def _parse_partial_noise(
    payload: dict[str, Any],
    key: str,
    default: float,
) -> float:
    try:
        return normalize_partial_noise(payload.get(key, default))
    except ValueError as exc:
        raise WorkerInputError(str(exc)) from exc


def _parse_thresholds(payload: dict[str, Any]) -> tuple[int, int]:
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
    return low, high


def parse_canny_fill_settings(
    payload: dict[str, Any],
    method: str | None = None,
) -> CannyFillRequestSettings:
    """Parse the provider-neutral hybrid method controls."""

    try:
        parsed_method = normalize_method(method or payload.get("method"))
    except ValueError as exc:
        raise WorkerInputError(str(exc)) from exc
    if parsed_method not in {
        FLUX_CANNY_FILL_METHOD,
        FLUX_FILL_CANNY_NATIVE_METHOD,
        FLUX_FILL_CANNY_FILL_METHOD,
    }:
        raise WorkerInputError(
            "The hybrid service requires method='flux_canny_fill', "
            "'flux_fill_canny_native', or 'flux_fill_canny_fill'."
        )

    if "prompt" not in payload:
        raise WorkerInputError("prompt is required.")
    prompt = _optional_prompt(payload, "prompt", "")
    seed = _parse_seed(payload)
    output_format = _parse_output_format(payload)
    shared_steps = _optional_positive_int(
        payload,
        "num_inference_steps",
        CANNY_FILL_DEFAULT_FILL_STEPS,
    )
    shared_max_sequence_length = _optional_positive_int(payload, "max_sequence_length", 512)
    shared_lora_scale = _optional_non_negative_float(payload, "lora_scale", 1.0)

    canny_low, canny_high = _parse_thresholds(payload)
    canny_blur = _payload_float(payload, "canny_blur_radius", CANNY_BLUR_RADIUS)
    if canny_blur < 0:
        raise WorkerInputError("canny_blur_radius must be non-negative.")

    canny_control_strategy = payload.get("canny_control_strategy", None)
    if canny_control_strategy is not None and canny_control_strategy not in {
        "masked_ink",
        "plain_canny",
    }:
        raise WorkerInputError(
            "canny_control_strategy must be 'masked_ink', 'plain_canny', or omitted."
        )

    lanpaint_inner_steps = _payload_int(payload, "lanpaint_inner_steps", LANPAINT_INNER_STEPS)
    if lanpaint_inner_steps < 0:
        raise WorkerInputError("lanpaint_inner_steps must be non-negative.")
    lanpaint_final_without_inner = _payload_int(
        payload,
        "lanpaint_final_outer_steps_without_inner",
        LANPAINT_FINAL_OUTER_STEPS_WITHOUT_INNER,
    )
    if lanpaint_final_without_inner < 0:
        raise WorkerInputError(
            "lanpaint_final_outer_steps_without_inner must be non-negative."
        )
    lanpaint_friction = _optional_non_negative_float(
        payload,
        "lanpaint_friction",
        LANPAINT_FRICTION,
    )
    lanpaint_lambda = _optional_non_negative_float(
        payload,
        "lanpaint_lambda",
        LANPAINT_LAMBDA,
    )
    lanpaint_beta = _optional_non_negative_float(payload, "lanpaint_beta", LANPAINT_BETA)
    lanpaint_step_size = _payload_float(payload, "lanpaint_step_size", LANPAINT_STEP_SIZE)
    if lanpaint_step_size <= 0:
        raise WorkerInputError("lanpaint_step_size must be greater than 0.")

    fill_partial_default = payload.get("partial_noise", CANNY_FILL_DEFAULT_FILL_PARTIAL_NOISE)

    fill_prompt = _optional_prompt(payload, "fill_prompt", prompt)
    fill_negative_prompt = str(
        payload.get("fill_negative_prompt", payload.get("negative_prompt", ""))
    )
    fill_partial_noise = _parse_partial_noise(payload, "fill_partial_noise", fill_partial_default)
    fill_guidance_scale = _optional_non_negative_float(
        payload, "fill_guidance_scale", CANNY_FILL_DEFAULT_FILL_GUIDANCE_SCALE
    )
    fill_num_inference_steps = _optional_positive_int(
        payload, "fill_num_inference_steps", shared_steps or CANNY_FILL_DEFAULT_FILL_STEPS
    )
    fill_lora_scale = _optional_non_negative_float(payload, "fill_lora_scale", shared_lora_scale)

    return CannyFillRequestSettings(
        prompt=prompt,
        seed=seed,
        output_format=output_format,
        canny_prompt=_optional_prompt(
            payload,
            "canny_prompt",
            prompt,
            allow_empty=True,
        ),
        canny_partial_noise=_parse_partial_noise(
            payload,
            "canny_partial_noise",
            CANNY_FILL_DEFAULT_CANNY_PARTIAL_NOISE,
        ),
        canny_guidance_scale=_optional_non_negative_float(
            payload,
            "canny_guidance_scale",
            CANNY_FILL_DEFAULT_CANNY_GUIDANCE_SCALE,
        ),
        canny_num_inference_steps=_optional_positive_int(
            payload,
            "canny_num_inference_steps",
            shared_steps or CANNY_FILL_DEFAULT_CANNY_STEPS,
        ),
        canny_lora_scale=_optional_non_negative_float(
            payload,
            "canny_lora_scale",
            shared_lora_scale,
        ),
        canny_max_sequence_length=_optional_positive_int(
            payload,
            "canny_max_sequence_length",
            shared_max_sequence_length,
        ),
        fill_prompt=fill_prompt,
        fill_negative_prompt=fill_negative_prompt,
        fill_partial_noise=fill_partial_noise,
        fill_guidance_scale=fill_guidance_scale,
        fill_num_inference_steps=fill_num_inference_steps,
        fill_lora_scale=fill_lora_scale,
        fill_max_sequence_length=_optional_positive_int(
            payload,
            "fill_max_sequence_length",
            shared_max_sequence_length,
        ),
        fill_pre_prompt=_optional_prompt(payload, "fill_pre_prompt", fill_prompt),
        fill_pre_negative_prompt=str(
            payload.get("fill_pre_negative_prompt", fill_negative_prompt)
        ),
        fill_pre_partial_noise=_parse_partial_noise(
            payload, "fill_pre_partial_noise", fill_partial_noise
        ),
        fill_pre_guidance_scale=_optional_non_negative_float(
            payload, "fill_pre_guidance_scale", fill_guidance_scale
        ),
        fill_pre_num_inference_steps=_optional_positive_int(
            payload, "fill_pre_num_inference_steps", fill_num_inference_steps
        ),
        fill_pre_lora_scale=_optional_non_negative_float(
            payload, "fill_pre_lora_scale", fill_lora_scale
        ),
        fill_post_prompt=_optional_prompt(payload, "fill_post_prompt", fill_prompt),
        fill_post_negative_prompt=str(
            payload.get("fill_post_negative_prompt", fill_negative_prompt)
        ),
        fill_post_partial_noise=_parse_partial_noise(
            payload, "fill_post_partial_noise", fill_partial_noise
        ),
        fill_post_guidance_scale=_optional_non_negative_float(
            payload, "fill_post_guidance_scale", fill_guidance_scale
        ),
        fill_post_num_inference_steps=_optional_positive_int(
            payload, "fill_post_num_inference_steps", fill_num_inference_steps
        ),
        fill_post_lora_scale=_optional_non_negative_float(
            payload, "fill_post_lora_scale", fill_lora_scale
        ),
        canny_low_threshold=canny_low,
        canny_high_threshold=canny_high,
        canny_blur_radius=canny_blur,
        canny_control_strategy=canny_control_strategy,
        skip_fill_pre=bool(payload.get("skip_fill_pre", False)),
        lanpaint_inner_steps=lanpaint_inner_steps,
        lanpaint_friction=lanpaint_friction,
        lanpaint_lambda=lanpaint_lambda,
        lanpaint_beta=lanpaint_beta,
        lanpaint_step_size=lanpaint_step_size,
        lanpaint_final_outer_steps_without_inner=lanpaint_final_without_inner,
    )


@contextmanager
def _temporary_environ(updates: dict[str, str | None] | None):
    if not updates:
        yield
        return
    previous = {key: os.environ.get(key) for key in updates}
    try:
        for key, value in updates.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _clear_remote_lora_env() -> dict[str, None]:
    return dict.fromkeys(_REMOTE_LORA_ENV_KEYS)


def _canny_payload(
    payload: dict[str, Any],
    settings: CannyFillRequestSettings,
) -> dict[str, Any]:
    canny_payload = dict(payload)
    canny_payload.update(
        {
            "method": FLUX_CANNY_LANPAINT_METHOD,
            "prompt": settings.canny_prompt,
            "partial_noise": settings.canny_partial_noise,
            "guidance_scale": settings.canny_guidance_scale,
            "num_inference_steps": settings.canny_num_inference_steps,
            "max_sequence_length": settings.canny_max_sequence_length,
            "lora_scale": settings.canny_lora_scale,
            "output_format": "png",
            "canny_low_threshold": settings.canny_low_threshold,
            "canny_high_threshold": settings.canny_high_threshold,
            "canny_blur_radius": settings.canny_blur_radius,
            "lanpaint_inner_steps": settings.lanpaint_inner_steps,
            "lanpaint_friction": settings.lanpaint_friction,
            "lanpaint_lambda": settings.lanpaint_lambda,
            "lanpaint_beta": settings.lanpaint_beta,
            "lanpaint_step_size": settings.lanpaint_step_size,
            "lanpaint_final_outer_steps_without_inner": (
                settings.lanpaint_final_outer_steps_without_inner
            ),
        }
    )
    canny_payload.pop("fill_prompt", None)
    canny_payload.pop("fill_negative_prompt", None)
    return canny_payload


def _fill_payload(
    payload: dict[str, Any],
    settings: CannyFillRequestSettings,
    image_base64: str,
    *,
    prompt: str | None = None,
    negative_prompt: str | None = None,
    partial_noise: float | None = None,
    guidance_scale: float | None = None,
    num_inference_steps: int | None = None,
    lora_scale: float | None = None,
) -> dict[str, Any]:
    fill_payload = dict(payload)
    for key in (
        "image_url",
        "control_image_url",
        "control_image_base64",
        "canny_prompt",
        "canny_partial_noise",
        "canny_guidance_scale",
        "canny_num_inference_steps",
        "canny_max_sequence_length",
        "canny_lora_scale",
        "canny_low_threshold",
        "canny_high_threshold",
        "canny_blur_radius",
        "lanpaint_inner_steps",
        "lanpaint_friction",
        "lanpaint_lambda",
        "lanpaint_beta",
        "lanpaint_step_size",
        "lanpaint_final_outer_steps_without_inner",
        "fill_partial_noise",
        "fill_guidance_scale",
        "fill_num_inference_steps",
        "fill_max_sequence_length",
        "fill_lora_scale",
        "fill_pre_prompt",
        "fill_pre_negative_prompt",
        "fill_pre_partial_noise",
        "fill_pre_guidance_scale",
        "fill_pre_num_inference_steps",
        "fill_pre_lora_scale",
        "fill_post_prompt",
        "fill_post_negative_prompt",
        "fill_post_partial_noise",
        "fill_post_guidance_scale",
        "fill_post_num_inference_steps",
        "fill_post_lora_scale",
    ):
        fill_payload.pop(key, None)
    fill_payload.update(
        {
            "method": FLUX_FILL_METHOD,
            "image_base64": image_base64,
            "prompt": prompt if prompt is not None else settings.fill_prompt,
            "negative_prompt": (
                negative_prompt if negative_prompt is not None else settings.fill_negative_prompt
            ),
            "partial_noise": (
                partial_noise if partial_noise is not None else settings.fill_partial_noise
            ),
            "guidance_scale": (
                guidance_scale if guidance_scale is not None else settings.fill_guidance_scale
            ),
            "num_inference_steps": (
                num_inference_steps
                if num_inference_steps is not None
                else settings.fill_num_inference_steps
            ),
            "max_sequence_length": settings.fill_max_sequence_length,
            "lora_scale": lora_scale if lora_scale is not None else settings.fill_lora_scale,
            "output_format": settings.output_format,
        }
    )
    return fill_payload


class FluxCannyFillService:
    """Run full FLUX-Canny/LanPaint, then refine that result with FLUX Fill."""

    def __init__(
        self,
        *,
        canny_env: dict[str, str | None] | None = None,
        fill_env: dict[str, str | None] | None = None,
        canny_service: FluxCannyLanPaintService | None = None,
        fill_service: InferenceService | None = None,
    ) -> None:
        self.canny_env = {**_clear_remote_lora_env(), **(canny_env or {})}
        self.fill_env = {**_clear_remote_lora_env(), **(fill_env or {})}
        self.canny_service = canny_service or FluxCannyLanPaintService()
        self.fill_service = fill_service or InferenceService()
        self._lock = threading.Lock()

    def run(
        self,
        payload: dict[str, Any],
        *,
        reporter: ProgressReporter | None = None,
    ) -> dict[str, Any]:
        reporter = reporter or ProgressReporter(enabled=False)
        settings = parse_canny_fill_settings(payload)
        started = time.perf_counter()

        with self._lock:
            reporter.emit(
                "hybrid_canny_start",
                stage="inference",
                message="Starting FLUX-Canny restoration stage.",
                metadata={"method": FLUX_CANNY_FILL_METHOD},
            )
            with _temporary_environ(self.canny_env):
                canny_output = self.canny_service.run(
                    _canny_payload(payload, settings),
                    reporter=reporter,
                )
            reporter.emit(
                "hybrid_canny_done",
                stage="inference",
                message="FLUX-Canny restoration stage completed.",
                metadata={
                    "width": canny_output.get("width"),
                    "height": canny_output.get("height"),
                    "outside_mask_changed_after_hard_composite": canny_output.get(
                        "outside_mask_changed_after_hard_composite"
                    ),
                },
            )

            canny_intermediate = image_from_base64(
                str(canny_output["image_base64"]),
                label="canny intermediate",
            ).convert("RGB")
            canny_intermediate_base64 = image_to_base64(
                canny_intermediate,
                output_format="png",
            )

            reporter.emit(
                "hybrid_fill_start",
                stage="inference",
                message="Starting FLUX Fill refinement stage.",
                metadata={
                    "method": FLUX_CANNY_FILL_METHOD,
                    "fill_partial_noise": settings.fill_partial_noise,
                    "fill_num_inference_steps": settings.fill_num_inference_steps,
                },
            )
            with _temporary_environ(self.fill_env):
                fill_output = self.fill_service.run(
                    _fill_payload(payload, settings, canny_intermediate_base64),
                    reporter=reporter,
                )
            reporter.emit(
                "hybrid_fill_done",
                stage="inference",
                message="FLUX Fill refinement stage completed.",
                metadata={
                    "width": fill_output.get("width"),
                    "height": fill_output.get("height"),
                    "outside_mask_changed_after_hard_composite": fill_output.get(
                        "outside_mask_changed_after_hard_composite"
                    ),
                },
            )

        elapsed = time.perf_counter() - started
        output = dict(fill_output)
        output["canny_image_base64"] = canny_intermediate_base64
        output["canny_output_format"] = "png"
        output["timings"] = {
            "hybrid_total_seconds": elapsed,
            "canny": canny_output.get("timings", {}),
            "fill": fill_output.get("timings", {}),
        }
        output["gpu_memory"] = {
            "canny": canny_output.get("gpu_memory", {}),
            "fill": fill_output.get("gpu_memory", {}),
        }
        output["model"] = {
            "method": FLUX_CANNY_FILL_METHOD,
            "canny": canny_output.get("model", {}),
            "fill": fill_output.get("model", {}),
        }
        output["lora"] = {
            "canny": canny_output.get("lora", {}),
            "fill": fill_output.get("lora", {}),
        }
        output["inference_settings"] = {
            "method": FLUX_CANNY_FILL_METHOD,
            "prompt": settings.prompt,
            "seed": settings.seed,
            "output_format": settings.output_format,
            "canny": {
                "prompt": settings.canny_prompt,
                "partial_noise": settings.canny_partial_noise,
                "guidance_scale": settings.canny_guidance_scale,
                "num_inference_steps": settings.canny_num_inference_steps,
                "max_sequence_length": settings.canny_max_sequence_length,
                "lora_scale": settings.canny_lora_scale,
                "low_threshold": settings.canny_low_threshold,
                "high_threshold": settings.canny_high_threshold,
                "blur_radius": settings.canny_blur_radius,
                "control_source": canny_output.get("inference_settings", {}).get(
                    "control_source"
                ),
                "backend_revision": canny_output.get("inference_settings", {}).get(
                    "backend_revision"
                ),
                "control_strategy": canny_output.get("inference_settings", {}).get(
                    "control_strategy"
                ),
                "latent_source_strategy": canny_output.get("inference_settings", {}).get(
                    "latent_source_strategy"
                ),
                "latent_source_fill_radius": canny_output.get("inference_settings", {}).get(
                    "latent_source_fill_radius"
                ),
            },
            "fill": {
                "prompt": settings.fill_prompt,
                "negative_prompt": settings.fill_negative_prompt,
                "partial_noise": settings.fill_partial_noise,
                "guidance_scale": settings.fill_guidance_scale,
                "num_inference_steps": settings.fill_num_inference_steps,
                "max_sequence_length": settings.fill_max_sequence_length,
                "lora_scale": settings.fill_lora_scale,
            },
            "lanpaint": {
                "inner_steps": settings.lanpaint_inner_steps,
                "friction": settings.lanpaint_friction,
                "lambda": settings.lanpaint_lambda,
                "beta": settings.lanpaint_beta,
                "step_size": settings.lanpaint_step_size,
                "final_outer_steps_without_inner": (
                    settings.lanpaint_final_outer_steps_without_inner
                ),
            },
        }
        output["schedule_debug"] = {
            "canny": canny_output.get("schedule_debug", {}),
            "fill": fill_output.get("schedule_debug", {}),
        }
        output["latent_init_debug"] = {
            "canny": canny_output.get("latent_init_debug", {}),
            "fill": fill_output.get("latent_init_debug", {}),
        }
        output["hybrid_intermediate"] = {
            "canny_width": canny_output.get("width"),
            "canny_height": canny_output.get("height"),
            "canny_output_format": canny_output.get("output_format"),
            "canny_outside_mask_changed_after_hard_composite": canny_output.get(
                "outside_mask_changed_after_hard_composite"
            ),
        }
        output["outside_mask_changed_after_hard_composite"] = fill_output.get(
            "outside_mask_changed_after_hard_composite"
        )
        output.pop("run_report", None)
        return output


class FluxFillCannyNativeService:
    """Run FLUX Fill, then native FLUX-Canny/LanPaint using the filled source."""

    def __init__(
        self,
        *,
        canny_env: dict[str, str | None] | None = None,
        fill_env: dict[str, str | None] | None = None,
        canny_service: FluxCannyLanPaintService | None = None,
        fill_service: InferenceService | None = None,
    ) -> None:
        self.canny_env = {**_clear_remote_lora_env(), **(canny_env or {})}
        self.fill_env = {**_clear_remote_lora_env(), **(fill_env or {})}
        self.canny_service = canny_service or FluxCannyLanPaintService(
            method=FLUX_CANNY_LANPAINT_NATIVE_METHOD,
            native_lanpaint=True,
        )
        self.fill_service = fill_service or InferenceService()
        self._lock = threading.Lock()

    def run(
        self,
        payload: dict[str, Any],
        *,
        reporter: ProgressReporter | None = None,
    ) -> dict[str, Any]:
        reporter = reporter or ProgressReporter(enabled=False)
        settings = parse_canny_fill_settings(payload, method=FLUX_FILL_CANNY_NATIVE_METHOD)
        started = time.perf_counter()

        original_image_base64 = payload.get("image_base64", "")

        with self._lock:
            reporter.emit(
                "hybrid_fill_start",
                stage="inference",
                message="Starting FLUX Fill pre-fill stage.",
                metadata={"method": FLUX_FILL_CANNY_NATIVE_METHOD},
            )
            with _temporary_environ(self.fill_env):
                fill_output = self.fill_service.run(
                    _fill_payload(payload, settings, original_image_base64),
                    reporter=reporter,
                )
            reporter.emit(
                "hybrid_fill_done",
                stage="inference",
                message="FLUX Fill pre-fill stage completed.",
                metadata={
                    "width": fill_output.get("width"),
                    "height": fill_output.get("height"),
                    "outside_mask_changed_after_hard_composite": fill_output.get(
                        "outside_mask_changed_after_hard_composite"
                    ),
                },
            )

            fill_intermediate_base64 = str(fill_output["image_base64"])

            reporter.emit(
                "hybrid_canny_start",
                stage="inference",
                message="Starting FLUX-Canny/LanPaint native stage.",
                metadata={
                    "method": FLUX_FILL_CANNY_NATIVE_METHOD,
                    "canny_partial_noise": settings.canny_partial_noise,
                    "canny_num_inference_steps": settings.canny_num_inference_steps,
                },
            )
            with _temporary_environ(self.canny_env):
                canny_output = self.canny_service.run(
                    _fill_canny_native_payload(
                        payload,
                        settings,
                        fill_intermediate_base64,
                        original_image_base64,
                    ),
                    reporter=reporter,
                )
            reporter.emit(
                "hybrid_canny_done",
                stage="inference",
                message="FLUX-Canny/LanPaint native stage completed.",
                metadata={
                    "width": canny_output.get("width"),
                    "height": canny_output.get("height"),
                    "outside_mask_changed_after_hard_composite": canny_output.get(
                        "outside_mask_changed_after_hard_composite"
                    ),
                },
            )

        elapsed = time.perf_counter() - started
        output = dict(canny_output)
        output["fill_intermediate_base64"] = fill_intermediate_base64
        output["fill_intermediate_format"] = "png"
        output["timings"] = {
            "hybrid_total_seconds": elapsed,
            "fill": fill_output.get("timings", {}),
            "canny": canny_output.get("timings", {}),
        }
        output["gpu_memory"] = {
            "fill": fill_output.get("gpu_memory", {}),
            "canny": canny_output.get("gpu_memory", {}),
        }
        output["model"] = {
            "method": FLUX_FILL_CANNY_NATIVE_METHOD,
            "fill": fill_output.get("model", {}),
            "canny": canny_output.get("model", {}),
        }
        output["lora"] = {
            "fill": fill_output.get("lora", {}),
            "canny": canny_output.get("lora", {}),
        }
        output["inference_settings"] = {
            "method": FLUX_FILL_CANNY_NATIVE_METHOD,
            "prompt": settings.prompt,
            "seed": settings.seed,
            "output_format": settings.output_format,
            "fill": {
                "prompt": settings.fill_prompt,
                "negative_prompt": settings.fill_negative_prompt,
                "partial_noise": settings.fill_partial_noise,
                "guidance_scale": settings.fill_guidance_scale,
                "num_inference_steps": settings.fill_num_inference_steps,
                "max_sequence_length": settings.fill_max_sequence_length,
                "lora_scale": settings.fill_lora_scale,
            },
            "canny": {
                "prompt": settings.canny_prompt,
                "partial_noise": settings.canny_partial_noise,
                "guidance_scale": settings.canny_guidance_scale,
                "num_inference_steps": settings.canny_num_inference_steps,
                "max_sequence_length": settings.canny_max_sequence_length,
                "lora_scale": settings.canny_lora_scale,
                "low_threshold": settings.canny_low_threshold,
                "high_threshold": settings.canny_high_threshold,
                "blur_radius": settings.canny_blur_radius,
                "backend_revision": canny_output.get("inference_settings", {}).get(
                    "backend_revision"
                ),
                "latent_source_strategy": canny_output.get("inference_settings", {}).get(
                    "latent_source_strategy"
                ),
            },
            "lanpaint": {
                "inner_steps": settings.lanpaint_inner_steps,
                "friction": settings.lanpaint_friction,
                "lambda": settings.lanpaint_lambda,
                "beta": settings.lanpaint_beta,
                "step_size": settings.lanpaint_step_size,
                "final_outer_steps_without_inner": (
                    settings.lanpaint_final_outer_steps_without_inner
                ),
            },
        }
        output["outside_mask_changed_after_hard_composite"] = canny_output.get(
            "outside_mask_changed_after_hard_composite"
        )
        output.pop("run_report", None)
        return output


class FluxFillCannyFillService:
    """Three-stage: FLUX Fill pre-fill → FLUX-Canny/LanPaint native → FLUX Fill refinement."""

    def __init__(
        self,
        *,
        canny_env: dict[str, str | None] | None = None,
        fill_env: dict[str, str | None] | None = None,
        canny_service: FluxCannyLanPaintService | None = None,
        fill_service: InferenceService | None = None,
    ) -> None:
        self.canny_env = {**_clear_remote_lora_env(), **(canny_env or {})}
        self.fill_env = {**_clear_remote_lora_env(), **(fill_env or {})}
        self.canny_service = canny_service or FluxCannyLanPaintService(
            method=FLUX_CANNY_LANPAINT_NATIVE_METHOD,
            native_lanpaint=True,
        )
        self.fill_service = fill_service or InferenceService()
        self._lock = threading.Lock()

    def run(
        self,
        payload: dict[str, Any],
        *,
        reporter: ProgressReporter | None = None,
    ) -> dict[str, Any]:
        reporter = reporter or ProgressReporter(enabled=False)
        settings = parse_canny_fill_settings(payload, method=FLUX_FILL_CANNY_FILL_METHOD)
        started = time.perf_counter()

        original_image_base64 = payload.get("image_base64", "")

        with self._lock:
            # Stage 1: Fill to replace white/missing content with plausible pixels.
            # Skipped when skip_fill_pre=True — the original image is used directly.
            if settings.skip_fill_pre:
                reporter.emit(
                    "hybrid_fill_pre_start",
                    stage="inference",
                    message="Skipping FLUX Fill pre-fill stage (skip_fill_pre=True).",
                    metadata={"method": FLUX_FILL_CANNY_FILL_METHOD, "stage": 1},
                )
                fill_pre_base64 = original_image_base64
                fill_pre_base64_for_response: str | None = None
            else:
                reporter.emit(
                    "hybrid_fill_pre_start",
                    stage="inference",
                    message="Starting FLUX Fill pre-fill stage.",
                    metadata={"method": FLUX_FILL_CANNY_FILL_METHOD, "stage": 1},
                )
                with _temporary_environ(self.fill_env):
                    fill_pre_output = self.fill_service.run(
                        _fill_payload(
                            payload,
                            settings,
                            original_image_base64,
                            prompt=settings.fill_pre_prompt,
                            negative_prompt=settings.fill_pre_negative_prompt,
                            partial_noise=settings.fill_pre_partial_noise,
                            guidance_scale=settings.fill_pre_guidance_scale,
                            num_inference_steps=settings.fill_pre_num_inference_steps,
                            lora_scale=settings.fill_pre_lora_scale,
                        ),
                        reporter=reporter,
                    )
                reporter.emit(
                    "hybrid_fill_pre_done",
                    stage="inference",
                    message="FLUX Fill pre-fill stage completed.",
                    metadata={
                        "width": fill_pre_output.get("width"),
                        "height": fill_pre_output.get("height"),
                    },
                )
                fill_pre_base64 = str(fill_pre_output["image_base64"])
                fill_pre_base64_for_response = fill_pre_base64

            # Stage 2: Canny/LanPaint native using the pre-filled image as source latent.
            reporter.emit(
                "hybrid_canny_start",
                stage="inference",
                message="Starting FLUX-Canny/LanPaint native stage.",
                metadata={"method": FLUX_FILL_CANNY_FILL_METHOD, "stage": 2},
            )
            with _temporary_environ(self.canny_env):
                canny_output = self.canny_service.run(
                    _fill_canny_native_payload(
                        payload,
                        settings,
                        fill_pre_base64,
                        original_image_base64,
                    ),
                    reporter=reporter,
                )
            reporter.emit(
                "hybrid_canny_done",
                stage="inference",
                message="FLUX-Canny/LanPaint native stage completed.",
                metadata={
                    "width": canny_output.get("width"),
                    "height": canny_output.get("height"),
                },
            )

            canny_intermediate_base64 = str(canny_output["image_base64"])

            # Stage 3: Fill refinement on the Canny result.
            reporter.emit(
                "hybrid_fill_post_start",
                stage="inference",
                message="Starting FLUX Fill refinement stage.",
                metadata={"method": FLUX_FILL_CANNY_FILL_METHOD, "stage": 3},
            )
            with _temporary_environ(self.fill_env):
                fill_post_output = self.fill_service.run(
                    _fill_payload(
                        payload,
                        settings,
                        canny_intermediate_base64,
                        prompt=settings.fill_post_prompt,
                        negative_prompt=settings.fill_post_negative_prompt,
                        partial_noise=settings.fill_post_partial_noise,
                        guidance_scale=settings.fill_post_guidance_scale,
                        num_inference_steps=settings.fill_post_num_inference_steps,
                        lora_scale=settings.fill_post_lora_scale,
                    ),
                    reporter=reporter,
                )
            reporter.emit(
                "hybrid_fill_post_done",
                stage="inference",
                message="FLUX Fill refinement stage completed.",
                metadata={
                    "width": fill_post_output.get("width"),
                    "height": fill_post_output.get("height"),
                    "outside_mask_changed_after_hard_composite": fill_post_output.get(
                        "outside_mask_changed_after_hard_composite"
                    ),
                },
            )

        elapsed = time.perf_counter() - started
        output = dict(fill_post_output)
        if fill_pre_base64_for_response is not None:
            output["fill_pre_image_base64"] = fill_pre_base64_for_response
        output["canny_image_base64"] = canny_intermediate_base64
        _fill_pre_timings = fill_pre_output.get("timings", {}) if not settings.skip_fill_pre else {}
        _fill_pre_gpu = fill_pre_output.get("gpu_memory", {}) if not settings.skip_fill_pre else {}
        _fill_pre_model = fill_pre_output.get("model", {}) if not settings.skip_fill_pre else {}
        output["timings"] = {
            "hybrid_total_seconds": elapsed,
            "fill_pre": _fill_pre_timings,
            "canny": canny_output.get("timings", {}),
            "fill_post": fill_post_output.get("timings", {}),
        }
        output["gpu_memory"] = {
            "fill_pre": _fill_pre_gpu,
            "canny": canny_output.get("gpu_memory", {}),
            "fill_post": fill_post_output.get("gpu_memory", {}),
        }
        output["model"] = {
            "method": FLUX_FILL_CANNY_FILL_METHOD,
            "fill_pre": _fill_pre_model,
            "canny": canny_output.get("model", {}),
            "fill_post": fill_post_output.get("model", {}),
        }
        output["lora"] = {
            "fill_pre": fill_pre_output.get("lora", {}) if not settings.skip_fill_pre else {},
            "canny": canny_output.get("lora", {}),
            "fill_post": fill_post_output.get("lora", {}),
        }
        output["inference_settings"] = {
            "method": FLUX_FILL_CANNY_FILL_METHOD,
            "prompt": settings.prompt,
            "seed": settings.seed,
            "output_format": settings.output_format,
            "fill_pre": {
                "prompt": settings.fill_pre_prompt,
                "negative_prompt": settings.fill_pre_negative_prompt,
                "partial_noise": settings.fill_pre_partial_noise,
                "guidance_scale": settings.fill_pre_guidance_scale,
                "num_inference_steps": settings.fill_pre_num_inference_steps,
                "max_sequence_length": settings.fill_max_sequence_length,
                "lora_scale": settings.fill_pre_lora_scale,
            },
            "fill_post": {
                "prompt": settings.fill_post_prompt,
                "negative_prompt": settings.fill_post_negative_prompt,
                "partial_noise": settings.fill_post_partial_noise,
                "guidance_scale": settings.fill_post_guidance_scale,
                "num_inference_steps": settings.fill_post_num_inference_steps,
                "max_sequence_length": settings.fill_max_sequence_length,
                "lora_scale": settings.fill_post_lora_scale,
            },
            "canny": {
                "prompt": settings.canny_prompt,
                "partial_noise": settings.canny_partial_noise,
                "guidance_scale": settings.canny_guidance_scale,
                "num_inference_steps": settings.canny_num_inference_steps,
                "max_sequence_length": settings.canny_max_sequence_length,
                "lora_scale": settings.canny_lora_scale,
                "low_threshold": settings.canny_low_threshold,
                "high_threshold": settings.canny_high_threshold,
                "blur_radius": settings.canny_blur_radius,
                "backend_revision": canny_output.get("inference_settings", {}).get(
                    "backend_revision"
                ),
                "latent_source_strategy": canny_output.get("inference_settings", {}).get(
                    "latent_source_strategy"
                ),
            },
            "lanpaint": {
                "inner_steps": settings.lanpaint_inner_steps,
                "friction": settings.lanpaint_friction,
                "lambda": settings.lanpaint_lambda,
                "beta": settings.lanpaint_beta,
                "step_size": settings.lanpaint_step_size,
                "final_outer_steps_without_inner": (
                    settings.lanpaint_final_outer_steps_without_inner
                ),
            },
        }
        output["outside_mask_changed_after_hard_composite"] = fill_post_output.get(
            "outside_mask_changed_after_hard_composite"
        )
        output.pop("run_report", None)
        return output


def _fill_canny_native_payload(
    payload: dict[str, Any],
    settings: CannyFillRequestSettings,
    filled_image_base64: str,
    original_image_base64: str,
) -> dict[str, Any]:
    canny_payload = dict(payload)
    canny_payload.update(
        {
            "method": FLUX_CANNY_LANPAINT_NATIVE_METHOD,
            "image_base64": filled_image_base64,
            "control_image_base64": payload.get("control_image_base64") or original_image_base64,
            "prompt": settings.canny_prompt,
            "partial_noise": settings.canny_partial_noise,
            "guidance_scale": settings.canny_guidance_scale,
            "num_inference_steps": settings.canny_num_inference_steps,
            "max_sequence_length": settings.canny_max_sequence_length,
            "lora_scale": settings.canny_lora_scale,
            "output_format": settings.output_format,
            "canny_low_threshold": settings.canny_low_threshold,
            "canny_high_threshold": settings.canny_high_threshold,
            "canny_blur_radius": settings.canny_blur_radius,
            "lanpaint_inner_steps": settings.lanpaint_inner_steps,
            "lanpaint_friction": settings.lanpaint_friction,
            "lanpaint_lambda": settings.lanpaint_lambda,
            "lanpaint_beta": settings.lanpaint_beta,
            "lanpaint_step_size": settings.lanpaint_step_size,
            "lanpaint_final_outer_steps_without_inner": (
                settings.lanpaint_final_outer_steps_without_inner
            ),
            "canny_control_strategy": settings.canny_control_strategy,
        }
    )
    if settings.canny_control_strategy is None:
        canny_payload.pop("canny_control_strategy", None)
    for key in (
        "fill_prompt", "fill_negative_prompt", "fill_partial_noise",
        "fill_guidance_scale", "fill_num_inference_steps",
        "fill_max_sequence_length", "fill_lora_scale",
        "fill_pre_prompt", "fill_pre_negative_prompt", "fill_pre_partial_noise",
        "fill_pre_guidance_scale", "fill_pre_num_inference_steps", "fill_pre_lora_scale",
        "fill_post_prompt", "fill_post_negative_prompt", "fill_post_partial_noise",
        "fill_post_guidance_scale", "fill_post_num_inference_steps", "fill_post_lora_scale",
    ):
        canny_payload.pop(key, None)
    return canny_payload


__all__ = [
    "CANNY_FILL_DEFAULT_CANNY_GUIDANCE_SCALE",
    "CANNY_FILL_DEFAULT_CANNY_PARTIAL_NOISE",
    "CANNY_FILL_DEFAULT_CANNY_STEPS",
    "CANNY_FILL_DEFAULT_FILL_GUIDANCE_SCALE",
    "CANNY_FILL_DEFAULT_FILL_PARTIAL_NOISE",
    "CANNY_FILL_DEFAULT_FILL_STEPS",
    "CannyFillRequestSettings",
    "FluxCannyFillService",
    "FluxFillCannyFillService",
    "FluxFillCannyNativeService",
    "parse_canny_fill_settings",
]
