"""Model, cache, and LoRA loading for the RunPod FLUX Fill worker."""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .partial_noise import FluxFillPartialNoisePipeline
from .progress import ProgressReporter

LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "black-forest-labs/FLUX.1-Fill-dev"
DEFAULT_LORA_PATH = "/app/runpod_worker/loras/pytorch_lora_weights.safetensors"
DEFAULT_LORA_CACHE_DIR = "/tmp/painting-inpaint-lora-cache"
DEFAULT_ADAPTER_NAME = "durer"


@dataclass(frozen=True)
class ModelPathResolution:
    """Resolved model source for ``from_pretrained``."""

    load_target: str
    source: str
    local_files_only: bool
    cache_root: str | None = None
    snapshot_path: str | None = None


@dataclass(frozen=True)
class LoadedPipeline:
    """Loaded pipeline plus JSON-safe loading metadata."""

    pipe: Any
    torch: Any
    torch_dtype: Any
    model: dict[str, Any]
    lora: dict[str, Any]
    supports_negative_prompt: bool
    timings: dict[str, float]


@dataclass(frozen=True)
class LoraConfig:
    """Resolved local or Hugging Face LoRA configuration."""

    required: bool
    source: str
    repo_id: str | None
    filename: str | None
    revision: str
    local_path: Path | None
    cache_dir: Path
    adapter_name: str
    error: str | None = None


def env_flag(name: str, *, default: bool = False) -> bool:
    """Return a boolean environment flag."""

    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_value(name: str) -> str | None:
    value = os.environ.get(name)
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


def _repo_cache_name(model_id: str) -> str:
    return f"models--{model_id.replace('/', '--')}"


def _candidate_cache_roots(extra_roots: list[str | Path] | None = None) -> list[Path]:
    roots: list[Path] = []
    if extra_roots:
        roots.extend(Path(root) for root in extra_roots)
    for env_name in ("HF_HUB_CACHE", "HF_HOME"):
        value = os.environ.get(env_name)
        if value:
            path = Path(value)
            roots.append(path if path.name == "hub" else path / "hub")
    roots.extend(
        [
            Path("/runpod-volume/huggingface-cache/hub"),
            Path("/runpod-volume/huggingface-cache"),
            Path.home() / ".cache" / "huggingface" / "hub",
        ]
    )

    deduped: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root)
        if key not in seen:
            seen.add(key)
            deduped.append(root)
    return deduped


def _snapshot_from_ref(repo_dir: Path, ref_name: str = "main") -> Path | None:
    ref_path = repo_dir / "refs" / ref_name
    if not ref_path.exists():
        return None
    snapshot_id = ref_path.read_text(encoding="utf-8").strip()
    if not snapshot_id:
        return None
    candidate = repo_dir / "snapshots" / snapshot_id
    return candidate if candidate.exists() and candidate.is_dir() else None


def resolve_hf_snapshot_path(
    model_id: str = DEFAULT_MODEL_ID,
    *,
    cache_roots: list[str | Path] | None = None,
) -> Path | None:
    """Resolve a Hugging Face cache snapshot path without hardcoding a commit hash."""

    repo_name = _repo_cache_name(model_id)
    for root in _candidate_cache_roots(cache_roots):
        candidates = [root] if root.name == repo_name else [root / repo_name]
        if root.name != "hub":
            candidates.append(root / "hub" / repo_name)
        for repo_dir in candidates:
            if not repo_dir.exists() or not repo_dir.is_dir():
                continue
            ref_snapshot = _snapshot_from_ref(repo_dir)
            if ref_snapshot is not None:
                return ref_snapshot
            snapshots = repo_dir / "snapshots"
            if snapshots.exists():
                available = [path for path in snapshots.iterdir() if path.is_dir()]
                if available:
                    return max(available, key=lambda path: path.stat().st_mtime)
    return None


def resolve_model_load_target(model_id: str = DEFAULT_MODEL_ID) -> ModelPathResolution:
    """Resolve the preferred model path, allowing explicit opt-in HF downloads."""

    started = time.perf_counter()
    snapshot_path = resolve_hf_snapshot_path(model_id)
    elapsed = time.perf_counter() - started
    LOGGER.info("Model path discovery finished in %.3fs", elapsed)
    if snapshot_path is not None:
        return ModelPathResolution(
            load_target=str(snapshot_path),
            source="huggingface_cache_snapshot",
            local_files_only=True,
            cache_root=str(snapshot_path.parents[2]) if len(snapshot_path.parents) >= 3 else None,
            snapshot_path=str(snapshot_path),
        )
    if env_flag("ALLOW_HF_DOWNLOAD", default=False):
        return ModelPathResolution(
            load_target=model_id,
            source="huggingface_download_fallback",
            local_files_only=False,
        )
    raise FileNotFoundError(
        "Missing cached Hugging Face model for "
        f"{model_id!r}. Configure the RunPod endpoint Hugging Face model cache for this "
        "model, or set ALLOW_HF_DOWNLOAD=1 to permit a gated Hugging Face download."
    )


def resolve_torch_dtype(torch: Any, dtype_config: str | None = None) -> Any:
    """Resolve the worker torch dtype."""

    value = (dtype_config or os.environ.get("TORCH_DTYPE") or "auto").lower()
    if value == "auto":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    if value in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if value in {"float16", "fp16"}:
        return torch.float16
    if value in {"float32", "fp32"}:
        return torch.float32
    raise ValueError("TORCH_DTYPE must be one of: auto, bfloat16, float16, float32.")


def resolve_lora_path(
    *,
    lora_path: str | Path | None = None,
    required: bool | None = None,
) -> Path | None:
    """Resolve the LoRA weights path, respecting production required mode."""

    required = env_flag("LORA_REQUIRED", default=False) if required is None else bool(required)
    configured = Path(lora_path or os.environ.get("LORA_PATH", DEFAULT_LORA_PATH))
    if configured.exists():
        return configured
    if required:
        raise FileNotFoundError(
            "LoRA is required because LORA_REQUIRED=1, but no weights file was found. "
            f"Searched: {configured}. Set LORA_PATH or bake the LoRA into "
            "runpod_worker/loras/ before building the image."
        )
    return None


def resolve_lora_config(
    *,
    lora_path: str | Path | None = None,
    required: bool | None = None,
) -> LoraConfig:
    """Resolve remote LoRA settings first, then preserve the local-file fallback."""

    required_value = (
        env_flag("LORA_REQUIRED", default=False) if required is None else bool(required)
    )
    repo_id = _env_value("LORA_REPO_ID")
    filename = _env_value("LORA_FILENAME")
    revision = _env_value("LORA_REVISION") or "main"
    adapter_name = _env_value("LORA_ADAPTER_NAME") or DEFAULT_ADAPTER_NAME
    cache_dir = Path(_env_value("LORA_CACHE_DIR") or DEFAULT_LORA_CACHE_DIR)

    if repo_id is not None or filename is not None:
        error = None
        source = "huggingface_repo"
        if repo_id is None or filename is None:
            source = "invalid_remote_config"
            error = (
                "LORA_REPO_ID and LORA_FILENAME must both be set to load a Hugging "
                "Face LoRA."
            )
        return LoraConfig(
            required=required_value,
            source=source,
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            local_path=None,
            cache_dir=cache_dir,
            adapter_name=adapter_name,
            error=error,
        )

    configured_path = Path(lora_path or _env_value("LORA_PATH") or DEFAULT_LORA_PATH)
    return LoraConfig(
        required=required_value,
        source="local_path" if configured_path.exists() else "unavailable",
        repo_id=None,
        filename=None,
        revision=revision,
        local_path=configured_path,
        cache_dir=cache_dir,
        adapter_name=adapter_name,
    )


def download_lora_from_huggingface(
    config: LoraConfig,
    *,
    download_fn: Callable[..., str] | None = None,
) -> Path:
    """Download and cache a configured Hugging Face LoRA file."""

    if config.source != "huggingface_repo" or not config.repo_id or not config.filename:
        raise ValueError("A complete Hugging Face LoRA configuration is required.")

    if download_fn is None:
        from huggingface_hub import hf_hub_download

        download_fn = hf_hub_download

    config.cache_dir.mkdir(parents=True, exist_ok=True)
    token = _env_value("HF_TOKEN") or _env_value("HUGGINGFACE_HUB_TOKEN")
    kwargs: dict[str, Any] = {
        "repo_id": config.repo_id,
        "filename": config.filename,
        "revision": config.revision,
        "cache_dir": str(config.cache_dir),
    }
    if token:
        kwargs["token"] = token

    local_path = Path(download_fn(**kwargs))
    if not local_path.exists() or not local_path.is_file():
        raise FileNotFoundError(
            "hf_hub_download did not return an existing LoRA file: "
            f"{local_path}"
        )
    return local_path


def _resolve_lora_file(path: Path) -> tuple[Path, str]:
    if path.is_file():
        return path.parent, path.name
    weight_name = os.environ.get("LORA_WEIGHT_NAME", "pytorch_lora_weights.safetensors")
    candidate = path / weight_name
    if candidate.exists():
        return candidate.parent, candidate.name
    candidates = sorted(path.rglob("*.safetensors"))
    if candidates:
        chosen = candidates[-1]
        return chosen.parent, chosen.name
    raise FileNotFoundError(f"No .safetensors LoRA weights found under {path}.")


def _lora_debug(config: LoraConfig) -> dict[str, Any]:
    local_path = str(config.local_path) if config.local_path is not None else None
    return {
        "loaded": False,
        "required": config.required,
        "source": config.source,
        "repo_id": config.repo_id,
        "filename": config.filename,
        "revision": config.revision,
        "local_path": local_path,
        "path": local_path,
        "cache_dir": str(config.cache_dir),
        "adapter_name": config.adapter_name,
        "strength_mode": "not_loaded_optional",
        "effective_scale": None,
        "elapsed_seconds": 0.0,
        "error": config.error,
    }


def _lora_event_metadata(config: LoraConfig, **extra: Any) -> dict[str, Any]:
    metadata = {
        "required": config.required,
        "source": config.source,
        "repo_id": config.repo_id,
        "filename": config.filename,
        "revision": config.revision,
        "local_path": str(config.local_path) if config.local_path is not None else None,
        "cache_dir": str(config.cache_dir),
        "adapter_name": config.adapter_name,
    }
    metadata.update(extra)
    return metadata


def load_lora_if_available(
    pipe: Any,
    *,
    lora_path: str | Path | None = None,
    config: LoraConfig | None = None,
    download_fn: Callable[..., str] | None = None,
    reporter: ProgressReporter | None = None,
) -> dict[str, Any]:
    """Download and load one LoRA, respecting optional and required modes."""

    started = time.perf_counter()
    if reporter is not None:
        reporter.emit(
            "lora_resolve_start",
            stage="lora",
            message="Resolving LoRA configuration.",
        )
    resolved_config = config or resolve_lora_config(lora_path=lora_path)
    debug = _lora_debug(resolved_config)

    if resolved_config.error is not None:
        debug["elapsed_seconds"] = time.perf_counter() - started
        if reporter is not None:
            reporter.emit(
                "lora_load_done",
                stage="lora",
                message="LoRA configuration is invalid.",
                metadata=_lora_event_metadata(
                    resolved_config,
                    loaded=False,
                    error=resolved_config.error,
                ),
            )
        if resolved_config.required:
            raise RuntimeError(f"Required LoRA configuration is invalid: {resolved_config.error}")
        return debug

    if resolved_config.source == "unavailable":
        debug["elapsed_seconds"] = time.perf_counter() - started
        if reporter is not None:
            reporter.emit(
                "lora_load_done",
                stage="lora",
                message="No LoRA weights configured; continuing without LoRA.",
                metadata=_lora_event_metadata(resolved_config, loaded=False),
            )
        if resolved_config.required:
            raise FileNotFoundError(
                "LoRA is required because LORA_REQUIRED=1, but neither a complete "
                "Hugging Face LoRA configuration nor an existing local LoRA file was found. "
                f"Searched local path: {resolved_config.local_path}"
            )
        return debug

    try:
        if resolved_config.source == "huggingface_repo":
            if reporter is not None:
                reporter.emit(
                    "lora_download_start",
                    stage="lora",
                    message="Downloading LoRA weights from Hugging Face.",
                    metadata=_lora_event_metadata(resolved_config),
                )
            resolved_path = download_lora_from_huggingface(
                resolved_config,
                download_fn=download_fn,
            )
            if reporter is not None:
                reporter.emit(
                    "lora_download_done",
                    stage="lora",
                    message="LoRA weights downloaded.",
                    metadata=_lora_event_metadata(
                        resolved_config,
                        resolved_path=str(resolved_path),
                    ),
                )
        else:
            resolved_path = resolved_config.local_path
            if resolved_path is None or not resolved_path.exists():
                raise FileNotFoundError(f"Missing configured LoRA file: {resolved_path}")

        if not hasattr(pipe, "load_lora_weights"):
            raise RuntimeError(f"{type(pipe).__name__} does not expose load_lora_weights.")

        parent, weight_name = _resolve_lora_file(resolved_path)
        if reporter is not None:
            reporter.emit(
                "lora_load_start",
                stage="lora",
                message="Loading LoRA weights into the pipeline.",
                metadata=_lora_event_metadata(
                    resolved_config,
                    weight_name=weight_name,
                    parent=str(parent),
                ),
            )
        load_result = pipe.load_lora_weights(
            str(parent),
            weight_name=weight_name,
            adapter_name=resolved_config.adapter_name,
        )
        scale_result = set_lora_scale(
            pipe,
            adapter_name=resolved_config.adapter_name,
            lora_scale=1.0,
        )
        elapsed = time.perf_counter() - started
        LOGGER.info("LoRA loading finished in %.3fs", elapsed)
        local_path = str(parent / weight_name)
        debug.update(
            {
                "loaded": True,
                "local_path": local_path,
                "path": local_path,
                "adapter_name": resolved_config.adapter_name,
                "load_method": "load_lora_weights",
                "load_result_repr": repr(load_result),
                "strength_mode": scale_result["mode"],
                "effective_scale": scale_result["effective_scale"],
                "elapsed_seconds": elapsed,
                "error": None,
            }
        )
        if reporter is not None:
            reporter.emit(
                "lora_load_done",
                stage="lora",
                message="LoRA weights loaded.",
                metadata=_lora_event_metadata(
                    resolved_config,
                    loaded=True,
                    local_path=local_path,
                    weight_name=weight_name,
                    effective_scale=scale_result["effective_scale"],
                    strength_mode=scale_result["mode"],
                    elapsed_seconds=elapsed,
                ),
            )
        return debug
    except Exception as exc:
        elapsed = time.perf_counter() - started
        error = f"{type(exc).__name__}: {exc}"
        if resolved_config.required:
            raise RuntimeError(f"Required LoRA download/load failed: {error}") from exc
        LOGGER.warning("Optional LoRA could not be loaded: %s", error)
        debug.update({"elapsed_seconds": elapsed, "error": error})
        if reporter is not None:
            reporter.emit(
                "lora_load_done",
                stage="lora",
                message="Optional LoRA could not be loaded; continuing without LoRA.",
                metadata=_lora_event_metadata(
                    resolved_config,
                    loaded=False,
                    error=error,
                    elapsed_seconds=elapsed,
                ),
            )
        return debug


def set_lora_scale(
    pipe: Any,
    *,
    adapter_name: str | None,
    lora_scale: float,
) -> dict[str, Any]:
    """Apply a request-specific LoRA scale and report the effective value."""

    scale = float(lora_scale)
    if scale < 0:
        raise ValueError("lora_scale must be non-negative.")
    if not adapter_name:
        return {"mode": "not_loaded", "effective_scale": None}

    if hasattr(pipe, "set_adapters"):
        try:
            pipe.set_adapters([adapter_name], adapter_weights=[scale])
            return {"mode": "set_adapters", "effective_scale": scale}
        except TypeError as exc:
            if scale == 0 and hasattr(pipe, "disable_lora"):
                pipe.disable_lora()
                return {"mode": "disable_lora", "effective_scale": 0.0}
            if scale == 1.0:
                pipe.set_adapters([adapter_name])
                return {"mode": "set_adapters_no_weights", "effective_scale": 1.0}
            raise RuntimeError(
                f"{type(pipe).__name__} cannot apply request-specific LoRA scale {scale}."
            ) from exc

    if scale == 0 and hasattr(pipe, "disable_lora"):
        pipe.disable_lora()
        return {"mode": "disable_lora", "effective_scale": 0.0}
    if scale == 1.0:
        return {"mode": "loaded_default_weight", "effective_scale": 1.0}
    raise RuntimeError(
        f"{type(pipe).__name__} does not expose an API for LoRA scale {scale}."
    )


def _pipeline_accepts(pipe: Any, parameter_name: str) -> bool:
    import inspect

    try:
        signature = inspect.signature(pipe.__call__)
    except Exception:
        return False
    if parameter_name in signature.parameters:
        return True
    return any(
        param.kind is inspect.Parameter.VAR_KEYWORD
        for param in signature.parameters.values()
    )


def load_pipeline(*, reporter: ProgressReporter | None = None) -> LoadedPipeline:
    """Load the FLUX Fill pipeline, optional LoRA, and performance settings."""

    if not hasattr(FluxFillPartialNoisePipeline, "from_pretrained"):
        raise ImportError(
            "FluxFillPartialNoisePipeline requires diffusers with FluxFillPipeline support."
        )

    import torch

    model_id = os.environ.get("MODEL_ID", DEFAULT_MODEL_ID)
    if reporter is not None:
        reporter.emit(
            "model_load_start",
            stage="model",
            message="Loading FLUX Fill pipeline.",
            metadata={"model_id": model_id},
        )
    timings: dict[str, float] = {}
    resolve_started = time.perf_counter()
    resolution = resolve_model_load_target(model_id)
    timings["model_path_discovery_seconds"] = time.perf_counter() - resolve_started

    torch_dtype = resolve_torch_dtype(torch)
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    kwargs: dict[str, Any] = {
        "torch_dtype": torch_dtype,
        "local_files_only": resolution.local_files_only,
    }
    if token:
        kwargs["token"] = token

    load_started = time.perf_counter()
    pipe = FluxFillPartialNoisePipeline.from_pretrained(resolution.load_target, **kwargs)
    timings["from_pretrained_seconds"] = time.perf_counter() - load_started

    device_started = time.perf_counter()
    if env_flag("ENABLE_MODEL_CPU_OFFLOAD", default=True) and hasattr(
        pipe, "enable_model_cpu_offload"
    ):
        pipe.enable_model_cpu_offload()
        device_mode = "model_cpu_offload"
    elif torch.cuda.is_available():
        pipe.to("cuda")
        device_mode = "cuda"
    else:
        device_mode = "cpu"

    if getattr(pipe, "vae", None) is not None:
        if env_flag("ENABLE_VAE_TILING", default=True) and hasattr(pipe.vae, "enable_tiling"):
            pipe.vae.enable_tiling()
        if env_flag("ENABLE_VAE_SLICING", default=True) and hasattr(pipe.vae, "enable_slicing"):
            pipe.vae.enable_slicing()
    timings["device_setup_seconds"] = time.perf_counter() - device_started
    model = {
        "model_id": model_id,
        "load_target": resolution.load_target,
        "source": resolution.source,
        "local_files_only": resolution.local_files_only,
        "cache_root": resolution.cache_root,
        "snapshot_path": resolution.snapshot_path,
        "torch_dtype": str(torch_dtype),
        "device_mode": device_mode,
        "pipeline_class": type(pipe).__name__,
    }
    if reporter is not None:
        reporter.emit(
            "model_load_done",
            stage="model",
            message="FLUX Fill pipeline loaded.",
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
        supports_negative_prompt=_pipeline_accepts(pipe, "negative_prompt"),
        timings=timings,
    )
