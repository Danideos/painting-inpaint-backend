"""Model, cache, and LoRA loading for the RunPod FLUX Fill worker."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .partial_noise import FluxFillPartialNoisePipeline

LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "black-forest-labs/FLUX.1-Fill-dev"
DEFAULT_LORA_PATH = "/app/runpod_worker/loras/pytorch_lora_weights.safetensors"
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


def env_flag(name: str, *, default: bool = False) -> bool:
    """Return a boolean environment flag."""

    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


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


def load_lora_if_available(pipe: Any, *, lora_path: str | Path | None = None) -> dict[str, Any]:
    """Load the configured LoRA when present, or return skipped metadata."""

    started = time.perf_counter()
    resolved = resolve_lora_path(lora_path=lora_path)
    if resolved is None:
        return {
            "loaded": False,
            "required": env_flag("LORA_REQUIRED", default=False),
            "path": str(lora_path or os.environ.get("LORA_PATH", DEFAULT_LORA_PATH)),
            "adapter_name": None,
            "strength_mode": "not_loaded_optional",
            "elapsed_seconds": time.perf_counter() - started,
        }
    if not hasattr(pipe, "load_lora_weights"):
        raise RuntimeError(f"{type(pipe).__name__} does not expose load_lora_weights.")

    adapter_name = os.environ.get("LORA_ADAPTER_NAME", DEFAULT_ADAPTER_NAME)
    parent, weight_name = _resolve_lora_file(resolved)
    load_result = pipe.load_lora_weights(
        str(parent),
        weight_name=weight_name,
        adapter_name=adapter_name,
    )
    strength_mode = set_lora_scale(pipe, adapter_name=adapter_name, lora_scale=1.0)
    elapsed = time.perf_counter() - started
    LOGGER.info("LoRA loading finished in %.3fs", elapsed)
    return {
        "loaded": True,
        "required": env_flag("LORA_REQUIRED", default=False),
        "path": str(parent / weight_name),
        "adapter_name": adapter_name,
        "load_method": "load_lora_weights",
        "load_result_repr": repr(load_result),
        "strength_mode": strength_mode,
        "elapsed_seconds": elapsed,
    }


def set_lora_scale(pipe: Any, *, adapter_name: str | None, lora_scale: float) -> str:
    """Apply per-request LoRA scale when Diffusers exposes adapter controls."""

    if not adapter_name:
        return "not_loaded"
    if hasattr(pipe, "set_adapters"):
        try:
            pipe.set_adapters([adapter_name], adapter_weights=[float(lora_scale)])
            return "set_adapters"
        except TypeError:
            pipe.set_adapters([adapter_name])
            return "set_adapters_no_weights"
    return "loaded_default_weight"


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


def load_pipeline() -> LoadedPipeline:
    """Load the FLUX Fill pipeline, optional LoRA, and performance settings."""

    if not hasattr(FluxFillPartialNoisePipeline, "from_pretrained"):
        raise ImportError(
            "FluxFillPartialNoisePipeline requires diffusers with FluxFillPipeline support."
        )

    import torch

    model_id = os.environ.get("MODEL_ID", DEFAULT_MODEL_ID)
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

    lora = load_lora_if_available(pipe)
    timings["lora_loading_seconds"] = float(lora.get("elapsed_seconds", 0.0))
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
    return LoadedPipeline(
        pipe=pipe,
        torch=torch,
        torch_dtype=torch_dtype,
        model=model,
        lora=lora,
        supports_negative_prompt=_pipeline_accepts(pipe, "negative_prompt"),
        timings=timings,
    )
