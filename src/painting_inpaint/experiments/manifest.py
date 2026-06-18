"""Run-folder and manifest helpers."""

from __future__ import annotations

import json
import platform
import re
import subprocess
import sys
from collections.abc import Iterable
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from painting_inpaint.paths import PROJECT_ROOT, runs_root


def slugify(value: str) -> str:
    """Return a filesystem-safe slug."""

    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value.strip()).strip("_")
    return slug or "run"


def timestamp_utc() -> str:
    """Return a compact UTC timestamp for run IDs."""

    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def create_run_dir(method_name: str, root: str | Path | None = None) -> Path:
    """Create and return ``runs/<timestamp>_<method_name>``."""

    base = Path(root).expanduser() if root is not None else runs_root()
    run_dir = base / f"{timestamp_utc()}_{slugify(method_name)}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _optional_version(module_name: str) -> str | None:
    try:
        module = __import__(module_name)
    except Exception:
        return None
    return getattr(module, "__version__", None)


def git_commit(cwd: str | Path = PROJECT_ROOT) -> str | None:
    """Return the current git commit hash when available."""

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(cwd),
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def package_versions(package_names: Iterable[str]) -> dict[str, str | None]:
    """Return installed distribution versions, using ``None`` when absent."""

    versions: dict[str, str | None] = {}
    for name in package_names:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return versions


def collect_environment() -> dict[str, Any]:
    """Collect runtime information useful for reproducibility."""

    env: dict[str, Any] = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": _optional_version("torch"),
        "diffusers_version": _optional_version("diffusers"),
        "cuda_available": None,
        "gpu_name": None,
        "git_commit": git_commit(),
    }

    try:
        import torch

        env["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            env["gpu_name"] = torch.cuda.get_device_name(0)
    except Exception:
        env["cuda_available"] = False

    return env


def write_json(path: str | Path, data: dict[str, Any]) -> Path:
    """Write stable pretty JSON."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def write_manifest(run_dir: str | Path, manifest: dict[str, Any]) -> Path:
    """Write ``manifest.json`` into a run directory."""

    return write_json(Path(run_dir) / "manifest.json", manifest)
