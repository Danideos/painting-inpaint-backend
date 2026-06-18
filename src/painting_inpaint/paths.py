"""Project path helpers with environment-variable overrides."""

from __future__ import annotations

import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parents[1]


def data_root() -> Path:
    """Return the configured data root, defaulting to ``<repo>/data``."""

    return Path(os.environ.get("PAINTING_INPAINT_DATA_ROOT", PROJECT_ROOT / "data")).expanduser()


def runs_root() -> Path:
    """Return the configured runs root, defaulting to ``<repo>/runs``."""

    return Path(os.environ.get("PAINTING_INPAINT_RUNS_ROOT", PROJECT_ROOT / "runs")).expanduser()


def resolve_project_path(path: str | os.PathLike[str] | None) -> Path | None:
    """Resolve a path relative to the repository root unless it is already absolute."""

    if path is None:
        return None
    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def resolve_data_path(path: str | os.PathLike[str] | None) -> Path | None:
    """Resolve an asset path relative to ``PAINTING_INPAINT_DATA_ROOT``.

    Use this for case inputs such as images, masks, references, aligned references,
    and prepared controls. Absolute paths are returned unchanged.
    """

    if path is None:
        return None
    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    return data_root() / p
