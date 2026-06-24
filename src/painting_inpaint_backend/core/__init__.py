"""Provider-neutral inference core."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .inference import InferenceService, WorkerInputError

__all__ = ["InferenceService", "WorkerInputError"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from .inference import InferenceService, WorkerInputError

        return {
            "InferenceService": InferenceService,
            "WorkerInputError": WorkerInputError,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
