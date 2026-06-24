"""Shared backend for remote painting restoration inference."""

from .core.inference import InferenceService, WorkerInputError

__all__ = ["InferenceService", "WorkerInputError"]
