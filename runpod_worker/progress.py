"""Structured progress events for the RunPod worker."""

from __future__ import annotations

import json
import os
import time
import uuid
from collections.abc import Callable, Mapping
from typing import Any

SENSITIVE_KEY_FRAGMENTS = (
    "base64",
    "token",
    "api_key",
    "apikey",
    "authorization",
    "password",
    "secret",
)
SENSITIVE_EXACT_KEYS = {"payload", "input"}


def _env_flag(name: str, *, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _is_sensitive_key(key: Any) -> bool:
    normalized = str(key).lower()
    return normalized in SENSITIVE_EXACT_KEYS or any(
        fragment in normalized for fragment in SENSITIVE_KEY_FRAGMENTS
    )


def sanitize_for_progress(value: Any) -> Any:
    """Return a JSON-safe copy with secrets and embedded image data removed."""

    if isinstance(value, Mapping):
        return {
            str(key): sanitize_for_progress(item)
            for key, item in value.items()
            if not _is_sensitive_key(key)
        }
    if isinstance(value, tuple | list):
        return [sanitize_for_progress(item) for item in value]
    if value is None or isinstance(value, str | int | float | bool):
        return value
    return str(value)


def _normalize_progress(progress: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if progress is None:
        return None
    normalized = {
        str(key): value
        for key, value in progress.items()
        if key in {"current", "total", "fraction"}
    }
    current = normalized.get("current")
    total = normalized.get("total")
    if "fraction" not in normalized and current is not None and total:
        try:
            normalized["fraction"] = float(current) / float(total)
        except (TypeError, ValueError, ZeroDivisionError):
            pass
    if "fraction" in normalized:
        try:
            normalized["fraction"] = max(0.0, min(1.0, float(normalized["fraction"])))
        except (TypeError, ValueError):
            normalized.pop("fraction", None)
    return normalized or None


class ProgressReporter:
    """Emit sanitized worker progress events to logs, history, and stream sinks."""

    def __init__(
        self,
        *,
        job_id: str | None = None,
        run_id: str | None = None,
        enabled: bool = True,
        keep_history: bool = False,
        emit_stdout_json: bool | None = None,
        on_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.job_id = job_id
        self.run_id = run_id or job_id or uuid.uuid4().hex
        self.enabled = enabled
        self.keep_history = keep_history
        self.emit_stdout_json = (
            _env_flag("PROGRESS_STDOUT_JSON", default=False)
            if emit_stdout_json is None
            else emit_stdout_json
        )
        self.on_event = on_event
        self.started_at = time.perf_counter()
        self.history: list[dict[str, Any]] = []

    def _base_event(
        self,
        *,
        type: str,
        event: str,
        stage: str,
        message: str,
        progress: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        progress_event: dict[str, Any] = {
            "type": type,
            "event": event,
            "run_id": self.run_id,
            "stage": stage,
            "message": message,
            "elapsed_seconds": round(time.perf_counter() - self.started_at, 3),
            "metadata": sanitize_for_progress(metadata or {}),
        }
        if self.job_id:
            progress_event["job_id"] = self.job_id
        normalized_progress = _normalize_progress(progress)
        if normalized_progress is not None:
            progress_event["progress"] = normalized_progress
        return progress_event

    def _record(self, event: dict[str, Any]) -> None:
        if not self.enabled:
            return
        sanitized_event = sanitize_for_progress(event)
        if self.keep_history:
            self.history.append(sanitized_event)
        if self.emit_stdout_json:
            print(json.dumps(sanitized_event, sort_keys=True), flush=True)
        if self.on_event is not None:
            self.on_event(event)

    def emit(
        self,
        event: str,
        *,
        stage: str,
        message: str,
        type: str = "progress",
        progress: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        progress_event = self._base_event(
            type=type,
            event=event,
            stage=stage,
            message=message,
            progress=progress,
            metadata=metadata,
        )
        self._record(progress_event)
        return progress_event

    def final(
        self,
        *,
        output: dict[str, Any],
        event: str = "job_done",
        stage: str = "output",
        message: str = "Job completed.",
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        progress_event = self._base_event(
            type="final",
            event=event,
            stage=stage,
            message=message,
            metadata=metadata,
        )
        progress_event["output"] = output
        self._record(progress_event)
        return progress_event

    def error(
        self,
        *,
        message: str,
        event: str = "job_failed",
        stage: str = "error",
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        progress_event = self._base_event(
            type="error",
            event=event,
            stage=stage,
            message=message,
            metadata=metadata,
        )
        self._record(progress_event)
        return progress_event
