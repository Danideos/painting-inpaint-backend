"""RunPod request and progress-streaming adapter."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from ..core.inference import InferenceService, WorkerInputError, _payload_flag
from ..core.progress import ProgressReporter
from ..core.streaming import stream_inference_events

_SERVICE = InferenceService()


def run_job_input(payload: dict[str, Any]) -> dict[str, Any]:
    """Run one validated RunPod input payload."""

    if not isinstance(payload, dict):
        raise WorkerInputError("RunPod job input must be a JSON object.")
    include_progress_history = _payload_flag(
        payload,
        "include_progress_history",
        default=False,
    )
    reporter = ProgressReporter(
        enabled=include_progress_history,
        keep_history=include_progress_history,
    )
    if include_progress_history:
        reporter.emit(
            "job_received",
            stage="input",
            message="RunPod job received.",
            metadata={"stream_progress": False},
        )
    return _SERVICE.run(payload, reporter=reporter)


def run_job_input_streaming(
    payload: dict[str, Any],
    *,
    job_id: str | None = None,
    service: InferenceService | None = None,
) -> Iterator[dict[str, Any]]:
    """Run one RunPod input payload and yield progress events as they happen."""

    yield from stream_inference_events(
        payload,
        service=service or _SERVICE,
        job_id=job_id,
        provider="runpod",
        received_message="RunPod job received.",
        completion_message="RunPod job completed.",
        received_metadata={"stream_progress": True},
    )
