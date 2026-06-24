"""RunPod request and progress-streaming adapter."""

from __future__ import annotations

import queue
import threading
import traceback
from collections.abc import Iterator
from typing import Any

from ..core.inference import InferenceService, WorkerInputError, _payload_flag
from ..core.progress import ProgressReporter

_STREAM_DONE = object()
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

    events: queue.Queue[dict[str, Any] | object] = queue.Queue()
    reporter = ProgressReporter(
        job_id=job_id,
        enabled=True,
        keep_history=True,
        on_event=events.put,
    )
    inference_service = service or _SERVICE

    def _run() -> None:
        try:
            if not isinstance(payload, dict):
                raise WorkerInputError("RunPod job input must be a JSON object.")
            reporter.emit(
                "job_received",
                stage="input",
                message="RunPod job received.",
                metadata={"stream_progress": True},
            )
            output = inference_service.run(payload, reporter=reporter)
            reporter.final(
                output=output,
                message="RunPod job completed.",
                metadata={
                    "output_format": output.get("output_format"),
                    "width": output.get("width"),
                    "height": output.get("height"),
                },
            )
        except Exception as exc:
            reporter.error(
                message=str(exc),
                metadata={
                    "error_type": type(exc).__name__,
                    "traceback": traceback.format_exc(limit=5),
                },
            )
        finally:
            events.put(_STREAM_DONE)

    thread = threading.Thread(target=_run, name="runpod-progress-worker", daemon=True)
    thread.start()
    try:
        while True:
            event = events.get()
            if event is _STREAM_DONE:
                break
            yield event
    finally:
        thread.join()
