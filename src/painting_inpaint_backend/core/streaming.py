"""Provider-neutral progress streaming for inference requests."""

from __future__ import annotations

import queue
import threading
import traceback
from collections.abc import Iterator, Mapping
from typing import Any

from .inference import InferenceService, WorkerInputError
from .progress import ProgressReporter

_STREAM_DONE = object()


def stream_inference_events(
    payload: dict[str, Any],
    *,
    service: InferenceService,
    run_id: str | None = None,
    job_id: str | None = None,
    provider: str,
    received_message: str,
    completion_message: str,
    received_metadata: Mapping[str, Any] | None = None,
) -> Iterator[dict[str, Any]]:
    """Run inference in a worker thread and yield structured events live."""

    events: queue.Queue[dict[str, Any] | object] = queue.Queue()
    reporter = ProgressReporter(
        run_id=run_id,
        job_id=job_id,
        enabled=True,
        keep_history=True,
        on_event=events.put,
    )

    def _run() -> None:
        try:
            if not isinstance(payload, dict):
                raise WorkerInputError("Restoration input must be a JSON object.")
            reporter.emit(
                "job_received",
                stage="input",
                message=received_message,
                metadata={"provider": provider, **dict(received_metadata or {})},
            )
            output = service.run(payload, reporter=reporter)
            reporter.final(
                output=output,
                message=completion_message,
                metadata={
                    "provider": provider,
                    "output_format": output.get("output_format"),
                    "width": output.get("width"),
                    "height": output.get("height"),
                },
            )
        except Exception as exc:
            reporter.error(
                message=str(exc),
                metadata={
                    "provider": provider,
                    "error_type": type(exc).__name__,
                    "traceback": traceback.format_exc(limit=5),
                },
            )
        finally:
            events.put(_STREAM_DONE)

    thread = threading.Thread(
        target=_run,
        name=f"{provider}-progress-worker",
        daemon=True,
    )
    thread.start()
    try:
        while True:
            event = events.get()
            if event is _STREAM_DONE:
                break
            yield event
    finally:
        thread.join()
