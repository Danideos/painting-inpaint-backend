"""Authenticated SSE gateway helpers for Modal restoration requests."""

from __future__ import annotations

import json
import queue
import secrets
import threading
import traceback
import uuid
from collections.abc import Callable, Iterable, Iterator
from typing import Any

from ..core.progress import ProgressReporter
from .boundary import json_response

SSE_MEDIA_TYPE = "text/event-stream"
_STREAM_DONE = object()


def bearer_token_matches(authorization: str | None, expected_key: str) -> bool:
    """Validate a bearer token without exposing it in errors or logs."""

    if not expected_key or not authorization:
        return False
    scheme, separator, supplied_key = authorization.partition(" ")
    if not separator or scheme.lower() != "bearer" or not supplied_key:
        return False
    return secrets.compare_digest(supplied_key, expected_key)


def modal_queued_event(run_id: str) -> dict[str, Any]:
    """Build the first event emitted before Modal GPU scheduling."""

    reporter = ProgressReporter(run_id=run_id, enabled=True)
    return reporter.emit(
        "modal_queued",
        stage="modal",
        message="Waiting for a Modal GPU container.",
        metadata={"provider": "modal"},
    )


def _gateway_error_event(run_id: str, exc: Exception) -> dict[str, Any]:
    reporter = ProgressReporter(run_id=run_id, enabled=True)
    return reporter.error(
        message=str(exc),
        metadata={
            "provider": "modal",
            "error_type": type(exc).__name__,
            "traceback": traceback.format_exc(limit=5),
        },
    )


def stream_sse_with_heartbeats(
    event_source: Callable[[], Iterable[str | dict[str, Any]]],
    *,
    run_id: str | None = None,
    heartbeat_seconds: float = 15.0,
) -> Iterator[str]:
    """Forward a blocking Modal event source while keeping HTTP connections alive."""

    resolved_run_id = run_id or uuid.uuid4().hex
    events: queue.Queue[str | dict[str, Any] | object] = queue.Queue()

    def _consume() -> None:
        try:
            for event in event_source():
                events.put(event)
        except Exception as exc:
            events.put(_gateway_error_event(resolved_run_id, exc))
        finally:
            events.put(_STREAM_DONE)

    yield f"data: {json_response(modal_queued_event(resolved_run_id))}\n\n"
    thread = threading.Thread(
        target=_consume,
        name="modal-http-stream-forwarder",
        daemon=True,
    )
    thread.start()
    try:
        while True:
            try:
                event = events.get(timeout=heartbeat_seconds)
            except queue.Empty:
                yield ": heartbeat\n\n"
                continue
            if event is _STREAM_DONE:
                break
            if isinstance(event, str):
                try:
                    decoded = json.loads(event)
                except (json.JSONDecodeError, ValueError) as exc:
                    raise ValueError("Modal GPU stream returned invalid JSON.") from exc
                if not isinstance(decoded, dict):
                    raise TypeError("Modal GPU stream event must be a JSON object.")
                encoded = event
            elif isinstance(event, dict):
                encoded = json_response(event)
            else:
                raise TypeError("Modal GPU stream returned an unsupported event value.")
            yield f"data: {encoded}\n\n"
    finally:
        thread.join(timeout=1.0)
