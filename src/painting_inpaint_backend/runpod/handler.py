"""RunPod Serverless entrypoint."""

from __future__ import annotations

import logging
import os
import traceback
from collections.abc import Iterator
from typing import Any

from ..core.inference import WorkerInputError
from .service import run_job_input, run_job_input_streaming

try:
    import runpod
except Exception:  # pragma: no cover - local import smoke tests may not install runpod.
    runpod = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
LOGGER = logging.getLogger(__name__)


def _payload_from_job(job: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(job, dict):
        raise WorkerInputError("RunPod job must be a JSON object.")
    payload = job.get("input", job)
    if not isinstance(payload, dict):
        raise WorkerInputError("RunPod job input must be a JSON object.")
    return payload


def _stream_progress_requested(payload: dict[str, Any]) -> bool:
    value = payload.get("stream_progress", False)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def handler(job: dict[str, Any]) -> dict[str, Any] | list[dict[str, Any]]:
    """Standard RunPod handler function.

    RunPod only treats configured generator functions as streaming handlers. This
    plain handler is kept for non-streaming mode and local compatibility.
    """

    try:
        payload = _payload_from_job(job)
        if _stream_progress_requested(payload):
            return list(
                run_job_input_streaming(
                    payload,
                    job_id=str(job.get("id")) if job.get("id") is not None else None,
                )
            )
        return run_job_input(payload)
    except WorkerInputError as exc:
        LOGGER.warning("Invalid request: %s", exc)
        return {"error": str(exc), "error_type": type(exc).__name__}
    except Exception as exc:
        LOGGER.exception("Inference failed")
        return {
            "error": str(exc),
            "error_type": type(exc).__name__,
            "traceback": traceback.format_exc(limit=5),
        }


def streaming_handler(job: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Generator RunPod handler function used for the /stream endpoint."""

    try:
        payload = _payload_from_job(job)
        if _stream_progress_requested(payload):
            yield from run_job_input_streaming(
                payload,
                job_id=str(job.get("id")) if job.get("id") is not None else None,
            )
            return
        yield run_job_input(payload)
    except WorkerInputError as exc:
        LOGGER.warning("Invalid request: %s", exc)
        yield {"error": str(exc), "error_type": type(exc).__name__}
    except Exception as exc:
        LOGGER.exception("Inference failed")
        yield {
            "error": str(exc),
            "error_type": type(exc).__name__,
            "traceback": traceback.format_exc(limit=5),
        }


def _runpod_handler() -> Any:
    mode = os.environ.get("RUNPOD_HANDLER_MODE", "streaming").strip().lower()
    if mode in {"standard", "sync", "non_streaming", "non-streaming"}:
        return handler
    return streaming_handler


if __name__ == "__main__":
    if runpod is None:
        raise RuntimeError("The runpod package is required to start the serverless worker.")
    runpod.serverless.start(
        {"handler": _runpod_handler(), "return_aggregate_stream": True}
    )
