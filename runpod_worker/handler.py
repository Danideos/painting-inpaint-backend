"""RunPod Serverless entrypoint."""

from __future__ import annotations

import logging
import traceback
from collections.abc import Iterator
from typing import Any

from .inference import WorkerInputError, run_job_input, run_job_input_streaming

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


def handler(
    job: dict[str, Any],
) -> dict[str, Any] | Iterator[dict[str, Any]] | list[dict[str, Any]]:
    """RunPod handler function."""

    try:
        payload = _payload_from_job(job)
        if _stream_progress_requested(payload):
            stream = run_job_input_streaming(
                payload,
                job_id=str(job.get("id")) if job.get("id") is not None else None,
            )
            if job.get("id") == "local_test":
                return list(stream)
            return stream
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


if __name__ == "__main__":
    if runpod is None:
        raise RuntimeError("The runpod package is required to start the serverless worker.")
    runpod.serverless.start({"handler": handler, "return_aggregate_stream": True})
