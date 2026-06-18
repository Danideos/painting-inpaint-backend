"""RunPod Serverless entrypoint."""

from __future__ import annotations

import logging
import traceback
from typing import Any

from .inference import WorkerInputError, run_job_input

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


def handler(job: dict[str, Any]) -> dict[str, Any]:
    """RunPod handler function."""

    try:
        return run_job_input(_payload_from_job(job))
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
    runpod.serverless.start({"handler": handler})
