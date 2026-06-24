"""Modal app using the shared backend package from an immutable registry image."""

import os
import time
import uuid
from typing import Any

import modal

from .boundary import json_response, safe_remote_error
from .config import (
    API_SECRET_KEY,
    API_SECRET_NAME,
    APP_NAME,
    GPU_TYPE,
    HTTP_HEARTBEAT_SECONDS,
    INFERENCE_ENV,
    MAX_HTTP_REQUEST_MB,
    MODELS_DIR,
    VOLUME_NAME,
    backend_image_ref,
)
from .http import SSE_MEDIA_TYPE, bearer_token_matches, stream_sse_with_heartbeats

app = modal.App(APP_NAME)
model_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
_BACKEND_IMAGE_REF = backend_image_ref()
inference_image = modal.Image.from_registry(_BACKEND_IMAGE_REF).env(
    {
        **INFERENCE_ENV,
        "PAINTING_INPAINT_BACKEND_IMAGE": _BACKEND_IMAGE_REF,
    }
)
web_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install("fastapi>=0.115,<1")
    .env({"PAINTING_INPAINT_BACKEND_IMAGE": _BACKEND_IMAGE_REF})
)


@app.cls(
    image=inference_image,
    gpu=GPU_TYPE,
    volumes={str(MODELS_DIR): model_volume},
    timeout=1800,
    scaledown_window=2,
    include_source=False,
)
class FluxFillModalBackend:
    """Scale-to-zero Modal adapter with one service per container lifecycle."""

    @modal.enter()
    def enter(self) -> None:
        from painting_inpaint_backend.core.inference import InferenceService

        started = time.perf_counter()
        self.service = None
        self.enter_error = None
        try:
            self.service = InferenceService()
        except Exception as exc:
            self.enter_error = safe_remote_error(exc, stage="container_initialization")
        finally:
            self.container_enter_seconds = time.perf_counter() - started

    @modal.method()
    def restore(self, payload: dict[str, Any]) -> str:
        """Run one provider-neutral restoration request."""

        if self.enter_error is not None:
            return json_response({"modal_error": self.enter_error})
        try:
            started = time.perf_counter()
            assert self.service is not None
            output = self.service.run(payload)
            timings = output.setdefault("timings", {})
            timings["modal_container_enter_seconds"] = self.container_enter_seconds
            timings["modal_request_seconds"] = time.perf_counter() - started
            return json_response(output)
        except Exception as exc:
            return json_response(
                {"modal_error": safe_remote_error(exc, stage="request_inference")}
            )

    @modal.method()
    def restore_stream(self, payload: dict[str, Any], run_id: str):
        """Yield JSON progress events across the Modal serialization boundary."""

        from painting_inpaint_backend.core.progress import ProgressReporter
        from painting_inpaint_backend.core.streaming import stream_inference_events

        if self.enter_error is not None:
            reporter = ProgressReporter(run_id=run_id, enabled=True)
            yield json_response(
                reporter.error(
                    stage="container_initialization",
                    message=self.enter_error["error"],
                    metadata=self.enter_error,
                )
            )
            return

        started = time.perf_counter()
        assert self.service is not None
        for event in stream_inference_events(
            payload,
            service=self.service,
            run_id=run_id,
            provider="modal",
            received_message="Modal restoration request received.",
            completion_message="Modal restoration completed.",
            received_metadata={"stream_progress": True},
        ):
            if event.get("type") == "final" and isinstance(event.get("output"), dict):
                timings = event["output"].setdefault("timings", {})
                timings["modal_container_enter_seconds"] = self.container_enter_seconds
                timings["modal_request_seconds"] = time.perf_counter() - started
            yield json_response(event)


@app.function(
    image=web_image,
    secrets=[modal.Secret.from_name(API_SECRET_NAME)],
    timeout=1800,
    scaledown_window=2,
)
@modal.asgi_app()
def restoration_api():
    """Expose the provider-neutral restoration stream over authenticated HTTP."""

    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import StreamingResponse

    api = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    @api.post("/v1/restore/stream")
    async def restore_stream_http(request: Request):
        expected_key = os.environ.get(API_SECRET_KEY, "")
        if not expected_key:
            raise HTTPException(status_code=503, detail="Restoration API is not configured.")
        if not bearer_token_matches(request.headers.get("authorization"), expected_key):
            raise HTTPException(status_code=401, detail="Unauthorized.")

        content_length = request.headers.get("content-length")
        if content_length:
            try:
                content_bytes = int(content_length)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid Content-Length.") from None
            if content_bytes > MAX_HTTP_REQUEST_MB * 1024 * 1024:
                raise HTTPException(status_code=413, detail="Restoration request is too large.")

        try:
            payload = await request.json()
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="Request body must be valid JSON.",
            ) from None
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="Request body must be a JSON object.")

        run_id = uuid.uuid4().hex

        def event_source():
            yield from FluxFillModalBackend().restore_stream.remote_gen(payload, run_id)

        return StreamingResponse(
            stream_sse_with_heartbeats(
                event_source,
                run_id=run_id,
                heartbeat_seconds=HTTP_HEARTBEAT_SECONDS,
            ),
            media_type=SSE_MEDIA_TYPE,
            headers={
                "Cache-Control": "no-cache, no-transform",
                "X-Accel-Buffering": "no",
            },
        )

    return api
