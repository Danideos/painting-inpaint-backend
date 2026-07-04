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
    CANNY_INFERENCE_ENV,
    CANNY_VOLUME_NAME,
    GPU_TYPE,
    HTTP_HEARTBEAT_SECONDS,
    HYBRID_CANNY_MODELS_DIR,
    HYBRID_FILL_MODELS_DIR,
    HYBRID_INFERENCE_ENV,
    INFERENCE_ENV,
    MAX_HTTP_REQUEST_MB,
    MODELS_DIR,
    VOLUME_NAME,
    backend_image_ref,
    canny_backend_image_ref,
)
from .http import SSE_MEDIA_TYPE, bearer_token_matches, stream_sse_with_heartbeats

app = modal.App(APP_NAME)
model_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
canny_model_volume = modal.Volume.from_name(CANNY_VOLUME_NAME, create_if_missing=True)
_INCLUDE_LOCAL_SOURCE = os.environ.get(
    "PAINTING_INPAINT_MODAL_INCLUDE_LOCAL_SOURCE",
    "",
).strip() in {"1", "true", "TRUE", "yes", "YES"}
_BACKEND_IMAGE_REF = backend_image_ref()
_CANNY_IMAGE_REF = canny_backend_image_ref(required=False)
_CANNY_IMAGE_CONFIGURED = _CANNY_IMAGE_REF is not None
_CANNY_IMAGE_EFFECTIVE_REF = _CANNY_IMAGE_REF or _BACKEND_IMAGE_REF


def _with_local_backend_source(image: modal.Image) -> modal.Image:
    if not _INCLUDE_LOCAL_SOURCE:
        return image
    return image.add_local_python_source("painting_inpaint_backend", copy=True).env(
        {"PYTHONPATH": "/root:/app/src"}
    )


inference_image = _with_local_backend_source(
    modal.Image.from_registry(_BACKEND_IMAGE_REF).env(
        {
            **INFERENCE_ENV,
            "PAINTING_INPAINT_BACKEND_IMAGE": _BACKEND_IMAGE_REF,
        }
    )
)
if _CANNY_IMAGE_CONFIGURED:
    canny_inference_image = _with_local_backend_source(
        modal.Image.from_registry(_CANNY_IMAGE_EFFECTIVE_REF).env(
            {
                **CANNY_INFERENCE_ENV,
                "PAINTING_INPAINT_BACKEND_IMAGE": _BACKEND_IMAGE_REF,
                "PAINTING_INPAINT_CANNY_IMAGE": _CANNY_IMAGE_EFFECTIVE_REF,
            }
        )
    )
    hybrid_inference_image = _with_local_backend_source(
        modal.Image.from_registry(_CANNY_IMAGE_EFFECTIVE_REF).env(
            {
                **HYBRID_INFERENCE_ENV,
                "PAINTING_INPAINT_BACKEND_IMAGE": _BACKEND_IMAGE_REF,
                "PAINTING_INPAINT_CANNY_IMAGE": _CANNY_IMAGE_EFFECTIVE_REF,
            }
        )
    )
else:
    canny_inference_image = inference_image
    hybrid_inference_image = inference_image
web_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install("fastapi>=0.115,<1")
    .env(
        {
            "PAINTING_INPAINT_BACKEND_IMAGE": _BACKEND_IMAGE_REF,
            "PAINTING_INPAINT_CANNY_IMAGE": _CANNY_IMAGE_EFFECTIVE_REF,
        }
    )
    .add_local_python_source("painting_inpaint_backend", copy=True)
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


@app.cls(
    image=canny_inference_image,
    gpu=GPU_TYPE,
    volumes={str(MODELS_DIR): canny_model_volume},
    timeout=1800,
    scaledown_window=2,
    include_source=False,
)
class FluxCannyLanPaintModalBackend:
    """Scale-to-zero FLUX-Canny/LanPaint service with isolated assets."""

    @modal.enter()
    def enter(self) -> None:
        started = time.perf_counter()
        self.service = None
        self.enter_error = None
        try:
            if not _CANNY_IMAGE_CONFIGURED:
                raise RuntimeError(
                    "PAINTING_INPAINT_CANNY_IMAGE must contain an immutable Canny image "
                    "tag before invoking method='flux_canny_lanpaint'."
                )
            from painting_inpaint_backend.core.canny_lanpaint import FluxCannyLanPaintService

            self.service = FluxCannyLanPaintService()
        except Exception as exc:
            self.enter_error = safe_remote_error(exc, stage="container_initialization")
        finally:
            self.container_enter_seconds = time.perf_counter() - started

    @modal.method()
    def restore(self, payload: dict[str, Any]) -> str:
        """Run one provider-neutral Canny restoration request."""

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
        """Yield Canny progress events across the Modal serialization boundary."""

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
            received_message="Modal FLUX-Canny request received.",
            completion_message="Modal FLUX-Canny restoration completed.",
            received_metadata={
                "stream_progress": True,
                "method": "flux_canny_lanpaint",
            },
        ):
            if event.get("type") == "final" and isinstance(event.get("output"), dict):
                timings = event["output"].setdefault("timings", {})
                timings["modal_container_enter_seconds"] = self.container_enter_seconds
                timings["modal_request_seconds"] = time.perf_counter() - started
            yield json_response(event)


@app.cls(
    image=hybrid_inference_image,
    gpu=GPU_TYPE,
    volumes={
        str(HYBRID_FILL_MODELS_DIR): model_volume,
        str(HYBRID_CANNY_MODELS_DIR): canny_model_volume,
    },
    timeout=2400,
    scaledown_window=2,
    include_source=False,
)
class FluxCannyFillModalBackend:
    """Scale-to-zero hybrid service: FLUX-Canny/LanPaint then FLUX Fill."""

    @modal.enter()
    def enter(self) -> None:
        started = time.perf_counter()
        self.service = None
        self.enter_error = None
        try:
            if not _CANNY_IMAGE_CONFIGURED:
                raise RuntimeError(
                    "PAINTING_INPAINT_CANNY_IMAGE must contain an immutable Canny image "
                    "tag before invoking method='flux_canny_fill'."
                )
            from painting_inpaint_backend.core.canny_fill import FluxCannyFillService

            canny_env = {
                "MODEL_PATH": os.environ["CANNY_MODEL_PATH"],
                "LORA_PATH": os.environ["CANNY_LORA_PATH"],
                "LORA_ADAPTER_NAME": os.environ["CANNY_LORA_ADAPTER_NAME"],
                "LORA_REQUIRED": "1",
            }
            fill_env = {
                "MODEL_PATH": os.environ["FILL_MODEL_PATH"],
                "LORA_PATH": os.environ["FILL_LORA_PATH"],
                "LORA_ADAPTER_NAME": os.environ["FILL_LORA_ADAPTER_NAME"],
                "LORA_REQUIRED": "1",
            }
            self.service = FluxCannyFillService(canny_env=canny_env, fill_env=fill_env)
        except Exception as exc:
            self.enter_error = safe_remote_error(exc, stage="container_initialization")
        finally:
            self.container_enter_seconds = time.perf_counter() - started

    @modal.method()
    def restore(self, payload: dict[str, Any]) -> str:
        """Run one provider-neutral hybrid restoration request."""

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
        """Yield hybrid progress events across the Modal serialization boundary."""

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
            received_message="Modal hybrid restoration request received.",
            completion_message="Modal hybrid restoration completed.",
            received_metadata={
                "stream_progress": True,
                "method": "flux_canny_fill",
            },
        ):
            if event.get("type") == "final" and isinstance(event.get("output"), dict):
                timings = event["output"].setdefault("timings", {})
                timings["modal_container_enter_seconds"] = self.container_enter_seconds
                timings["modal_request_seconds"] = time.perf_counter() - started
            yield json_response(event)


@app.cls(
    image=canny_inference_image,
    gpu=GPU_TYPE,
    volumes={str(MODELS_DIR): canny_model_volume},
    timeout=1800,
    scaledown_window=2,
    include_source=False,
)
class FluxCannyLanPaintNativeModalBackend:
    """Scale-to-zero direct FLUX-Canny/LanPaint service without mask-aware Canny tweaks."""

    @modal.enter()
    def enter(self) -> None:
        started = time.perf_counter()
        self.service = None
        self.enter_error = None
        try:
            if not _CANNY_IMAGE_CONFIGURED:
                raise RuntimeError(
                    "PAINTING_INPAINT_CANNY_IMAGE must contain an immutable Canny image "
                    "tag before invoking method='flux_canny_lanpaint_native'."
                )
            from painting_inpaint_backend.core.canny_lanpaint import FluxCannyLanPaintService
            from painting_inpaint_backend.core.methods import FLUX_CANNY_LANPAINT_NATIVE_METHOD

            self.service = FluxCannyLanPaintService(
                method=FLUX_CANNY_LANPAINT_NATIVE_METHOD,
                native_lanpaint=True,
            )
        except Exception as exc:
            self.enter_error = safe_remote_error(exc, stage="container_initialization")
        finally:
            self.container_enter_seconds = time.perf_counter() - started

    @modal.method()
    def restore(self, payload: dict[str, Any]) -> str:
        """Run one provider-neutral native Canny/LanPaint restoration request."""

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
        """Yield native Canny/LanPaint progress events across the Modal boundary."""

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
            received_message="Modal native FLUX-Canny request received.",
            completion_message="Modal native FLUX-Canny restoration completed.",
            received_metadata={
                "stream_progress": True,
                "method": "flux_canny_lanpaint_native",
            },
        ):
            if event.get("type") == "final" and isinstance(event.get("output"), dict):
                timings = event["output"].setdefault("timings", {})
                timings["modal_container_enter_seconds"] = self.container_enter_seconds
                timings["modal_request_seconds"] = time.perf_counter() - started
            yield json_response(event)


@app.cls(
    image=hybrid_inference_image,
    gpu=GPU_TYPE,
    volumes={
        str(HYBRID_FILL_MODELS_DIR): model_volume,
        str(HYBRID_CANNY_MODELS_DIR): canny_model_volume,
    },
    timeout=2400,
    scaledown_window=2,
    include_source=False,
)
class FluxFillCannyNativeModalBackend:
    """Scale-to-zero hybrid service: FLUX Fill pre-fill then FLUX-Canny/LanPaint native."""

    @modal.enter()
    def enter(self) -> None:
        started = time.perf_counter()
        self.service = None
        self.enter_error = None
        try:
            if not _CANNY_IMAGE_CONFIGURED:
                raise RuntimeError(
                    "PAINTING_INPAINT_CANNY_IMAGE must contain an immutable Canny image "
                    "tag before invoking method='flux_fill_canny_native'."
                )
            from painting_inpaint_backend.core.canny_fill import FluxFillCannyNativeService

            canny_env = {
                "MODEL_PATH": os.environ["CANNY_MODEL_PATH"],
                "LORA_PATH": os.environ["CANNY_LORA_PATH"],
                "LORA_ADAPTER_NAME": os.environ["CANNY_LORA_ADAPTER_NAME"],
                "LORA_REQUIRED": "1",
            }
            fill_env = {
                "MODEL_PATH": os.environ["FILL_MODEL_PATH"],
                "LORA_PATH": os.environ["FILL_LORA_PATH"],
                "LORA_ADAPTER_NAME": os.environ["FILL_LORA_ADAPTER_NAME"],
                "LORA_REQUIRED": "1",
            }
            self.service = FluxFillCannyNativeService(canny_env=canny_env, fill_env=fill_env)
        except Exception as exc:
            self.enter_error = safe_remote_error(exc, stage="container_initialization")
        finally:
            self.container_enter_seconds = time.perf_counter() - started

    @modal.method()
    def restore_stream(self, payload: dict[str, Any], run_id: str):
        """Yield fill-then-canny progress events across the Modal boundary."""

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
            received_message="Modal fill-then-canny request received.",
            completion_message="Modal fill-then-canny restoration completed.",
            received_metadata={
                "stream_progress": True,
                "method": "flux_fill_canny_native",
            },
        ):
            if event.get("type") == "final" and isinstance(event.get("output"), dict):
                timings = event["output"].setdefault("timings", {})
                timings["modal_container_enter_seconds"] = self.container_enter_seconds
                timings["modal_request_seconds"] = time.perf_counter() - started
            yield json_response(event)


@app.cls(
    image=hybrid_inference_image,
    gpu=GPU_TYPE,
    volumes={
        str(HYBRID_FILL_MODELS_DIR): model_volume,
        str(HYBRID_CANNY_MODELS_DIR): canny_model_volume,
    },
    timeout=3600,
    scaledown_window=2,
    include_source=False,
)
class FluxFillCannyFillModalBackend:
    """Scale-to-zero three-stage service: FLUX Fill → Canny/LanPaint native → FLUX Fill."""

    @modal.enter()
    def enter(self) -> None:
        started = time.perf_counter()
        self.service = None
        self.enter_error = None
        try:
            if not _CANNY_IMAGE_CONFIGURED:
                raise RuntimeError(
                    "PAINTING_INPAINT_CANNY_IMAGE must contain an immutable Canny image "
                    "tag before invoking method='flux_fill_canny_fill'."
                )
            from painting_inpaint_backend.core.canny_fill import FluxFillCannyFillService

            canny_env = {
                "MODEL_PATH": os.environ["CANNY_MODEL_PATH"],
                "LORA_PATH": os.environ["CANNY_LORA_PATH"],
                "LORA_ADAPTER_NAME": os.environ["CANNY_LORA_ADAPTER_NAME"],
                "LORA_REQUIRED": "1",
            }
            fill_env = {
                "MODEL_PATH": os.environ["FILL_MODEL_PATH"],
                "LORA_PATH": os.environ["FILL_LORA_PATH"],
                "LORA_ADAPTER_NAME": os.environ["FILL_LORA_ADAPTER_NAME"],
                "LORA_REQUIRED": "1",
            }
            self.service = FluxFillCannyFillService(canny_env=canny_env, fill_env=fill_env)
        except Exception as exc:
            self.enter_error = safe_remote_error(exc, stage="container_initialization")
        finally:
            self.container_enter_seconds = time.perf_counter() - started

    @modal.method()
    def restore_stream(self, payload: dict[str, Any], run_id: str):
        """Yield fill→canny→fill progress events across the Modal boundary."""

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
            received_message="Modal fill→canny→fill request received.",
            completion_message="Modal fill→canny→fill restoration completed.",
            received_metadata={
                "stream_progress": True,
                "method": "flux_fill_canny_fill",
            },
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
        from painting_inpaint_backend.core.methods import (
            FLUX_CANNY_FILL_METHOD,
            FLUX_CANNY_LANPAINT_METHOD,
            FLUX_CANNY_LANPAINT_NATIVE_METHOD,
            FLUX_FILL_CANNY_FILL_METHOD,
            FLUX_FILL_CANNY_NATIVE_METHOD,
            normalize_method,
        )

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

        try:
            method = normalize_method(payload.get("method"))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None

        run_id = uuid.uuid4().hex

        def event_source():
            if method == FLUX_CANNY_FILL_METHOD:
                yield from FluxCannyFillModalBackend().restore_stream.remote_gen(
                    payload,
                    run_id,
                )
            elif method == FLUX_CANNY_LANPAINT_METHOD:
                yield from FluxCannyLanPaintModalBackend().restore_stream.remote_gen(
                    payload,
                    run_id,
                )
            elif method == FLUX_CANNY_LANPAINT_NATIVE_METHOD:
                yield from FluxCannyLanPaintNativeModalBackend().restore_stream.remote_gen(
                    payload,
                    run_id,
                )
            elif method == FLUX_FILL_CANNY_NATIVE_METHOD:
                yield from FluxFillCannyNativeModalBackend().restore_stream.remote_gen(
                    payload,
                    run_id,
                )
            elif method == FLUX_FILL_CANNY_FILL_METHOD:
                yield from FluxFillCannyFillModalBackend().restore_stream.remote_gen(
                    payload,
                    run_id,
                )
            else:
                yield from FluxFillModalBackend().restore_stream.remote_gen(payload, run_id)

        return StreamingResponse(
            stream_sse_with_heartbeats(
                event_source,
                run_id=run_id,
                method=method,
                heartbeat_seconds=HTTP_HEARTBEAT_SECONDS,
            ),
            media_type=SSE_MEDIA_TYPE,
            headers={
                "Cache-Control": "no-cache, no-transform",
                "X-Accel-Buffering": "no",
            },
        )

    return api
