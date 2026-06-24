"""Modal app using the shared backend package from an immutable registry image."""

from __future__ import annotations

import time
from typing import Any

import modal

from .boundary import json_response, safe_remote_error
from .config import (
    APP_NAME,
    GPU_TYPE,
    INFERENCE_ENV,
    MODELS_DIR,
    VOLUME_NAME,
    backend_image_ref,
)

app = modal.App(APP_NAME)
model_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
_BACKEND_IMAGE_REF = backend_image_ref()
inference_image = modal.Image.from_registry(_BACKEND_IMAGE_REF).env(
    {
        **INFERENCE_ENV,
        "PAINTING_INPAINT_BACKEND_IMAGE": _BACKEND_IMAGE_REF,
    }
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
        self.service = InferenceService()
        self.enter_error = None
        try:
            self.service.get_loaded()
            self.container_init_seconds = time.perf_counter() - started
        except Exception as exc:
            self.enter_error = safe_remote_error(exc, stage="container_initialization")

    @modal.method()
    def restore(self, payload: dict[str, Any]) -> str:
        """Run one provider-neutral restoration request."""

        if self.enter_error is not None:
            return json_response({"modal_error": self.enter_error})
        try:
            output = self.service.run(payload)
            output.setdefault("timings", {})["modal_container_init_seconds"] = (
                self.container_init_seconds
            )
            return json_response(output)
        except Exception as exc:
            return json_response(
                {"modal_error": safe_remote_error(exc, stage="request_inference")}
            )
