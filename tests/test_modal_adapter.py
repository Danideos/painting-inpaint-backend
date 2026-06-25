from __future__ import annotations

import json
from pathlib import Path

import pytest

from painting_inpaint_backend.modal.boundary import json_response, safe_remote_error
from painting_inpaint_backend.modal.config import (
    CANNY_INFERENCE_ENV,
    INFERENCE_ENV,
    backend_image_ref,
    canny_backend_image_ref,
)
from painting_inpaint_backend.modal.http import (
    bearer_token_matches,
    stream_sse_with_heartbeats,
)

ROOT = Path(__file__).resolve().parents[1]


def test_modal_error_boundary_is_json_safe():
    try:
        raise RuntimeError("model load failed")
    except RuntimeError as exc:
        error = safe_remote_error(exc, stage="container_initialization")

    encoded = json_response({"modal_error": error})

    assert json.loads(encoded)["modal_error"]["error_type"] == "RuntimeError"
    assert "RuntimeError: model load failed" in error["traceback"]


def test_backend_image_requires_immutable_reference(monkeypatch):
    monkeypatch.delenv("PAINTING_INPAINT_BACKEND_IMAGE", raising=False)
    with pytest.raises(RuntimeError, match="PAINTING_INPAINT_BACKEND_IMAGE"):
        backend_image_ref()

    monkeypatch.setenv(
        "PAINTING_INPAINT_BACKEND_IMAGE",
        "ghcr.io/danideos/painting-inpaint-backend:latest",
    )
    with pytest.raises(RuntimeError, match="immutable"):
        backend_image_ref()


def test_backend_image_accepts_commit_tag(monkeypatch):
    image = "ghcr.io/danideos/painting-inpaint-backend:" + "a" * 40
    monkeypatch.setenv("PAINTING_INPAINT_BACKEND_IMAGE", image)

    assert backend_image_ref() == image


def test_canny_image_requires_immutable_reference(monkeypatch):
    monkeypatch.delenv("PAINTING_INPAINT_CANNY_IMAGE", raising=False)
    assert canny_backend_image_ref(required=False) is None
    with pytest.raises(RuntimeError, match="PAINTING_INPAINT_CANNY_IMAGE"):
        canny_backend_image_ref()

    image = "ghcr.io/danideos/painting-inpaint-backend-canny:" + "b" * 40
    monkeypatch.setenv("PAINTING_INPAINT_CANNY_IMAGE", image)

    assert canny_backend_image_ref() == image


def test_modal_paths_are_posix_even_on_windows():
    assert INFERENCE_ENV["MODEL_PATH"].startswith("/models/")
    assert "\\" not in INFERENCE_ENV["MODEL_PATH"]
    assert INFERENCE_ENV["LORA_PATH"].startswith("/models/")
    assert "\\" not in INFERENCE_ENV["LORA_PATH"]
    assert CANNY_INFERENCE_ENV["MODEL_PATH"].startswith("/models/")
    assert CANNY_INFERENCE_ENV["LORA_PATH"].startswith("/models/")
    assert "\\" not in CANNY_INFERENCE_ENV["MODEL_PATH"]
    assert "\\" not in CANNY_INFERENCE_ENV["LORA_PATH"]


def test_modal_adapter_uses_registry_image_without_source_overlay_or_warm_workers():
    source = (ROOT / "src" / "painting_inpaint_backend" / "modal" / "app.py").read_text(
        encoding="utf-8"
    )

    assert "modal.Image.from_registry" in source
    assert "include_source=False" in source
    assert "scaledown_window=2" in source
    assert "min_containers" not in source
    assert "keep_warm" not in source
    assert "schedule=" not in source
    assert "InferenceService" in source
    assert "FluxCannyLanPaintModalBackend" in source
    assert "FluxCannyLanPaintService" in source
    assert "painting-inpaint-ghcr" not in source


@pytest.mark.parametrize(
    ("authorization", "expected"),
    [
        ("Bearer correct-key", True),
        ("bearer correct-key", True),
        ("Bearer wrong-key", False),
        ("Basic correct-key", False),
        (None, False),
    ],
)
def test_modal_bearer_authentication(authorization, expected):
    assert bearer_token_matches(authorization, "correct-key") is expected


def test_modal_sse_stream_has_queued_progress_and_final_event():
    def source():
        yield json_response(
            {
                "type": "progress",
                "event": "inference_step",
                "run_id": "run-1",
                "stage": "inference",
                "message": "step",
                "progress": {"current": 1, "total": 2, "fraction": 0.5},
                "metadata": {},
            }
        )
        yield json_response(
            {
                "type": "final",
                "event": "job_done",
                "run_id": "run-1",
                "stage": "output",
                "message": "done",
                "metadata": {},
                "output": {"image_base64": "expected-final-output"},
            }
        )

    frames = list(stream_sse_with_heartbeats(source, run_id="run-1"))
    events = [json.loads(frame.removeprefix("data: ").strip()) for frame in frames]

    assert [event["event"] for event in events] == [
        "modal_queued",
        "inference_step",
        "job_done",
    ]
    assert "base64" not in json.dumps(events[:-1]).lower()
    assert events[-1]["output"]["image_base64"] == "expected-final-output"


def test_modal_sse_stream_converts_gateway_failure_to_safe_error():
    def source():
        raise RuntimeError("gateway failed")
        yield

    frames = list(stream_sse_with_heartbeats(source, run_id="run-1"))
    events = [json.loads(frame.removeprefix("data: ").strip()) for frame in frames]

    assert events[-1]["type"] == "error"
    assert events[-1]["event"] == "job_failed"
    assert events[-1]["metadata"]["error_type"] == "RuntimeError"


def test_modal_adapter_exposes_streaming_http_contract():
    source = (ROOT / "src" / "painting_inpaint_backend" / "modal" / "app.py").read_text(
        encoding="utf-8"
    )

    assert '@api.post("/v1/restore/stream")' in source
    assert "text/event-stream" in (
        ROOT / "src" / "painting_inpaint_backend" / "modal" / "http.py"
    ).read_text(encoding="utf-8")
    assert "painting-inpaint-restoration-api" in (
        ROOT / "src" / "painting_inpaint_backend" / "modal" / "config.py"
    ).read_text(encoding="utf-8")
    assert '"PAINTING_INPAINT_BACKEND_IMAGE": _BACKEND_IMAGE_REF' in source
    assert '"PAINTING_INPAINT_CANNY_IMAGE": _CANNY_IMAGE_EFFECTIVE_REF' in source
    assert "min_containers" not in source
    assert "from __future__ import annotations" not in source


def test_backend_package_initializers_do_not_eagerly_import_inference():
    root_source = (ROOT / "src" / "painting_inpaint_backend" / "__init__.py").read_text(
        encoding="utf-8"
    )
    core_source = (
        ROOT / "src" / "painting_inpaint_backend" / "core" / "__init__.py"
    ).read_text(encoding="utf-8")

    assert "if TYPE_CHECKING:" in root_source
    assert "if TYPE_CHECKING:" in core_source
    assert "def __getattr__" in root_source
    assert "def __getattr__" in core_source
