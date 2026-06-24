from __future__ import annotations

import json
from pathlib import Path

import pytest

from painting_inpaint_backend.modal.boundary import json_response, safe_remote_error
from painting_inpaint_backend.modal.config import backend_image_ref

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
