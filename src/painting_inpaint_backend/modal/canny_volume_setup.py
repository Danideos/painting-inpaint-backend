"""Populate and inspect the isolated FLUX-Canny Modal Volume."""

from __future__ import annotations

import os
import re
import shutil
import time
from pathlib import Path
from typing import Any

import modal

from .config import (
    APP_NAME,
    CANNY_LORA_DIR,
    CANNY_LORA_REPO_ID,
    CANNY_MODEL_DIR,
    CANNY_MODEL_ID,
    CANNY_VOLUME_NAME,
    HF_SECRET_NAME,
    MODELS_DIR,
    backend_image_ref,
)
from .volume_utils import READY_MARKER_NAME, snapshot_summary, write_ready_marker

app = modal.App(f"{APP_NAME}-canny-volume")
volume = modal.Volume.from_name(CANNY_VOLUME_NAME, create_if_missing=True)

_BACKEND_IMAGE_REF = backend_image_ref()
_SHA_PATTERN = re.compile(r"^[0-9a-fA-F]{40}$")

download_image = modal.Image.from_registry(_BACKEND_IMAGE_REF).env(
    {"PAINTING_INPAINT_BACKEND_IMAGE": _BACKEND_IMAGE_REF}
)


def _required_revision(value: str, env_name: str) -> str:
    if not _SHA_PATTERN.fullmatch(value):
        raise RuntimeError(f"{env_name} must be an immutable 40-character Hugging Face SHA.")
    return value


@app.function(
    image=download_image,
    volumes={str(MODELS_DIR): volume},
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
    timeout=7200,
)
def populate_canny_volume(
    model_revision: str,
    lora_revision: str,
    force: bool = False,
) -> dict[str, Any]:
    """Download pinned FLUX-Canny and rank-64 LoRA snapshots."""

    from huggingface_hub import HfApi, snapshot_download
    from huggingface_hub.errors import HfHubHTTPError

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(f"Modal secret {HF_SECRET_NAME!r} must expose HF_TOKEN.")
    assets = (
        (
            CANNY_MODEL_ID,
            Path(CANNY_MODEL_DIR),
            _required_revision(model_revision, "CANNY_MODEL_REVISION"),
        ),
        (
            CANNY_LORA_REPO_ID,
            Path(CANNY_LORA_DIR),
            _required_revision(lora_revision, "CANNY_LORA_REVISION"),
        ),
    )
    started = time.perf_counter()
    downloads: list[dict[str, Any]] = []
    for repo_id, target, revision in assets:
        item_started = time.perf_counter()
        marker = target / READY_MARKER_NAME
        if force or not marker.exists():
            if target.exists():
                shutil.rmtree(target)
            target.mkdir(parents=True, exist_ok=True)
            try:
                HfApi().model_info(repo_id=repo_id, revision=revision, token=token)
                snapshot_download(
                    repo_id=repo_id,
                    revision=revision,
                    local_dir=str(target),
                    token=token,
                    local_dir_use_symlinks=False,
                )
            except HfHubHTTPError as exc:
                status = getattr(getattr(exc, "response", None), "status_code", "unknown")
                shutil.rmtree(target, ignore_errors=True)
                volume.commit()
                raise RuntimeError(
                    f"Hugging Face access failed for {repo_id!r} (HTTP {status})."
                ) from None
            except Exception:
                shutil.rmtree(target, ignore_errors=True)
                volume.commit()
                raise
            write_ready_marker(target, repo_id=repo_id, revision=revision)
            volume.commit()
            action = "downloaded_and_committed"
        else:
            action = "already_present"
        downloads.append(
            {
                "repo_id": repo_id,
                "revision": revision,
                "target": str(target),
                "action": action,
                "elapsed_seconds": time.perf_counter() - item_started,
                **snapshot_summary(target),
            }
        )
    return {
        "volume": CANNY_VOLUME_NAME,
        "mount": str(MODELS_DIR),
        "downloads": downloads,
        "total_elapsed_seconds": time.perf_counter() - started,
    }


@app.function(image=download_image, volumes={str(MODELS_DIR): volume}, timeout=600)
def inspect_canny_volume() -> dict[str, Any]:
    """Inspect expected Canny model and LoRA directories."""

    volume.reload()
    return {
        "volume": CANNY_VOLUME_NAME,
        "mount": str(MODELS_DIR),
        "model": snapshot_summary(Path(CANNY_MODEL_DIR)),
        "lora": snapshot_summary(Path(CANNY_LORA_DIR)),
    }
