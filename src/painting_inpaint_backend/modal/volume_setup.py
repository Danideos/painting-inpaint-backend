"""Modal Volume population and inspection utilities."""

from __future__ import annotations

import os
import shutil
import time
from typing import Any

import modal

from .config import (
    APP_NAME,
    HF_SECRET_NAME,
    LORA_DIR,
    LORA_REPO_ID,
    MODEL_DIR,
    MODEL_ID,
    MODELS_DIR,
    VOLUME_NAME,
    backend_image_ref,
)
from .volume_utils import READY_MARKER_NAME, snapshot_summary, write_ready_marker

app = modal.App(f"{APP_NAME}-volume")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

_BACKEND_IMAGE_REF = backend_image_ref()
download_image = modal.Image.from_registry(_BACKEND_IMAGE_REF).env(
    {"PAINTING_INPAINT_BACKEND_IMAGE": _BACKEND_IMAGE_REF}
)

@app.function(
    image=download_image,
    volumes={str(MODELS_DIR): volume},
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
    timeout=7200,
)
def populate_volume(force: bool = False) -> dict[str, Any]:
    """Download FLUX Fill and LoRA snapshots into the Modal Volume."""

    from huggingface_hub import HfApi, snapshot_download
    from huggingface_hub.errors import HfHubHTTPError

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(f"Modal secret {HF_SECRET_NAME!r} must expose HF_TOKEN.")

    started = time.perf_counter()
    MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
    LORA_DIR.parent.mkdir(parents=True, exist_ok=True)

    downloads: list[dict[str, Any]] = []
    for repo_id, target in ((MODEL_ID, MODEL_DIR), (LORA_REPO_ID, LORA_DIR)):
        item_started = time.perf_counter()
        ready_marker = target / READY_MARKER_NAME
        if force or not ready_marker.exists():
            if target.exists():
                shutil.rmtree(target)
            target.mkdir(parents=True, exist_ok=True)
            try:
                HfApi().model_info(repo_id=repo_id, token=token)
                snapshot_download(
                    repo_id=repo_id,
                    local_dir=str(target),
                    token=token,
                    local_dir_use_symlinks=False,
                )
            except HfHubHTTPError as exc:
                status = getattr(getattr(exc, "response", None), "status_code", "unknown")
                shutil.rmtree(target, ignore_errors=True)
                volume.commit()
                raise RuntimeError(
                    f"Hugging Face access failed for {repo_id!r} (HTTP {status}). "
                    "Confirm that HF_TOKEN is valid, has read access, and belongs to an "
                    "account that accepted the gated model terms."
                ) from None
            except Exception:
                shutil.rmtree(target, ignore_errors=True)
                volume.commit()
                raise
            write_ready_marker(target, repo_id=repo_id)
            volume.commit()
            action = "downloaded_and_committed"
        else:
            action = "already_present"
        downloads.append(
            {
                "repo_id": repo_id,
                "target": str(target),
                "action": action,
                "elapsed_seconds": time.perf_counter() - item_started,
                **snapshot_summary(target),
            }
        )

    return {
        "volume": VOLUME_NAME,
        "mount": str(MODELS_DIR),
        "downloads": downloads,
        "total_elapsed_seconds": time.perf_counter() - started,
    }


@app.function(image=download_image, volumes={str(MODELS_DIR): volume}, timeout=600)
def inspect_volume() -> dict[str, Any]:
    """Inspect expected model directories in the Modal Volume."""

    volume.reload()
    return {
        "volume": VOLUME_NAME,
        "mount": str(MODELS_DIR),
        "model": snapshot_summary(MODEL_DIR),
        "lora": snapshot_summary(LORA_DIR),
    }


@app.local_entrypoint()
def main(action: str = "inspect", force: bool = False) -> None:
    """Run Volume utility actions through the backend package."""

    if action == "populate":
        print(populate_volume.remote(force=force))
    elif action == "inspect":
        print(inspect_volume.remote())
    else:
        raise ValueError("action must be one of: inspect, populate")
