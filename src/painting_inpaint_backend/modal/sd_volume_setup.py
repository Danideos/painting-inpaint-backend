"""Populate and inspect Stable Diffusion baseline Modal Volumes."""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import Any

import modal

from .config import (
    APP_NAME,
    HF_SECRET_NAME,
    QWEN_EDIT_MODEL_DIR,
    QWEN_EDIT_MODEL_ID,
    QWEN_EDIT_MODELS_DIR,
    QWEN_EDIT_VOLUME_NAME,
    QWEN_IMAGE_MODEL_DIR,
    QWEN_IMAGE_MODEL_ID,
    QWEN_IMAGE_MODELS_DIR,
    QWEN_IMAGE_VOLUME_NAME,
    SD15_MODEL_DIR,
    SD15_MODEL_ID,
    SD15_MODELS_DIR,
    SD15_VOLUME_NAME,
    SD35_BASE_MODEL_DIR,
    SD35_BASE_MODEL_ID,
    SD35_CONTROLNET_MODEL_DIR,
    SD35_CONTROLNET_MODEL_ID,
    SD35_MODELS_DIR,
    SD35_VOLUME_NAME,
    SDXL_BRUSHNET_BASE_MODEL_DIR,
    SDXL_BRUSHNET_BASE_MODEL_ID,
    SDXL_BRUSHNET_MODEL_ID,
    SDXL_BRUSHNET_MODELS_DIR,
    SDXL_BRUSHNET_REPO_DIR,
    SDXL_BRUSHNET_VAE_MODEL_DIR,
    SDXL_BRUSHNET_VAE_MODEL_ID,
    SDXL_BRUSHNET_VOLUME_NAME,
    SDXL_MODEL_DIR,
    SDXL_MODEL_ID,
    SDXL_MODELS_DIR,
    SDXL_VOLUME_NAME,
    backend_image_ref,
)
from .volume_utils import READY_MARKER_NAME, snapshot_summary, write_ready_marker

app = modal.App(f"{APP_NAME}-sd-volume")
sd15_volume = modal.Volume.from_name(SD15_VOLUME_NAME, create_if_missing=True)
sdxl_volume = modal.Volume.from_name(SDXL_VOLUME_NAME, create_if_missing=True)
qwen_edit_volume = modal.Volume.from_name(QWEN_EDIT_VOLUME_NAME, create_if_missing=True)
qwen_image_volume = modal.Volume.from_name(QWEN_IMAGE_VOLUME_NAME, create_if_missing=True)
sd35_volume = modal.Volume.from_name(SD35_VOLUME_NAME, create_if_missing=True)
sdxl_brushnet_volume = modal.Volume.from_name(
    SDXL_BRUSHNET_VOLUME_NAME,
    create_if_missing=True,
)

_BACKEND_IMAGE_REF = backend_image_ref()
download_image = (
    modal.Image.from_registry(_BACKEND_IMAGE_REF)
    .add_local_python_source("painting_inpaint_backend", copy=True)
    .env(
        {
            "PAINTING_INPAINT_BACKEND_IMAGE": _BACKEND_IMAGE_REF,
            "PYTHONPATH": "/root:/app/src",
        }
    )
)


def _populate_model(
    *,
    repo_id: str,
    target: Path,
    volume: Any,
    volume_name: str,
    mount: Path,
    force: bool,
) -> dict[str, Any]:
    from huggingface_hub import HfApi, snapshot_download
    from huggingface_hub.errors import HfHubHTTPError

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(f"Modal secret {HF_SECRET_NAME!r} must expose HF_TOKEN.")

    started = time.perf_counter()
    target.parent.mkdir(parents=True, exist_ok=True)
    marker = target / READY_MARKER_NAME
    if force or not marker.exists():
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
                "Confirm that HF_TOKEN is valid and has read access."
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

    return {
        "volume": volume_name,
        "mount": str(mount),
        "downloads": [
            {
                "repo_id": repo_id,
                "target": str(target),
                "action": action,
                "elapsed_seconds": time.perf_counter() - started,
                **snapshot_summary(target),
            }
        ],
        "total_elapsed_seconds": time.perf_counter() - started,
    }


def _populate_models(
    *,
    downloads: list[tuple[str, Path]],
    volume: Any,
    volume_name: str,
    mount: Path,
    force: bool,
) -> dict[str, Any]:
    """Populate multiple Hugging Face snapshots into one Modal Volume."""

    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    for repo_id, target in downloads:
        result = _populate_model(
            repo_id=repo_id,
            target=target,
            volume=volume,
            volume_name=volume_name,
            mount=mount,
            force=force,
        )
        results.extend(result["downloads"])

    return {
        "volume": volume_name,
        "mount": str(mount),
        "downloads": results,
        "total_elapsed_seconds": time.perf_counter() - started,
    }


@app.function(
    image=download_image,
    volumes={str(SD15_MODELS_DIR): sd15_volume},
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
    timeout=7200,
)
def populate_sd15_volume(force: bool = False) -> dict[str, Any]:
    """Download SD1.5 inpainting weights into the SD1.5 Modal Volume."""

    return _populate_model(
        repo_id=SD15_MODEL_ID,
        target=Path(SD15_MODEL_DIR),
        volume=sd15_volume,
        volume_name=SD15_VOLUME_NAME,
        mount=Path(SD15_MODELS_DIR),
        force=force,
    )


@app.function(
    image=download_image,
    volumes={str(SDXL_MODELS_DIR): sdxl_volume},
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
    timeout=7200,
)
def populate_sdxl_volume(force: bool = False) -> dict[str, Any]:
    """Download SDXL inpainting weights into the SDXL Modal Volume."""

    return _populate_model(
        repo_id=SDXL_MODEL_ID,
        target=Path(SDXL_MODEL_DIR),
        volume=sdxl_volume,
        volume_name=SDXL_VOLUME_NAME,
        mount=Path(SDXL_MODELS_DIR),
        force=force,
    )


@app.function(
    image=download_image,
    volumes={str(QWEN_EDIT_MODELS_DIR): qwen_edit_volume},
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
    timeout=14400,
)
def populate_qwen_edit_volume(force: bool = False) -> dict[str, Any]:
    """Download Qwen-Image-Edit weights into the Qwen Modal Volume."""

    return _populate_model(
        repo_id=QWEN_EDIT_MODEL_ID,
        target=Path(QWEN_EDIT_MODEL_DIR),
        volume=qwen_edit_volume,
        volume_name=QWEN_EDIT_VOLUME_NAME,
        mount=Path(QWEN_EDIT_MODELS_DIR),
        force=force,
    )


@app.function(
    image=download_image,
    volumes={str(QWEN_IMAGE_MODELS_DIR): qwen_image_volume},
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
    timeout=14400,
)
def populate_qwen_image_volume(force: bool = False) -> dict[str, Any]:
    """Download Qwen-Image text-to-image weights into the Qwen Image Modal Volume."""

    return _populate_model(
        repo_id=QWEN_IMAGE_MODEL_ID,
        target=Path(QWEN_IMAGE_MODEL_DIR),
        volume=qwen_image_volume,
        volume_name=QWEN_IMAGE_VOLUME_NAME,
        mount=Path(QWEN_IMAGE_MODELS_DIR),
        force=force,
    )


@app.function(
    image=download_image,
    volumes={str(SD35_MODELS_DIR): sd35_volume},
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
    timeout=14400,
)
def populate_sd35_volume(force: bool = False) -> dict[str, Any]:
    """Download SD3 base and Alimama inpainting ControlNet weights."""

    return _populate_models(
        downloads=[
            (SD35_BASE_MODEL_ID, Path(SD35_BASE_MODEL_DIR)),
            (SD35_CONTROLNET_MODEL_ID, Path(SD35_CONTROLNET_MODEL_DIR)),
        ],
        volume=sd35_volume,
        volume_name=SD35_VOLUME_NAME,
        mount=Path(SD35_MODELS_DIR),
        force=force,
    )


@app.function(
    image=download_image,
    volumes={str(SDXL_BRUSHNET_MODELS_DIR): sdxl_brushnet_volume},
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
    timeout=14400,
)
def populate_sdxl_brushnet_volume(force: bool = False) -> dict[str, Any]:
    """Download SDXL base, BrushNet adapter, and fp16 VAE weights."""

    return _populate_models(
        downloads=[
            (SDXL_BRUSHNET_BASE_MODEL_ID, Path(SDXL_BRUSHNET_BASE_MODEL_DIR)),
            (SDXL_BRUSHNET_MODEL_ID, Path(SDXL_BRUSHNET_REPO_DIR)),
            (SDXL_BRUSHNET_VAE_MODEL_ID, Path(SDXL_BRUSHNET_VAE_MODEL_DIR)),
        ],
        volume=sdxl_brushnet_volume,
        volume_name=SDXL_BRUSHNET_VOLUME_NAME,
        mount=Path(SDXL_BRUSHNET_MODELS_DIR),
        force=force,
    )


@app.function(image=download_image, volumes={str(SD15_MODELS_DIR): sd15_volume}, timeout=600)
def inspect_sd15_volume() -> dict[str, Any]:
    """Inspect expected SD1.5 model directory."""

    sd15_volume.reload()
    return {
        "volume": SD15_VOLUME_NAME,
        "mount": str(SD15_MODELS_DIR),
        "model": snapshot_summary(Path(SD15_MODEL_DIR)),
    }


@app.function(image=download_image, volumes={str(SDXL_MODELS_DIR): sdxl_volume}, timeout=600)
def inspect_sdxl_volume() -> dict[str, Any]:
    """Inspect expected SDXL model directory."""

    sdxl_volume.reload()
    return {
        "volume": SDXL_VOLUME_NAME,
        "mount": str(SDXL_MODELS_DIR),
        "model": snapshot_summary(Path(SDXL_MODEL_DIR)),
    }


@app.function(
    image=download_image,
    volumes={str(QWEN_EDIT_MODELS_DIR): qwen_edit_volume},
    timeout=600,
)
def inspect_qwen_edit_volume() -> dict[str, Any]:
    """Inspect expected Qwen model directory."""

    qwen_edit_volume.reload()
    return {
        "volume": QWEN_EDIT_VOLUME_NAME,
        "mount": str(QWEN_EDIT_MODELS_DIR),
        "model": snapshot_summary(Path(QWEN_EDIT_MODEL_DIR)),
    }


@app.function(
    image=download_image,
    volumes={str(QWEN_IMAGE_MODELS_DIR): qwen_image_volume},
    timeout=600,
)
def inspect_qwen_image_volume() -> dict[str, Any]:
    """Inspect expected Qwen Image model directory."""

    qwen_image_volume.reload()
    return {
        "volume": QWEN_IMAGE_VOLUME_NAME,
        "mount": str(QWEN_IMAGE_MODELS_DIR),
        "model": snapshot_summary(Path(QWEN_IMAGE_MODEL_DIR)),
    }


@app.function(image=download_image, volumes={str(SD35_MODELS_DIR): sd35_volume}, timeout=600)
def inspect_sd35_volume() -> dict[str, Any]:
    """Inspect expected SD3 model directories."""

    sd35_volume.reload()
    return {
        "volume": SD35_VOLUME_NAME,
        "mount": str(SD35_MODELS_DIR),
        "base_model": snapshot_summary(Path(SD35_BASE_MODEL_DIR)),
        "controlnet": snapshot_summary(Path(SD35_CONTROLNET_MODEL_DIR)),
    }


@app.function(
    image=download_image,
    volumes={str(SDXL_BRUSHNET_MODELS_DIR): sdxl_brushnet_volume},
    timeout=600,
)
def inspect_sdxl_brushnet_volume() -> dict[str, Any]:
    """Inspect expected SDXL BrushNet model directories."""

    sdxl_brushnet_volume.reload()
    return {
        "volume": SDXL_BRUSHNET_VOLUME_NAME,
        "mount": str(SDXL_BRUSHNET_MODELS_DIR),
        "base_model": snapshot_summary(Path(SDXL_BRUSHNET_BASE_MODEL_DIR)),
        "brushnet": snapshot_summary(Path(SDXL_BRUSHNET_REPO_DIR)),
        "vae": snapshot_summary(Path(SDXL_BRUSHNET_VAE_MODEL_DIR)),
    }
