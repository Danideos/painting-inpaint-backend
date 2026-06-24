"""Configuration for the Modal backend adapter."""

from __future__ import annotations

import os
import re
from pathlib import Path

APP_NAME = "painting-inpaint-backend-modal"
VOLUME_NAME = "flux-fill-models"
HF_SECRET_NAME = "huggingface-secret"

MODEL_ID = "black-forest-labs/FLUX.1-Fill-dev"
LORA_REPO_ID = "danideos/durer-flux-fill-lora"

MODELS_DIR = Path("/models")
MODEL_DIR = MODELS_DIR / MODEL_ID
LORA_DIR = MODELS_DIR / LORA_REPO_ID

GPU_TYPE = os.environ.get("MODAL_GPU", "L40S")
PYTHON_VERSION = "3.11"

INFERENCE_ENV = {
    "MODEL_PATH": str(MODEL_DIR),
    "LORA_PATH": str(LORA_DIR),
    "LORA_REQUIRED": "1",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "DIFFUSERS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "PYTHONUNBUFFERED": "1",
}

_COMMIT_TAG_PATTERN = re.compile(r":[0-9a-fA-F]{40}$")
_DIGEST_PATTERN = re.compile(r"@sha256:[0-9a-fA-F]{64}$")


def backend_image_ref() -> str:
    """Return the required immutable backend image reference."""

    value = os.environ.get("PAINTING_INPAINT_BACKEND_IMAGE", "").strip()
    if not value:
        raise RuntimeError(
            "PAINTING_INPAINT_BACKEND_IMAGE must contain an immutable backend image tag."
        )
    if not (_COMMIT_TAG_PATTERN.search(value) or _DIGEST_PATTERN.search(value)):
        raise RuntimeError(
            "PAINTING_INPAINT_BACKEND_IMAGE must use an immutable 40-character commit tag or "
            "sha256 digest; mutable tags such as latest are not allowed."
        )
    return value
