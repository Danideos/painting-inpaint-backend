"""Configuration for the Modal backend adapter."""

from __future__ import annotations

import os
import re
from pathlib import PurePosixPath

APP_NAME = os.environ.get("MODAL_APP_NAME", "painting-inpaint-backend-modal")
VOLUME_NAME = "flux-fill-models"
CANNY_VOLUME_NAME = "flux-canny-models"
HF_SECRET_NAME = "huggingface-secret"
API_SECRET_NAME = "painting-inpaint-restoration-api"
API_SECRET_KEY = "RESTORATION_API_KEY"
REGISTRY_SECRET_NAME = "painting-inpaint-ghcr"
HTTP_HEARTBEAT_SECONDS = 15.0
MAX_HTTP_REQUEST_MB = 45

MODEL_ID = "black-forest-labs/FLUX.1-Fill-dev"
LORA_REPO_ID = "danideos/durer-flux-fill-lora"
CANNY_MODEL_ID = "black-forest-labs/FLUX.1-Canny-dev"
CANNY_LORA_REPO_ID = "danideos/durer-flux-canny-lora"

MODELS_DIR = PurePosixPath("/models")
MODEL_DIR = MODELS_DIR / MODEL_ID
LORA_DIR = MODELS_DIR / LORA_REPO_ID
CANNY_MODEL_DIR = MODELS_DIR / CANNY_MODEL_ID
CANNY_LORA_DIR = MODELS_DIR / CANNY_LORA_REPO_ID

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

CANNY_INFERENCE_ENV = {
    "MODEL_PATH": str(CANNY_MODEL_DIR),
    "LORA_PATH": str(CANNY_LORA_DIR),
    "LORA_REQUIRED": "1",
    "LORA_ADAPTER_NAME": "durer_canny_rank64",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "DIFFUSERS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "PYTHONUNBUFFERED": "1",
}

_COMMIT_TAG_PATTERN = re.compile(r":[0-9a-fA-F]{40}$")
_DIGEST_PATTERN = re.compile(r"@sha256:[0-9a-fA-F]{64}$")


def _immutable_image_ref(env_name: str) -> str:
    value = os.environ.get(env_name, "").strip()
    if not value:
        raise RuntimeError(f"{env_name} must contain an immutable backend image tag.")
    if not (_COMMIT_TAG_PATTERN.search(value) or _DIGEST_PATTERN.search(value)):
        raise RuntimeError(
            f"{env_name} must use an immutable 40-character commit tag or "
            "sha256 digest; mutable tags such as latest are not allowed."
        )
    return value


def backend_image_ref() -> str:
    """Return the immutable FLUX Fill backend image reference."""

    return _immutable_image_ref("PAINTING_INPAINT_BACKEND_IMAGE")


def canny_backend_image_ref() -> str:
    """Return the immutable private FLUX-Canny backend image reference."""

    return _immutable_image_ref("PAINTING_INPAINT_CANNY_IMAGE")
