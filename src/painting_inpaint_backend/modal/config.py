"""Configuration for the Modal backend adapter."""

from __future__ import annotations

import os
import re
from pathlib import PurePosixPath

APP_NAME = os.environ.get("MODAL_APP_NAME", "painting-inpaint-backend-modal")
VOLUME_NAME = "flux-fill-models"
CANNY_VOLUME_NAME = "flux-canny-models"
SD15_VOLUME_NAME = "sd15-inpaint-models"
SDXL_VOLUME_NAME = "sdxl-inpaint-models"
QWEN_EDIT_VOLUME_NAME = "qwen-edit-models"
SD35_VOLUME_NAME = "sd35-inpaint-models"
SDXL_BRUSHNET_VOLUME_NAME = "sdxl-brushnet-models"
HF_SECRET_NAME = "huggingface-secret"
API_SECRET_NAME = os.environ.get(
    "MODAL_API_SECRET_NAME",
    "painting-inpaint-restoration-api",
)
API_SECRET_KEY = "RESTORATION_API_KEY"
HTTP_HEARTBEAT_SECONDS = 15.0
MAX_HTTP_REQUEST_MB = 45

MODEL_ID = "black-forest-labs/FLUX.1-Fill-dev"
LORA_REPO_ID = "danideos/durer-flux-fill-lora"
CANNY_MODEL_ID = "black-forest-labs/FLUX.1-Canny-dev"
CANNY_LORA_REPO_ID = "danideos/durer-flux-canny-lora"
SD15_MODEL_ID = "runwayml/stable-diffusion-inpainting"
SDXL_MODEL_ID = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
QWEN_EDIT_MODEL_ID = "Qwen/Qwen-Image-Edit"
SD35_BASE_MODEL_ID = "stabilityai/stable-diffusion-3-medium-diffusers"
SD35_CONTROLNET_MODEL_ID = "alimama-creative/SD3-Controlnet-Inpainting"
SDXL_BRUSHNET_BASE_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
SDXL_BRUSHNET_MODEL_ID = "hngan/brushnet_segmentation_mask_brushnet_ckpt_sdxl_v0"
SDXL_BRUSHNET_MODEL_SUBDIR = "segmentation_mask_brushnet_ckpt_sdxl_v0"
SDXL_BRUSHNET_VAE_MODEL_ID = "madebyollin/sdxl-vae-fp16-fix"

MODELS_DIR = PurePosixPath("/models")
SD15_MODELS_DIR = PurePosixPath("/sd15_models")
SDXL_MODELS_DIR = PurePosixPath("/sdxl_models")
QWEN_EDIT_MODELS_DIR = PurePosixPath("/qwen_edit_models")
SD35_MODELS_DIR = PurePosixPath("/sd35_models")
SDXL_BRUSHNET_MODELS_DIR = PurePosixPath("/sdxl_brushnet_models")
MODEL_DIR = MODELS_DIR / MODEL_ID
LORA_DIR = MODELS_DIR / LORA_REPO_ID
CANNY_MODEL_DIR = MODELS_DIR / CANNY_MODEL_ID
CANNY_LORA_DIR = MODELS_DIR / CANNY_LORA_REPO_ID
SD15_MODEL_DIR = SD15_MODELS_DIR / SD15_MODEL_ID
SDXL_MODEL_DIR = SDXL_MODELS_DIR / SDXL_MODEL_ID
QWEN_EDIT_MODEL_DIR = QWEN_EDIT_MODELS_DIR / QWEN_EDIT_MODEL_ID
SD35_BASE_MODEL_DIR = SD35_MODELS_DIR / SD35_BASE_MODEL_ID
SD35_CONTROLNET_MODEL_DIR = SD35_MODELS_DIR / SD35_CONTROLNET_MODEL_ID
SDXL_BRUSHNET_BASE_MODEL_DIR = SDXL_BRUSHNET_MODELS_DIR / SDXL_BRUSHNET_BASE_MODEL_ID
SDXL_BRUSHNET_REPO_DIR = SDXL_BRUSHNET_MODELS_DIR / SDXL_BRUSHNET_MODEL_ID
SDXL_BRUSHNET_MODEL_DIR = SDXL_BRUSHNET_REPO_DIR / SDXL_BRUSHNET_MODEL_SUBDIR
SDXL_BRUSHNET_VAE_MODEL_DIR = SDXL_BRUSHNET_MODELS_DIR / SDXL_BRUSHNET_VAE_MODEL_ID
HYBRID_FILL_MODELS_DIR = PurePosixPath("/models/fill")
HYBRID_CANNY_MODELS_DIR = PurePosixPath("/models/canny")
HYBRID_FILL_MODEL_DIR = HYBRID_FILL_MODELS_DIR / MODEL_ID
HYBRID_FILL_LORA_DIR = HYBRID_FILL_MODELS_DIR / LORA_REPO_ID
HYBRID_CANNY_MODEL_DIR = HYBRID_CANNY_MODELS_DIR / CANNY_MODEL_ID
HYBRID_CANNY_LORA_DIR = HYBRID_CANNY_MODELS_DIR / CANNY_LORA_REPO_ID

GPU_TYPE = os.environ.get("MODAL_GPU", "L40S")
QWEN_GPU_TYPE = os.environ.get("MODAL_QWEN_GPU", "H100")
SD35_GPU_TYPE = os.environ.get("MODAL_SD35_GPU", "A100-80GB")
SDXL_BRUSHNET_GPU_TYPE = os.environ.get("MODAL_SDXL_BRUSHNET_GPU", "A100-80GB")
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

HYBRID_INFERENCE_ENV = {
    "CANNY_MODEL_PATH": str(HYBRID_CANNY_MODEL_DIR),
    "CANNY_LORA_PATH": str(HYBRID_CANNY_LORA_DIR),
    "CANNY_LORA_ADAPTER_NAME": "durer_canny_rank64",
    "FILL_MODEL_PATH": str(HYBRID_FILL_MODEL_DIR),
    "FILL_LORA_PATH": str(HYBRID_FILL_LORA_DIR),
    "FILL_LORA_ADAPTER_NAME": "durer",
    "LORA_REQUIRED": "1",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "DIFFUSERS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "PYTHONUNBUFFERED": "1",
}

SD15_INFERENCE_ENV = {
    "MODEL_PATH": str(SD15_MODEL_DIR),
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "DIFFUSERS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "PYTHONUNBUFFERED": "1",
}

SDXL_INFERENCE_ENV = {
    "MODEL_PATH": str(SDXL_MODEL_DIR),
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "DIFFUSERS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "PYTHONUNBUFFERED": "1",
}

QWEN_EDIT_INFERENCE_ENV = {
    "MODEL_PATH": str(QWEN_EDIT_MODEL_DIR),
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "DIFFUSERS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "PYTHONUNBUFFERED": "1",
}

SD35_INFERENCE_ENV = {
    "MODEL_PATH": str(SD35_BASE_MODEL_DIR),
    "CONTROLNET_PATH": str(SD35_CONTROLNET_MODEL_DIR),
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "DIFFUSERS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "PYTHONUNBUFFERED": "1",
}

SDXL_BRUSHNET_INFERENCE_ENV = {
    "MODEL_PATH": str(SDXL_BRUSHNET_BASE_MODEL_DIR),
    "BRUSHNET_PATH": str(SDXL_BRUSHNET_MODEL_DIR),
    "VAE_PATH": str(SDXL_BRUSHNET_VAE_MODEL_DIR),
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


def canny_backend_image_ref(*, required: bool = True) -> str | None:
    """Return the immutable FLUX-Canny backend image reference."""

    if not required and not os.environ.get("PAINTING_INPAINT_CANNY_IMAGE", "").strip():
        return None
    return _immutable_image_ref("PAINTING_INPAINT_CANNY_IMAGE")
