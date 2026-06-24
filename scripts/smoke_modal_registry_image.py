"""Verify that Modal can start the pinned registry image with backend environment."""

from __future__ import annotations

import os
import sys

import modal

from painting_inpaint_backend.modal.config import INFERENCE_ENV, backend_image_ref

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

app = modal.App("painting-inpaint-backend-registry-smoke")
image_ref = backend_image_ref()
image = modal.Image.from_registry(image_ref).env(
    {
        **INFERENCE_ENV,
        "PAINTING_INPAINT_BACKEND_IMAGE": image_ref,
    }
)


@app.function(image=image, timeout=300)
def inspect_image() -> dict[str, object]:
    import importlib.util

    return {
        "backend_importable": importlib.util.find_spec("painting_inpaint_backend") is not None,
        "model_path": os.environ.get("MODEL_PATH"),
        "offline": os.environ.get("HF_HUB_OFFLINE"),
    }


def main() -> int:
    with modal.enable_output(), app.run():
        print(inspect_image.remote())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
