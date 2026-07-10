from __future__ import annotations

from base64 import b64decode
from contextlib import nullcontext
from io import BytesIO
from types import SimpleNamespace

import pytest
from PIL import Image

from painting_inpaint_backend.core.image_io import image_to_base64
from painting_inpaint_backend.core.inference import (
    InferenceService,
    WorkerInputError,
    load_request_images,
    parse_request_settings,
)
from painting_inpaint_backend.core.model_loading import LoadedPipeline
from painting_inpaint_backend.core.sd15_service import SD15InferenceService
from painting_inpaint_backend.core.sd_inpaint_base import LoadedSDPipeline


class _FakeTorch:
    @staticmethod
    def inference_mode():
        return nullcontext()


class _FakeInferencePipeline:
    def __init__(self):
        self.adapter_calls = []

    def set_adapters(self, adapter_names, *, adapter_weights):
        self.adapter_calls.append((list(adapter_names), list(adapter_weights)))

    def __call__(self, **kwargs):
        return SimpleNamespace(images=[Image.new("RGB", kwargs["image"].size, "blue")])


class _FakeSDPipeline:
    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(images=[Image.new("RGB", kwargs["image"].size, "blue")])


def _worker_with_loaded_lora():
    pipe = _FakeInferencePipeline()
    worker = InferenceService()
    worker._loaded = LoadedPipeline(
        pipe=pipe,
        torch=_FakeTorch(),
        torch_dtype="fake",
        model={"model_id": "fake"},
        lora={
            "loaded": True,
            "repo_id": "example/durer-lora",
            "filename": "pytorch_lora_weights.safetensors",
            "revision": "main",
            "local_path": "/tmp/fake-lora.safetensors",
            "adapter_name": "durer",
            "elapsed_seconds": 0.1,
            "required": False,
            "error": None,
        },
        supports_negative_prompt=False,
        timings={},
    )
    return worker, pipe


def _base64_request(**overrides):
    image = Image.new("RGB", (2, 2), "red")
    mask = Image.new("L", (2, 2), 255)
    payload = {
        "prompt": "DURER_RESTO",
        "image_base64": image_to_base64(image),
        "mask_base64": image_to_base64(mask),
    }
    payload.update(overrides)
    return payload


def _decode_response_image(encoded: str) -> Image.Image:
    return Image.open(BytesIO(b64decode(encoded))).convert("RGB")


def test_parse_request_settings_defaults_partial_noise_to_full_schedule():
    settings = parse_request_settings({"prompt": ""})

    assert settings.prompt == ""
    assert settings.partial_noise == 1.0
    assert settings.guidance_scale == 30.0
    assert settings.num_inference_steps == 28
    assert settings.lora_scale == 1.0


def test_parse_request_settings_validates_prompt_presence():
    with pytest.raises(WorkerInputError, match="prompt is required"):
        parse_request_settings({})


def test_parse_request_settings_rejects_negative_lora_scale():
    with pytest.raises(WorkerInputError, match="lora_scale must be non-negative"):
        parse_request_settings({"prompt": "", "lora_scale": -1})


def test_load_request_images_accepts_base64_image_and_mask():
    image = Image.new("RGB", (2, 2), "red")
    mask = Image.new("L", (2, 2), 0)
    mask.putpixel((0, 0), 255)

    loaded_image, loaded_mask = load_request_images(
        {
            "image_base64": image_to_base64(image),
            "mask_base64": image_to_base64(mask),
        }
    )

    assert loaded_image.size == (2, 2)
    assert loaded_mask.mode == "L"
    assert loaded_mask.getpixel((0, 0)) == 255
    assert loaded_mask.getpixel((1, 1)) == 0


def test_worker_reports_default_and_request_specific_lora_scales():
    worker, pipe = _worker_with_loaded_lora()

    default_response = worker.run(_base64_request())
    half_response = worker.run(_base64_request(lora_scale=0.5))
    zero_response = worker.run(_base64_request(lora_scale=0))

    assert default_response["lora"]["loaded"] is True
    assert default_response["lora"]["requested_scale"] == 1.0
    assert default_response["lora"]["effective_scale"] == 1.0
    assert half_response["lora"]["requested_scale"] == 0.5
    assert half_response["lora"]["effective_scale"] == 0.5
    assert zero_response["lora"]["requested_scale"] == 0.0
    assert zero_response["lora"]["effective_scale"] == 0.0
    assert pipe.adapter_calls == [
        (["durer"], [1.0]),
        (["durer"], [0.5]),
        (["durer"], [0.0]),
    ]


def test_sd15_service_hard_composites_and_reports_baseline_settings():
    pipe = _FakeSDPipeline()
    worker = SD15InferenceService()
    worker._loaded = LoadedSDPipeline(
        pipe=pipe,
        torch=_FakeTorch(),
        torch_dtype="fake",
        model={"method": "sd15_inpaint", "model_id": "fake-sd15"},
        supports_negative_prompt=True,
        timings={"from_pretrained_seconds": 0.0},
    )
    image = Image.new("RGB", (2, 2), "red")
    mask = Image.new("L", (2, 2), 0)
    mask.putpixel((0, 0), 255)

    response = worker.run(
        {
            "method": "sd15_inpaint",
            "prompt": "",
            "negative_prompt": "bad",
            "image_base64": image_to_base64(image),
            "mask_base64": image_to_base64(mask),
            "guidance_scale": 7.5,
            "num_inference_steps": 30,
            "strength": 0.8,
        }
    )
    restored = _decode_response_image(response["image_base64"])

    assert restored.getpixel((0, 0)) == (0, 0, 255)
    assert restored.getpixel((1, 1)) == (255, 0, 0)
    assert response["model"]["method"] == "sd15_inpaint"
    assert response["inference_settings"]["strength"] == 0.8
    assert response["inference_settings"]["negative_prompt"] == "bad"
    assert pipe.calls[0]["strength"] == 0.8
    assert pipe.calls[0]["negative_prompt"] == "bad"
