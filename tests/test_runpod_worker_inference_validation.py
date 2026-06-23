from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from PIL import Image

from runpod_worker.image_io import image_to_base64
from runpod_worker.inference import (
    FluxFillWorker,
    WorkerInputError,
    load_request_images,
    parse_request_settings,
)
from runpod_worker.model_loading import LoadedPipeline


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


def _worker_with_loaded_lora():
    pipe = _FakeInferencePipeline()
    worker = FluxFillWorker()
    worker._loaded = LoadedPipeline(
        pipe=pipe,
        torch=_FakeTorch(),
        torch_dtype="fake",
        model={"model_id": "fake", "fp8_enabled": False},
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


def test_parse_request_settings_defaults_partial_noise_to_full_schedule():
    settings = parse_request_settings({"prompt": ""})

    assert settings.prompt == ""
    assert settings.partial_noise == 1.0
    assert settings.guidance_scale == 30.0
    assert settings.num_inference_steps == 28
    assert settings.lora_scale == 1.0
    assert settings.fp8 is False


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (True, True),
        (False, False),
        ("true", True),
        ("false", False),
    ],
)
def test_parse_request_settings_accepts_fp8_flag(value, expected):
    settings = parse_request_settings({"prompt": "", "fp8": value})

    assert settings.fp8 is expected


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
    assert default_response["inference_settings"]["fp8"] is False
    assert default_response["model"]["fp8_enabled"] is False
    assert half_response["lora"]["requested_scale"] == 0.5
    assert half_response["lora"]["effective_scale"] == 0.5
    assert zero_response["lora"]["requested_scale"] == 0.0
    assert zero_response["lora"]["effective_scale"] == 0.0
    assert pipe.adapter_calls == [
        (["durer"], [1.0]),
        (["durer"], [0.5]),
        (["durer"], [0.0]),
    ]


def test_worker_reloads_cached_pipeline_when_fp8_mode_changes(monkeypatch):
    worker = FluxFillWorker()
    load_calls = []

    def fake_load_pipeline(*, reporter=None, fp8=False):
        load_calls.append(fp8)
        return LoadedPipeline(
            pipe=_FakeInferencePipeline(),
            torch=_FakeTorch(),
            torch_dtype="fake",
            model={"model_id": "fake", "fp8_enabled": fp8},
            lora={"loaded": False},
            supports_negative_prompt=False,
            timings={},
            fp8_enabled=fp8,
        )

    monkeypatch.setattr("runpod_worker.inference.load_pipeline", fake_load_pipeline)

    assert worker.get_loaded(fp8=False).fp8_enabled is False
    assert worker.get_loaded(fp8=False).fp8_enabled is False
    assert worker.get_loaded(fp8=True).fp8_enabled is True

    assert load_calls == [False, True]
