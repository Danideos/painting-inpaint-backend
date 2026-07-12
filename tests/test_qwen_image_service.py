from __future__ import annotations

import pytest

from painting_inpaint_backend.core.inference import WorkerInputError
from painting_inpaint_backend.core.qwen_image_service import (
    QWEN_IMAGE_BACKEND_REVISION,
    QWEN_IMAGE_MODEL_ID,
    parse_qwen_image_settings,
)


def test_qwen_image_settings_default_to_native_text_to_image_controls():
    settings = parse_qwen_image_settings(
        {
            "method": "qwen_image",
            "prompt": "restore the damaged regions as a devotional painting",
        }
    )

    assert QWEN_IMAGE_MODEL_ID == "Qwen/Qwen-Image"
    assert QWEN_IMAGE_BACKEND_REVISION == "qwen-image-text-to-image-v1"
    assert settings.method == "qwen_image"
    assert settings.negative_prompt == " "
    assert settings.true_cfg_scale == 4.0
    assert settings.num_inference_steps == 50
    assert settings.width == 1024
    assert settings.height == 1024
    assert settings.max_sequence_length == 512


def test_qwen_image_settings_accept_guidance_scale_as_true_cfg_alias():
    settings = parse_qwen_image_settings(
        {
            "method": "qwen_image",
            "prompt": "painted window",
            "guidance_scale": 3.5,
            "width": 1440,
            "height": 1024,
            "num_inference_steps": 12,
            "seed": "123",
        }
    )

    assert settings.true_cfg_scale == 3.5
    assert settings.width == 1440
    assert settings.height == 1024
    assert settings.num_inference_steps == 12
    assert settings.seed == 123


def test_qwen_image_true_cfg_scale_takes_precedence_over_guidance_alias():
    settings = parse_qwen_image_settings(
        {
            "method": "qwen_image",
            "prompt": "painted window",
            "guidance_scale": 3.5,
            "true_cfg_scale": 5.0,
        }
    )

    assert settings.true_cfg_scale == 5.0


@pytest.mark.parametrize(
    "payload, message",
    [
        ({"method": "qwen_image"}, "prompt is required"),
        ({"method": "qwen_image", "prompt": ""}, "non-empty"),
        ({"method": "qwen_image", "prompt": "x", "width": 1025}, "width"),
        ({"method": "qwen_image", "prompt": "x", "height": 0}, "height"),
        ({"method": "qwen_image", "prompt": "x", "num_inference_steps": 0}, "steps"),
        ({"method": "qwen_image", "prompt": "x", "true_cfg_scale": -1}, "true_cfg"),
    ],
)
def test_qwen_image_settings_reject_invalid_values(payload, message):
    with pytest.raises(WorkerInputError, match=message):
        parse_qwen_image_settings(payload)
