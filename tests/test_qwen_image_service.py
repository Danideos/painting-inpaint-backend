from __future__ import annotations

import math

import pytest

from painting_inpaint_backend.core.inference import WorkerInputError
from painting_inpaint_backend.core.qwen_image_service import (
    QWEN_IMAGE_BACKEND_REVISION,
    QWEN_IMAGE_INPAINT_BACKEND_REVISION,
    QWEN_IMAGE_LANPAINT_BACKEND_REVISION,
    QWEN_IMAGE_MODEL_ID,
    parse_qwen_image_inpaint_settings,
    parse_qwen_image_lanpaint_settings,
    parse_qwen_image_settings,
    resolve_qwen_sampling_shift,
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


def test_qwen_image_inpaint_settings_default_to_native_inpaint_controls():
    settings = parse_qwen_image_inpaint_settings({"method": "qwen_image_inpaint"})

    assert QWEN_IMAGE_INPAINT_BACKEND_REVISION == "qwen-image-native-inpaint-v1"
    assert settings.method == "qwen_image_inpaint"
    assert settings.prompt == ""
    assert settings.negative_prompt == " "
    assert settings.true_cfg_scale == 4.0
    assert settings.num_inference_steps == 50
    assert settings.strength == 1.0
    assert settings.max_sequence_length == 512
    assert settings.padding_mask_crop is None
    assert settings.qwen_source_strategy == "telea"


def test_qwen_image_inpaint_settings_accept_native_controls():
    settings = parse_qwen_image_inpaint_settings(
        {
            "method": "qwen_image_inpaint",
            "prompt": "restore the painted folds",
            "negative_prompt": " ",
            "guidance_scale": 3.5,
            "strength": 0.65,
            "num_inference_steps": 12,
            "seed": "123",
            "padding_mask_crop": "32",
            "qwen_source_strategy": "original",
        }
    )

    assert settings.prompt == "restore the painted folds"
    assert settings.true_cfg_scale == 3.5
    assert settings.strength == 0.65
    assert settings.num_inference_steps == 12
    assert settings.seed == 123
    assert settings.padding_mask_crop == 32
    assert settings.qwen_source_strategy == "original"


def test_qwen_image_lanpaint_settings_default_to_dynamic_sampling_shift():
    settings = parse_qwen_image_lanpaint_settings({"method": "qwen_image_lanpaint"})

    assert QWEN_IMAGE_LANPAINT_BACKEND_REVISION == "qwen-image-lanpaint-sampling-shift-v2"
    assert settings.method == "qwen_image_lanpaint"
    assert settings.prompt == ""
    assert settings.negative_prompt == " "
    assert settings.true_cfg_scale == 4.0
    assert settings.num_inference_steps == 50
    assert settings.max_sequence_length == 512
    assert settings.qwen_source_strategy == "telea"
    assert settings.qwen_sampling_shift is None


def test_qwen_image_lanpaint_settings_accept_fixed_sampling_shift():
    settings = parse_qwen_image_lanpaint_settings(
        {
            "method": "qwen_image_lanpaint",
            "qwen_sampling_shift": "3.5",
        }
    )

    assert settings.qwen_sampling_shift == 3.5


def test_qwen_sampling_shift_dynamic_mode_preserves_calculated_mu():
    metadata = resolve_qwen_sampling_shift(
        image_seq_len=1024,
        scheduler_config={
            "base_image_seq_len": 256,
            "max_image_seq_len": 4096,
            "base_shift": 0.5,
            "max_shift": 1.15,
        },
        calculate_shift=lambda *args: 1.234,
        qwen_sampling_shift=None,
    )

    assert metadata["mu"] == 1.234
    assert metadata["qwen_sampling_shift_requested"] is None
    assert metadata["qwen_sampling_shift_mode"] == "dynamic"
    assert metadata["qwen_sampling_shift_effective"] == pytest.approx(math.exp(1.234))


def test_qwen_sampling_shift_fixed_mode_uses_log_shift():
    metadata = resolve_qwen_sampling_shift(
        image_seq_len=1024,
        scheduler_config={},
        calculate_shift=lambda *args: (_ for _ in ()).throw(AssertionError),
        qwen_sampling_shift=3.5,
    )

    assert metadata["mu"] == pytest.approx(math.log(3.5))
    assert metadata["qwen_sampling_shift_requested"] == 3.5
    assert metadata["qwen_sampling_shift_effective"] == pytest.approx(3.5)
    assert metadata["qwen_sampling_shift_mode"] == "fixed"


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), "-inf"])
def test_qwen_image_lanpaint_settings_reject_invalid_sampling_shift(value):
    with pytest.raises(WorkerInputError, match="qwen_sampling_shift"):
        parse_qwen_image_lanpaint_settings(
            {
                "method": "qwen_image_lanpaint",
                "qwen_sampling_shift": value,
            }
        )


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


@pytest.mark.parametrize(
    "payload, message",
    [
        ({"method": "qwen_image_inpaint", "prompt": 123}, "prompt"),
        ({"method": "qwen_image_inpaint", "strength": -0.1}, "strength"),
        ({"method": "qwen_image_inpaint", "strength": 1.1}, "strength"),
        ({"method": "qwen_image_inpaint", "num_inference_steps": 0}, "steps"),
        ({"method": "qwen_image_inpaint", "padding_mask_crop": -1}, "padding_mask_crop"),
    ],
)
def test_qwen_image_inpaint_settings_reject_invalid_values(payload, message):
    with pytest.raises(WorkerInputError, match=message):
        parse_qwen_image_inpaint_settings(payload)
