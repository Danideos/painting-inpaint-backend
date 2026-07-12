import pytest

from painting_inpaint_backend.core.inference import WorkerInputError
from painting_inpaint_backend.core.sd3_lanpaint_service import (
    SD3_LANPAINT_BACKEND_REVISION,
    parse_sd3_lanpaint_settings,
)


def test_sd3_lanpaint_settings_default_to_experiment_controls():
    settings = parse_sd3_lanpaint_settings({"method": "sd3_lanpaint"})

    assert SD3_LANPAINT_BACKEND_REVISION == "sd3-medium-lanpaint-v1"
    assert settings.method == "sd3_lanpaint"
    assert settings.prompt == ""
    assert settings.negative_prompt == ""
    assert settings.guidance_scale == 5.0
    assert settings.num_inference_steps == 50
    assert settings.partial_noise == 1.0
    assert settings.max_sequence_length == 256
    assert settings.sd3_source_strategy == "telea"
    assert settings.lanpaint_inner_steps == 10
    assert settings.lanpaint_friction == 15.0
    assert settings.lanpaint_lambda == 10.0
    assert settings.lanpaint_beta == 1.0
    assert settings.lanpaint_step_size == 0.2
    assert settings.lanpaint_final_outer_steps_without_inner == 3


def test_sd3_lanpaint_settings_accept_overrides():
    settings = parse_sd3_lanpaint_settings(
        {
            "method": "sd3_lanpaint",
            "prompt": "restore",
            "negative_prompt": "blur",
            "guidance_scale": "3.5",
            "num_inference_steps": "28",
            "partial_noise": "0.5",
            "max_sequence_length": "128",
            "seed": "123",
            "sd3_source_strategy": "original",
            "lanpaint_inner_steps": "2",
            "lanpaint_friction": "12.5",
            "lanpaint_lambda": "8",
            "lanpaint_beta": "2",
            "lanpaint_step_size": "0.5",
            "lanpaint_final_outer_steps_without_inner": "1",
        }
    )

    assert settings.prompt == "restore"
    assert settings.negative_prompt == "blur"
    assert settings.guidance_scale == 3.5
    assert settings.num_inference_steps == 28
    assert settings.partial_noise == 0.5
    assert settings.max_sequence_length == 128
    assert settings.seed == 123
    assert settings.sd3_source_strategy == "original"
    assert settings.lanpaint_inner_steps == 2
    assert settings.lanpaint_friction == 12.5
    assert settings.lanpaint_lambda == 8.0
    assert settings.lanpaint_beta == 2.0
    assert settings.lanpaint_step_size == 0.5
    assert settings.lanpaint_final_outer_steps_without_inner == 1


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"method": "sd35_inpaint"}, "only accepts"),
        ({"method": "sd3_lanpaint", "prompt": 123}, "prompt"),
        ({"method": "sd3_lanpaint", "guidance_scale": -1}, "guidance_scale"),
        ({"method": "sd3_lanpaint", "num_inference_steps": 0}, "steps"),
        ({"method": "sd3_lanpaint", "partial_noise": 0}, "partial_noise"),
        ({"method": "sd3_lanpaint", "max_sequence_length": 0}, "max_sequence_length"),
        ({"method": "sd3_lanpaint", "sd3_source_strategy": "white"}, "source_strategy"),
        ({"method": "sd3_lanpaint", "lanpaint_inner_steps": -1}, "inner_steps"),
        ({"method": "sd3_lanpaint", "lanpaint_friction": -1}, "friction"),
        ({"method": "sd3_lanpaint", "lanpaint_lambda": -1}, "lambda"),
        ({"method": "sd3_lanpaint", "lanpaint_beta": -1}, "beta"),
        ({"method": "sd3_lanpaint", "lanpaint_step_size": 0}, "step_size"),
    ],
)
def test_sd3_lanpaint_settings_reject_invalid_values(payload, message):
    with pytest.raises(WorkerInputError, match=message):
        parse_sd3_lanpaint_settings(payload)
