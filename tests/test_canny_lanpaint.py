from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from painting_inpaint_backend.core.canny_lanpaint import (
    CANNY_HIGH_THRESHOLD,
    CANNY_LOW_THRESHOLD,
    LANPAINT_BETA,
    LANPAINT_FINAL_OUTER_STEPS_WITHOUT_INNER,
    LANPAINT_FRICTION,
    LANPAINT_INNER_STEPS,
    LANPAINT_LAMBDA,
    LANPAINT_STEP_SIZE,
    load_control_image,
    make_canny_control,
    mask_edit_to_lanpaint_keep,
    parse_canny_lanpaint_settings,
    reinject_keep_latents,
)
from painting_inpaint_backend.core.image_io import image_to_base64
from painting_inpaint_backend.core.inference import (
    InferenceService,
    WorkerInputError,
    parse_request_settings,
)
from painting_inpaint_backend.core.methods import normalize_method


def test_method_defaults_to_fill_and_accepts_canny():
    assert normalize_method() == "flux_fill"
    assert normalize_method("flux_canny_lanpaint") == "flux_canny_lanpaint"
    with pytest.raises(ValueError, match="method must be one of"):
        normalize_method("unknown")


def test_canny_request_uses_method_specific_guidance_default():
    fill = parse_request_settings({"prompt": "DURER_RESTO"})
    canny = parse_request_settings(
        {"prompt": "DURER_RESTO", "method": "flux_canny_lanpaint"}
    )
    explicit = parse_request_settings(
        {
            "prompt": "DURER_RESTO",
            "method": "flux_canny_lanpaint",
            "guidance_scale": 2.0,
        }
    )

    assert fill.method == "flux_fill"
    assert fill.guidance_scale == 30.0
    assert canny.guidance_scale == 1.5
    assert explicit.guidance_scale == 2.0


def test_fill_service_rejects_canny_method_before_model_loading():
    with pytest.raises(WorkerInputError, match="FLUX Fill service"):
        InferenceService().run(
            {"prompt": "DURER_RESTO", "method": "flux_canny_lanpaint"}
        )


def test_control_image_falls_back_to_input_and_accepts_explicit_base64():
    image = Image.new("RGB", (4, 4), "red")
    fallback, fallback_name = load_control_image({}, fallback=image)
    explicit, explicit_name = load_control_image(
        {"control_image_base64": image_to_base64(Image.new("RGB", (4, 4), "blue"))},
        fallback=image,
    )

    assert fallback_name == "input_image"
    assert fallback.tobytes() == image.tobytes()
    assert explicit_name == "request_control_image"
    assert explicit.getpixel((0, 0)) == (0, 0, 255)


def test_control_image_must_match_restoration_dimensions():
    with pytest.raises(WorkerInputError, match="sizes do not match"):
        load_control_image(
            {
                "control_image_base64": image_to_base64(
                    Image.new("RGB", (2, 2), "blue")
                )
            },
            fallback=Image.new("RGB", (4, 4), "red"),
        )


def test_project_edit_mask_is_inverted_for_lanpaint():
    edit = Image.new("L", (2, 1), 0)
    edit.putpixel((0, 0), 255)

    keep = mask_edit_to_lanpaint_keep(edit)

    assert list(keep.getdata()) == [0, 255]


def test_keep_latent_reinjection_preserves_known_region():
    latents = np.array([10.0, 20.0])
    reference = np.array([1.0, 2.0])
    edit_mask = np.array([1.0, 0.0])

    result = reinject_keep_latents(latents, reference, edit_mask)

    assert result.tolist() == [10.0, 2.0]


def test_canny_generation_uses_validated_thresholds(monkeypatch):
    calls = {}

    def cvt_color(array, code):
        calls["color_code"] = code
        return array[..., 0]

    def canny(array, *, threshold1, threshold2):
        calls["thresholds"] = (threshold1, threshold2)
        return np.where(array > 0, 255, 0).astype(np.uint8)

    monkeypatch.setitem(
        sys.modules,
        "cv2",
        SimpleNamespace(COLOR_RGB2GRAY=7, cvtColor=cvt_color, Canny=canny),
    )

    result = make_canny_control(Image.new("RGB", (3, 2), "white"))

    assert result.mode == "RGB"
    assert result.size == (3, 2)
    assert calls == {
        "color_code": 7,
        "thresholds": (CANNY_LOW_THRESHOLD, CANNY_HIGH_THRESHOLD),
    }


def test_lanpaint_constants_match_validated_configuration():
    assert LANPAINT_INNER_STEPS == 10
    assert LANPAINT_FRICTION == 15.0
    assert LANPAINT_LAMBDA == 10.0
    assert LANPAINT_BETA == 1.0
    assert LANPAINT_STEP_SIZE == 0.1
    assert LANPAINT_FINAL_OUTER_STEPS_WITHOUT_INNER == 3


def test_canny_lanpaint_request_settings_default_to_validated_constants():
    settings = parse_canny_lanpaint_settings({})

    assert settings.canny_low_threshold == CANNY_LOW_THRESHOLD
    assert settings.canny_high_threshold == CANNY_HIGH_THRESHOLD
    assert settings.canny_blur_radius == 0.0
    assert settings.lanpaint_inner_steps == LANPAINT_INNER_STEPS
    assert settings.lanpaint_friction == LANPAINT_FRICTION
    assert settings.lanpaint_lambda == LANPAINT_LAMBDA
    assert settings.lanpaint_beta == LANPAINT_BETA
    assert settings.lanpaint_step_size == LANPAINT_STEP_SIZE
    assert (
        settings.lanpaint_final_outer_steps_without_inner
        == LANPAINT_FINAL_OUTER_STEPS_WITHOUT_INNER
    )


def test_canny_lanpaint_request_settings_accept_overrides():
    settings = parse_canny_lanpaint_settings(
        {
            "canny_low_threshold": "50",
            "canny_high_threshold": "140",
            "canny_blur_radius": "1.5",
            "lanpaint_inner_steps": "8",
            "lanpaint_friction": "12.5",
            "lanpaint_lambda": "9",
            "lanpaint_beta": "0.75",
            "lanpaint_step_size": "0.05",
            "lanpaint_final_outer_steps_without_inner": "2",
        }
    )

    assert settings.canny_low_threshold == 50
    assert settings.canny_high_threshold == 140
    assert settings.canny_blur_radius == 1.5
    assert settings.lanpaint_inner_steps == 8
    assert settings.lanpaint_friction == 12.5
    assert settings.lanpaint_lambda == 9.0
    assert settings.lanpaint_beta == 0.75
    assert settings.lanpaint_step_size == 0.05
    assert settings.lanpaint_final_outer_steps_without_inner == 2


@pytest.mark.parametrize(
    "payload, message",
    [
        ({"canny_low_threshold": -1}, "canny_low_threshold"),
        ({"canny_high_threshold": 256}, "canny_high_threshold"),
        (
            {"canny_low_threshold": 200, "canny_high_threshold": 100},
            "greater than or equal",
        ),
        ({"canny_blur_radius": -0.1}, "canny_blur_radius"),
        ({"lanpaint_inner_steps": -1}, "lanpaint_inner_steps"),
        (
            {"lanpaint_final_outer_steps_without_inner": -1},
            "lanpaint_final_outer_steps_without_inner",
        ),
        ({"lanpaint_friction": -1}, "lanpaint_friction"),
        ({"lanpaint_lambda": -1}, "lanpaint_lambda"),
        ({"lanpaint_beta": -1}, "lanpaint_beta"),
        ({"lanpaint_step_size": 0}, "lanpaint_step_size"),
    ],
)
def test_canny_lanpaint_request_settings_reject_invalid_values(payload, message):
    with pytest.raises(WorkerInputError, match=message):
        parse_canny_lanpaint_settings(payload)
