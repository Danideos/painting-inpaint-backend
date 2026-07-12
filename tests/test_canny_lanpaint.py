from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from painting_inpaint_backend.core.canny_fill import (
    CANNY_FILL_DEFAULT_FILL_PARTIAL_NOISE,
    FluxCannyFillService,
    parse_canny_fill_settings,
)
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
    make_lanpaint_source_image,
    make_masked_canny_control,
    mask_edit_to_lanpaint_keep,
    parse_canny_lanpaint_settings,
    reinject_keep_latents,
)
from painting_inpaint_backend.core.image_io import image_from_base64, image_to_base64
from painting_inpaint_backend.core.inference import (
    InferenceService,
    WorkerInputError,
    parse_request_settings,
)
from painting_inpaint_backend.core.methods import normalize_method


def test_method_defaults_to_fill_and_accepts_canny():
    assert normalize_method() == "flux_fill"
    assert normalize_method("flux_canny_lanpaint") == "flux_canny_lanpaint"
    assert normalize_method("flux_canny_lanpaint_native") == "flux_canny_lanpaint_native"
    assert normalize_method("flux_canny_fill") == "flux_canny_fill"
    assert normalize_method("sd15_inpaint") == "sd15_inpaint"
    assert normalize_method("sdxl_inpaint") == "sdxl_inpaint"
    assert normalize_method("qwen_edit") == "qwen_edit"
    assert normalize_method("qwen_image") == "qwen_image"
    assert normalize_method("qwen_image_inpaint") == "qwen_image_inpaint"
    assert normalize_method("qwen_image_lanpaint") == "qwen_image_lanpaint"
    assert normalize_method("sd35_inpaint") == "sd35_inpaint"
    assert normalize_method("sdxl_brushnet") == "sdxl_brushnet"
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
    with pytest.raises(WorkerInputError, match="FLUX Fill service"):
        InferenceService().run({"prompt": "DURER_RESTO", "method": "flux_canny_fill"})


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


def test_masked_canny_control_thins_thick_drawn_lines_inside_mask():
    pytest.importorskip("cv2")
    image = Image.new("RGB", (32, 32), "white")
    mask = Image.new("L", (32, 32), 0)
    for y in range(8, 24):
        for x in range(8, 24):
            mask.putpixel((x, y), 255)
        for dx in range(-3, 4):
            image.putpixel((16 + dx, y), (0, 0, 0))

    control = make_masked_canny_control(
        image,
        mask,
        low_threshold=100,
        high_threshold=200,
        boundary_fill_radius=0,
    )
    edges = np.asarray(control.convert("L"))

    assert np.count_nonzero(edges[8:24, 13:20]) >= 8
    assert all(np.count_nonzero(edges[y, 13:20]) <= 2 for y in range(8, 24))


def test_masked_canny_control_avoids_mask_rectangle_edges():
    pytest.importorskip("cv2")
    image = Image.new("RGB", (32, 32), "gray")
    mask = Image.new("L", (32, 32), 0)
    for y in range(8, 24):
        for x in range(8, 24):
            mask.putpixel((x, y), 255)
            image.putpixel((x, y), (255, 255, 255))

    control = make_masked_canny_control(
        image,
        mask,
        low_threshold=100,
        high_threshold=200,
        boundary_fill_radius=3,
    )
    edges = np.asarray(control.convert("L"))

    border_pixels = [
        *edges[7, 8:24].tolist(),
        *edges[24, 8:24].tolist(),
        *edges[8:24, 7].tolist(),
        *edges[8:24, 24].tolist(),
    ]
    assert max(border_pixels) == 0


def test_lanpaint_source_image_uses_telea_fill_and_preserves_outside():
    pytest.importorskip("cv2")
    image = Image.new("RGB", (32, 32), (64, 64, 64))
    mask = Image.new("L", (32, 32), 0)
    for y in range(8, 24):
        for x in range(8, 24):
            mask.putpixel((x, y), 255)
            image.putpixel((x, y), (255, 255, 255))
    for y in range(10, 22):
        image.putpixel((16, y), (0, 0, 0))

    source = make_lanpaint_source_image(image, mask)
    source_array = np.asarray(source)
    image_array = np.asarray(image)
    mask_array = np.asarray(mask) > 0

    assert np.array_equal(source_array[~mask_array], image_array[~mask_array])
    assert source_array[mask_array].mean() < 128
    assert not np.any(np.all(source_array[mask_array] == [255, 255, 255], axis=1))
    assert not np.any(np.all(source_array[mask_array] == [0, 0, 0], axis=1))


def test_lanpaint_constants_match_validated_configuration():
    assert LANPAINT_INNER_STEPS == 10
    assert LANPAINT_FRICTION == 15.0
    assert LANPAINT_LAMBDA == 10.0
    assert LANPAINT_BETA == 1.0
    assert LANPAINT_STEP_SIZE == 0.2
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


def test_canny_fill_settings_default_to_full_canny_then_partial_fill():
    settings = parse_canny_fill_settings(
        {
            "prompt": "DURER_RESTO",
            "method": "flux_canny_fill",
            "num_inference_steps": 30,
        }
    )

    assert settings.canny_prompt == "DURER_RESTO"
    assert settings.fill_prompt == "DURER_RESTO"
    assert settings.canny_partial_noise == 1.0
    assert settings.fill_partial_noise == CANNY_FILL_DEFAULT_FILL_PARTIAL_NOISE
    assert settings.canny_guidance_scale == 7.0
    assert settings.fill_guidance_scale == 30.0
    assert settings.canny_num_inference_steps == 30
    assert settings.fill_num_inference_steps == 30


def test_canny_fill_accepts_empty_canny_prompt_without_changing_fill_prompt():
    settings = parse_canny_fill_settings(
        {
            "prompt": "DURER_RESTO",
            "method": "flux_canny_fill",
            "canny_prompt": "",
        }
    )

    assert settings.canny_prompt == ""
    assert settings.fill_prompt == "DURER_RESTO"


def test_canny_fill_accepts_stage_specific_overrides():
    settings = parse_canny_fill_settings(
        {
            "prompt": "DURER_RESTO",
            "method": "flux_canny_fill",
            "canny_prompt": "DURER_RESTO canny",
            "fill_prompt": "DURER_RESTO fill",
            "canny_num_inference_steps": "20",
            "fill_num_inference_steps": "12",
            "canny_guidance_scale": "1.25",
            "fill_guidance_scale": "22",
            "canny_lora_scale": "0.8",
            "fill_lora_scale": "0.6",
            "fill_partial_noise": "0.25",
            "canny_low_threshold": "40",
            "canny_high_threshold": "120",
        }
    )

    assert settings.canny_prompt == "DURER_RESTO canny"
    assert settings.fill_prompt == "DURER_RESTO fill"
    assert settings.canny_num_inference_steps == 20
    assert settings.fill_num_inference_steps == 12
    assert settings.canny_guidance_scale == 1.25
    assert settings.fill_guidance_scale == 22.0
    assert settings.canny_lora_scale == 0.8
    assert settings.fill_lora_scale == 0.6
    assert settings.fill_partial_noise == 0.25
    assert settings.canny_low_threshold == 40
    assert settings.canny_high_threshold == 120


class _FakeStageService:
    def __init__(self, color: str, *, method_model: str) -> None:
        self.calls: list[dict] = []
        self.color = color
        self.method_model = method_model

    def run(self, payload, *, reporter=None):
        self.calls.append(dict(payload))
        image = Image.new("RGB", (4, 4), self.color)
        return {
            "image_base64": image_to_base64(image),
            "output_format": payload.get("output_format", "png"),
            "width": 4,
            "height": 4,
            "mask_convention": "white = inpaint/edit, black = preserve",
            "timings": {"stage_seconds": 1.0},
            "gpu_memory": {"peak_allocated_mb": 10.0},
            "model": {"method": self.method_model},
            "lora": {"loaded": True},
            "inference_settings": {
                "control_source": payload.get("control_image_base64") and "request_control_image"
                or "input_image"
            },
            "schedule_debug": {"steps": payload.get("num_inference_steps")},
            "latent_init_debug": {},
            "outside_mask_changed_after_hard_composite": False,
        }


def test_canny_fill_orchestrates_canny_then_fill_without_leaking_intermediate_base64():
    canny = _FakeStageService("blue", method_model="flux_canny_lanpaint")
    fill = _FakeStageService("green", method_model="flux_fill")
    service = FluxCannyFillService(canny_service=canny, fill_service=fill)
    image = Image.new("RGB", (4, 4), "red")
    mask = Image.new("L", (4, 4), 255)

    output = service.run(
        {
            "method": "flux_canny_fill",
            "prompt": "DURER_RESTO",
            "image_base64": image_to_base64(image),
            "mask_base64": image_to_base64(mask),
            "seed": 123,
            "num_inference_steps": 30,
            "fill_partial_noise": 0.25,
            "fill_lora_scale": 0.7,
            "canny_lora_scale": 0.9,
        }
    )

    assert canny.calls[0]["method"] == "flux_canny_lanpaint"
    assert canny.calls[0]["partial_noise"] == 1.0
    assert canny.calls[0]["guidance_scale"] == 7.0
    assert canny.calls[0]["lora_scale"] == 0.9
    assert fill.calls[0]["method"] == "flux_fill"
    assert fill.calls[0]["partial_noise"] == 0.25
    assert fill.calls[0]["guidance_scale"] == 30.0
    assert fill.calls[0]["lora_scale"] == 0.7
    assert fill.calls[0]["image_base64"] != canny.calls[0]["image_base64"]
    assert output["model"]["method"] == "flux_canny_fill"
    assert output["inference_settings"]["method"] == "flux_canny_fill"
    assert output["inference_settings"]["canny"]["lora_scale"] == 0.9
    assert output["inference_settings"]["fill"]["partial_noise"] == 0.25
    assert output["canny_image_base64"]
    assert (
        image_from_base64(output["canny_image_base64"], label="canny output").getpixel(
            (0, 0)
        )
        == (0, 0, 255)
    )
    assert output["canny_output_format"] == "png"
    assert "image_base64" not in output["hybrid_intermediate"]
