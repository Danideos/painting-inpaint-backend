from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from painting_inpaint_backend.core.inference import WorkerInputError
from painting_inpaint_backend.core.qwen_edit_service import (
    QWEN_LANPAINT_BACKEND_REVISION,
    QWEN_LANPAINT_BETA,
    QWEN_LANPAINT_FINAL_OUTER_STEPS_WITHOUT_INNER,
    QWEN_LANPAINT_FRICTION,
    QWEN_LANPAINT_INNER_STEPS,
    QWEN_LANPAINT_LAMBDA,
    QWEN_LANPAINT_STEP_SIZE,
    QWEN_MASKED_REGION_FILL_RGB,
    make_qwen_source_image,
    parse_qwen_edit_settings,
    parse_qwen_native_pipeline_flag,
)


def test_qwen_source_image_hides_only_masked_pixels():
    image = Image.fromarray(
        np.array(
            [
                [[10, 20, 30], [40, 50, 60]],
                [[70, 80, 90], [100, 110, 120]],
            ],
            dtype=np.uint8,
        )
    ).convert("RGB")
    mask = Image.fromarray(
        np.array(
            [
                [0, 255],
                [127, 128],
            ],
            dtype=np.uint8,
        )
    ).convert("L")

    source = make_qwen_source_image(image, mask)

    assert np.asarray(source).tolist() == [
        [[10, 20, 30], list(QWEN_MASKED_REGION_FILL_RGB)],
        [[70, 80, 90], list(QWEN_MASKED_REGION_FILL_RGB)],
    ]


def test_qwen_settings_default_to_lanpaint_route_controls():
    settings = parse_qwen_edit_settings({"method": "qwen_edit"})

    assert QWEN_LANPAINT_BACKEND_REVISION == "qwen-image-edit-lanpaint-v1"
    assert settings.lanpaint_inner_steps == QWEN_LANPAINT_INNER_STEPS
    assert settings.lanpaint_friction == QWEN_LANPAINT_FRICTION
    assert settings.lanpaint_lambda == QWEN_LANPAINT_LAMBDA
    assert settings.lanpaint_beta == QWEN_LANPAINT_BETA
    assert settings.lanpaint_step_size == QWEN_LANPAINT_STEP_SIZE
    assert (
        settings.lanpaint_final_outer_steps_without_inner
        == QWEN_LANPAINT_FINAL_OUTER_STEPS_WITHOUT_INNER
    )


def test_qwen_lanpaint_rejects_native_padding_mask_crop():
    with pytest.raises(WorkerInputError, match="padding_mask_crop is not supported"):
        parse_qwen_edit_settings({"method": "qwen_edit", "padding_mask_crop": 16})


def test_qwen_native_pipeline_flag_is_explicit_and_boolean_only():
    assert parse_qwen_native_pipeline_flag({}) is False
    assert parse_qwen_native_pipeline_flag({"qwen_native_pipeline": True}) is True

    with pytest.raises(WorkerInputError, match="qwen_native_pipeline must be a boolean"):
        parse_qwen_native_pipeline_flag({"qwen_native_pipeline": "true"})
