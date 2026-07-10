from __future__ import annotations

import numpy as np
from PIL import Image

from painting_inpaint_backend.core.qwen_edit_service import (
    QWEN_MASKED_REGION_FILL_RGB,
    make_qwen_source_image,
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
