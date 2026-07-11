import pytest

from painting_inpaint_backend.core.inference import WorkerInputError
from painting_inpaint_backend.core.sdxl_brushnet_service import (
    parse_sdxl_brushnet_settings,
)


def test_brushnet_guess_mode_defaults_to_false():
    settings = parse_sdxl_brushnet_settings(
        {"method": "sdxl_brushnet", "prompt": ""}
    )

    assert settings.guess_mode is False


def test_brushnet_guess_mode_accepts_boolean_flags():
    enabled = parse_sdxl_brushnet_settings(
        {"method": "sdxl_brushnet", "prompt": "", "guess_mode": True}
    )
    disabled = parse_sdxl_brushnet_settings(
        {"method": "sdxl_brushnet", "prompt": "", "guess_mode": False}
    )

    assert enabled.guess_mode is True
    assert disabled.guess_mode is False


def test_brushnet_guess_mode_rejects_invalid_values():
    with pytest.raises(WorkerInputError, match="guess_mode must be a boolean"):
        parse_sdxl_brushnet_settings(
            {"method": "sdxl_brushnet", "prompt": "", "guess_mode": "sometimes"}
        )
