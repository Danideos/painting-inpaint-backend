from __future__ import annotations

from io import BytesIO

import pytest
from PIL import Image

import painting_inpaint_backend.core.image_io as image_io
from painting_inpaint_backend.core.image_io import (
    ImageInputError,
    image_from_url,
    image_to_base64,
    load_request_image,
    normalize_output_format,
)


class _FakeResponse:
    def __init__(self, raw: bytes, *, content_length: int | None = None):
        self._raw = BytesIO(raw)
        self.headers = {}
        if content_length is not None:
            self.headers["Content-Length"] = str(content_length)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self, size: int = -1) -> bytes:
        return self._raw.read(size)


def _png_bytes(size: tuple[int, int] = (2, 2)) -> bytes:
    image = Image.new("RGB", size, "red")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_base64_image_round_trip():
    image = Image.new("RGB", (3, 2), "red")
    encoded = image_to_base64(image, output_format="png")

    decoded = load_request_image(
        {"image_base64": encoded},
        label="image",
        url_key="image_url",
        base64_key="image_base64",
        mode="RGB",
    )

    assert decoded.size == (3, 2)
    assert decoded.mode == "RGB"


def test_base64_data_uri_prefix_is_allowed():
    image = Image.new("RGB", (2, 2), "blue")
    encoded = "data:image/png;base64," + image_to_base64(image, output_format="png")

    decoded = load_request_image(
        {"image_base64": encoded},
        label="image",
        url_key="image_url",
        base64_key="image_base64",
        mode="RGB",
    )

    assert decoded.size == (2, 2)


def test_oversized_base64_is_rejected_before_decoding(monkeypatch):
    monkeypatch.setenv("MAX_BASE64_MB", "1")
    oversized = "A" * (1024 * 1024 + 1)

    with pytest.raises(ImageInputError, match="MAX_BASE64_MB"):
        load_request_image(
            {"image_base64": oversized},
            label="image",
            url_key="image_url",
            base64_key="image_base64",
            mode="RGB",
        )


def test_oversized_image_dimensions_are_rejected(monkeypatch):
    monkeypatch.setenv("MAX_IMAGE_PIXELS", "3")
    image = Image.new("RGB", (2, 2), "red")

    with pytest.raises(ImageInputError, match="2x2"):
        load_request_image(
            {"image_base64": image_to_base64(image)},
            label="image",
            url_key="image_url",
            base64_key="image_base64",
            mode="RGB",
        )


def test_default_limit_accepts_1440_square_image(monkeypatch):
    monkeypatch.delenv("MAX_IMAGE_PIXELS", raising=False)
    image = Image.new("RGB", (1440, 1440), "red")

    decoded = load_request_image(
        {"image_base64": image_to_base64(image)},
        label="image",
        url_key="image_url",
        base64_key="image_base64",
        mode="RGB",
    )

    assert decoded.size == (1440, 1440)


def test_default_limit_rejects_more_than_1440_square_pixels(monkeypatch):
    monkeypatch.delenv("MAX_IMAGE_PIXELS", raising=False)
    image = Image.new("RGB", (1441, 1440), "red")

    with pytest.raises(ImageInputError, match="MAX_IMAGE_PIXELS=2073600"):
        load_request_image(
            {"image_base64": image_to_base64(image)},
            label="image",
            url_key="image_url",
            base64_key="image_base64",
            mode="RGB",
        )


def test_url_content_length_over_limit_is_rejected(monkeypatch):
    monkeypatch.setenv("MAX_DOWNLOAD_MB", "1")

    def fake_urlopen(request, timeout):
        assert timeout == 60
        return _FakeResponse(_png_bytes(), content_length=1024 * 1024 + 1)

    monkeypatch.setattr(image_io, "urlopen", fake_urlopen)

    with pytest.raises(ImageInputError, match="Content-Length"):
        image_from_url("https://example.test/image.png", label="image")


def test_url_download_without_content_length_still_has_hard_limit(monkeypatch):
    monkeypatch.setenv("MAX_DOWNLOAD_MB", "1")

    def fake_urlopen(request, timeout):
        return _FakeResponse(b"x" * (1024 * 1024 + 1))

    monkeypatch.setattr(image_io, "urlopen", fake_urlopen)

    with pytest.raises(ImageInputError, match="download exceeded"):
        image_from_url("https://example.test/image.png", label="image")


def test_url_download_uses_configured_timeout_and_accepts_small_image(monkeypatch):
    monkeypatch.setenv("IMAGE_DOWNLOAD_TIMEOUT_SECONDS", "12.5")
    seen = {}

    def fake_urlopen(request, timeout):
        seen["timeout"] = timeout
        return _FakeResponse(_png_bytes(), content_length=100)

    monkeypatch.setattr(image_io, "urlopen", fake_urlopen)

    decoded = image_from_url("https://example.test/image.png", label="image")

    assert decoded.size == (2, 2)
    assert seen["timeout"] == 12.5


def test_output_format_normalization():
    assert normalize_output_format(None) == "png"
    assert normalize_output_format("jpg") == "jpeg"
    assert normalize_output_format("JPEG") == "jpeg"
