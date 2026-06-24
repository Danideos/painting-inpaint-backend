"""Image loading and encoding helpers for RunPod request payloads."""

from __future__ import annotations

import base64
import binascii
import os
from io import BytesIO
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

from PIL import Image

DEFAULT_MAX_IMAGE_PIXELS = 2_073_600
DEFAULT_MAX_BASE64_MB = 20
DEFAULT_MAX_DOWNLOAD_MB = 25
DEFAULT_IMAGE_DOWNLOAD_TIMEOUT_SECONDS = 60
BYTES_PER_MB = 1024 * 1024
DOWNLOAD_CHUNK_SIZE = 64 * 1024


class ImageInputError(ValueError):
    """Raised when an image input field is missing or invalid."""


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ImageInputError(f"{name} must be an integer.") from exc
    if parsed <= 0:
        raise ImageInputError(f"{name} must be positive.")
    return parsed


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ImageInputError(f"{name} must be a number.") from exc
    if parsed <= 0:
        raise ImageInputError(f"{name} must be positive.")
    return parsed


def max_image_pixels() -> int:
    """Return the configured maximum image pixel count."""

    return _env_int("MAX_IMAGE_PIXELS", DEFAULT_MAX_IMAGE_PIXELS)


def max_base64_bytes() -> int:
    """Return the configured maximum base64 input string size in bytes."""

    return _env_int("MAX_BASE64_MB", DEFAULT_MAX_BASE64_MB) * BYTES_PER_MB


def max_download_bytes() -> int:
    """Return the configured maximum URL download size in bytes."""

    return _env_int("MAX_DOWNLOAD_MB", DEFAULT_MAX_DOWNLOAD_MB) * BYTES_PER_MB


def image_download_timeout_seconds() -> float:
    """Return the configured image download timeout."""

    return _env_float(
        "IMAGE_DOWNLOAD_TIMEOUT_SECONDS",
        float(DEFAULT_IMAGE_DOWNLOAD_TIMEOUT_SECONDS),
    )


def _strip_data_uri(value: str) -> str:
    if value.startswith("data:"):
        if "," not in value:
            raise ImageInputError("Invalid data URI image input.")
        return value.split(",", 1)[1]
    return value


def _check_base64_size(encoded: str, *, label: str) -> None:
    size = len(encoded.encode("utf-8"))
    limit = max_base64_bytes()
    if size > limit:
        limit_mb = limit / BYTES_PER_MB
        size_mb = size / BYTES_PER_MB
        raise ImageInputError(
            f"{label} base64 input is too large: {size_mb:.2f} MiB exceeds "
            f"MAX_BASE64_MB={limit_mb:.0f} MiB."
        )


def _check_image_pixels(image: Image.Image, *, label: str) -> None:
    width, height = image.size
    pixels = int(width) * int(height)
    limit = max_image_pixels()
    if pixels > limit:
        raise ImageInputError(
            f"{label} image dimensions are too large: {width}x{height} "
            f"({pixels} pixels) exceeds MAX_IMAGE_PIXELS={limit}."
        )


def _open_checked_image(raw: bytes, *, label: str) -> Image.Image:
    try:
        with Image.open(BytesIO(raw)) as image:
            _check_image_pixels(image, label=label)
            return image.convert("RGB").copy()
    except ImageInputError:
        raise
    except Exception as exc:
        raise ImageInputError(f"{label} input is not a readable image.") from exc


def image_from_base64(value: str, *, label: str) -> Image.Image:
    """Decode a base64 image payload into a detached PIL image."""

    if not isinstance(value, str) or not value.strip():
        raise ImageInputError(f"{label} base64 input must be a non-empty string.")
    encoded = _strip_data_uri(value.strip())
    _check_base64_size(encoded, label=label)
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ImageInputError(f"{label} base64 input is not valid base64.") from exc
    return _open_checked_image(raw, label=label)


def _content_length(response: Any) -> int | None:
    headers = getattr(response, "headers", None)
    value = None
    if headers is not None:
        value = headers.get("Content-Length")
    if value is None and hasattr(response, "getheader"):
        value = response.getheader("Content-Length")
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _read_limited_response(response: Any, *, label: str, limit: int) -> bytes:
    content_length = _content_length(response)
    if content_length is not None and content_length > limit:
        raise ImageInputError(
            f"{label} download is too large: Content-Length {content_length} bytes "
            f"exceeds MAX_DOWNLOAD_MB={limit / BYTES_PER_MB:.0f} MiB."
        )

    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = response.read(min(DOWNLOAD_CHUNK_SIZE, limit - total + 1))
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
        if total > limit:
            raise ImageInputError(
                f"{label} download exceeded MAX_DOWNLOAD_MB="
                f"{limit / BYTES_PER_MB:.0f} MiB."
            )
    return b"".join(chunks)


def image_from_url(value: str, *, label: str, timeout: float | None = None) -> Image.Image:
    """Download an image URL into a detached PIL image."""

    if not isinstance(value, str) or not value.strip():
        raise ImageInputError(f"{label} URL input must be a non-empty string.")
    request = Request(value.strip(), headers={"User-Agent": "painting-inpaint-backend/1.0"})
    timeout = image_download_timeout_seconds() if timeout is None else timeout
    limit = max_download_bytes()
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = _read_limited_response(response, label=label, limit=limit)
    except (OSError, URLError) as exc:
        raise ImageInputError(f"Could not download {label} from URL: {exc}") from exc
    return _open_checked_image(raw, label=label)


def load_request_image(
    payload: dict[str, Any],
    *,
    label: str,
    url_key: str,
    base64_key: str,
    mode: str,
) -> Image.Image:
    """Load one required image from either URL or base64 request fields."""

    url_value = payload.get(url_key)
    base64_value = payload.get(base64_key)
    if url_value and base64_value:
        raise ImageInputError(f"Provide only one of {url_key} or {base64_key}.")
    if base64_value:
        image = image_from_base64(str(base64_value), label=label)
    elif url_value:
        image = image_from_url(str(url_value), label=label)
    else:
        raise ImageInputError(f"Missing {label}; provide {url_key} or {base64_key}.")
    return image.convert(mode)


def normalize_output_format(value: Any) -> str:
    """Normalize and validate the requested response image format."""

    if value is None:
        return "png"
    normalized = str(value).strip().lower()
    if normalized == "jpg":
        normalized = "jpeg"
    if normalized not in {"png", "jpeg"}:
        raise ValueError("output_format must be one of: png, jpeg.")
    return normalized


def image_to_base64(image: Image.Image, *, output_format: str = "png") -> str:
    """Encode a PIL image as a base64 string without a data URI prefix."""

    normalized = normalize_output_format(output_format)
    buffer = BytesIO()
    save_kwargs: dict[str, Any] = {}
    if normalized == "jpeg":
        image = image.convert("RGB")
        save_kwargs["quality"] = 95
    image.save(buffer, format=normalized.upper(), **save_kwargs)
    return base64.b64encode(buffer.getvalue()).decode("ascii")
