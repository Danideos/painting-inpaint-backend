"""JSON-safe Modal invocation boundary helpers."""

from __future__ import annotations

import json
import traceback
from typing import Any


def safe_remote_error(exc: Exception, *, stage: str) -> dict[str, Any]:
    return {
        "stage": stage,
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback": traceback.format_exc(limit=20),
    }


def json_response(value: dict[str, Any]) -> str:
    """Cross the Modal boundary using JSON instead of pickle."""

    return json.dumps(value, ensure_ascii=True, separators=(",", ":"))
