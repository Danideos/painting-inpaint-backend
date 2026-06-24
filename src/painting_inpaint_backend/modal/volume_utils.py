"""Pure helpers for validating Modal Volume snapshots."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

READY_MARKER_NAME = ".painting-inpaint-volume-ready.json"


def dir_size(path: Path) -> tuple[int, int]:
    files = 0
    total = 0
    if not path.exists():
        return files, total
    for item in path.rglob("*"):
        if item.is_file():
            files += 1
            total += item.stat().st_size
    return files, total


def snapshot_summary(path: Path) -> dict[str, Any]:
    files, bytes_total = dir_size(path)
    return {
        "path": str(path),
        "exists": path.exists(),
        "ready": (path / READY_MARKER_NAME).exists(),
        "files": files,
        "bytes": bytes_total,
        "gib": round(bytes_total / (1024**3), 3),
    }


def write_ready_marker(path: Path, *, repo_id: str) -> None:
    marker = path / READY_MARKER_NAME
    marker.write_text(
        json.dumps({"repo_id": repo_id, "complete": True}, sort_keys=True),
        encoding="utf-8",
    )
