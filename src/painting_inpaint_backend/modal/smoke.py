"""Tiny Modal CPU smoke app for authentication/deploy verification."""

from __future__ import annotations

import time

import modal

app = modal.App("painting-inpaint-backend-modal-smoke")


@app.function()
def ping() -> dict[str, float | str]:
    started = time.perf_counter()
    return {"message": "ok", "elapsed_seconds": time.perf_counter() - started}


@app.local_entrypoint()
def main() -> None:
    started = time.perf_counter()
    result = ping.remote()
    print({"result": result, "observed_seconds": time.perf_counter() - started})
