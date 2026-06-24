"""Invoke the Modal FLUX Fill backend with a JSON payload."""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

import modal

from painting_inpaint_backend.core.progress import sanitize_for_progress
from painting_inpaint_backend.modal.app import FluxFillModalBackend, app


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", type=Path, default=Path("generated/modal_smoke_payload.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    payload = json.loads(args.payload.read_text(encoding="utf-8"))
    started_response_path = args.out_dir / "modal_response.json"
    output_image_path = args.out_dir / "modal_output.png"
    timings_path = args.out_dir / "modal_timings.json"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    with modal.enable_output(), app.run():
        serialized_result = FluxFillModalBackend().restore.remote(payload)

    if not isinstance(serialized_result, str):
        raise TypeError(
            "Modal restore response must be a JSON string, got "
            f"{type(serialized_result).__name__}."
        )
    result = json.loads(serialized_result)

    started_response_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    remote_error = result.get("modal_error")
    if isinstance(remote_error, dict):
        print(f"response: {started_response_path}")
        print(f"remote stage: {remote_error.get('stage', 'unknown')}")
        print(f"remote error: {remote_error.get('error_type')}: {remote_error.get('error')}")
        traceback_text = remote_error.get("traceback")
        if traceback_text:
            print(traceback_text)
        return 1

    output_image_path.write_bytes(base64.b64decode(result["image_base64"]))
    timings_path.write_text(
        json.dumps(sanitize_for_progress(result), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"response: {started_response_path}")
    print(f"image: {output_image_path}")
    print(f"timings: {timings_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
