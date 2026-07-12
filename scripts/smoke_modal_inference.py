"""Invoke a Modal restoration backend with a JSON payload."""

from __future__ import annotations

import argparse
import base64
import json
import sys
import uuid
from pathlib import Path

import modal

from painting_inpaint_backend.core.methods import (
    FLUX_CANNY_FILL_METHOD,
    FLUX_CANNY_LANPAINT_METHOD,
    QWEN_EDIT_METHOD,
    QWEN_IMAGE_LANPAINT_METHOD,
    QWEN_IMAGE_METHOD,
    SD15_INPAINT_METHOD,
    SD35_INPAINT_METHOD,
    SDXL_BRUSHNET_METHOD,
    SDXL_INPAINT_METHOD,
    normalize_method,
)
from painting_inpaint_backend.core.progress import sanitize_for_progress
from painting_inpaint_backend.modal.app import (
    FluxCannyFillModalBackend,
    FluxCannyLanPaintModalBackend,
    FluxFillModalBackend,
    QwenEditModalBackend,
    QwenImageModalBackend,
    SD15ModalBackend,
    SD35ModalBackend,
    SDXLBrushNetModalBackend,
    SDXLModalBackend,
    app,
)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", type=Path, default=Path("generated/modal_smoke_payload.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--stream", action="store_true")
    args = parser.parse_args()

    payload = json.loads(args.payload.read_text(encoding="utf-8"))
    started_response_path = args.out_dir / "modal_response.json"
    output_image_path = args.out_dir / "modal_output.png"
    timings_path = args.out_dir / "modal_timings.json"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    method = normalize_method(payload.get("method"))
    if method == FLUX_CANNY_FILL_METHOD:
        backend = FluxCannyFillModalBackend()
    elif method == FLUX_CANNY_LANPAINT_METHOD:
        backend = FluxCannyLanPaintModalBackend()
    elif method == SD15_INPAINT_METHOD:
        backend = SD15ModalBackend()
    elif method == SDXL_INPAINT_METHOD:
        backend = SDXLModalBackend()
    elif method == QWEN_EDIT_METHOD:
        backend = QwenEditModalBackend()
    elif method == QWEN_IMAGE_METHOD:
        backend = QwenImageModalBackend()
    elif method == QWEN_IMAGE_LANPAINT_METHOD:
        backend = QwenImageModalBackend()
    elif method == SD35_INPAINT_METHOD:
        backend = SD35ModalBackend()
    elif method == SDXL_BRUSHNET_METHOD:
        backend = SDXLBrushNetModalBackend()
    else:
        backend = FluxFillModalBackend()

    with modal.enable_output(), app.run():
        if args.stream:
            events_path = args.out_dir / "modal_events.jsonl"
            final_event = None
            with events_path.open("w", encoding="utf-8") as events_file:
                for serialized_event in backend.restore_stream.remote_gen(
                    payload,
                    uuid.uuid4().hex,
                ):
                    event = json.loads(serialized_event)
                    events_file.write(json.dumps(event, sort_keys=True) + "\n")
                    print(event.get("message", event.get("event", "progress")), flush=True)
                    if event.get("type") == "error":
                        raise RuntimeError(event.get("message", "Modal restoration failed."))
                    if event.get("type") == "final":
                        final_event = event
            if final_event is None:
                raise RuntimeError("Modal stream closed without a final event.")
            result = final_event["output"]
            serialized_result = json.dumps(result)
        else:
            serialized_result = backend.restore.remote(payload)

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
