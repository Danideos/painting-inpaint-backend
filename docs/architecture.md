# Backend Architecture

`painting-inpaint-backend` owns all server-side restoration behavior. The colleague-facing
`painting-inpaint-remote-client` remains a separate lightweight package and is not part of
this repository.

## Shared Core

The core contains one service per restoration method: the existing `InferenceService`
for FLUX Fill, `FluxCannyLanPaintService` for direct FLUX-Canny validation, and
`FluxCannyFillService` for the client-facing hybrid method that runs Canny/LanPaint
first and then refines with FLUX Fill. They share the same logical request, progress,
hard-composite, and response contracts.

The core owns request validation, image limits, model and LoRA loading, partial-noise
scheduling, inference callbacks, hard compositing, progress metadata, and response image
encoding. Provider modules must not copy those responsibilities.

## RunPod Adapter

The RunPod adapter unwraps `job["input"]`, emits the existing streaming event sequence,
and delegates to `InferenceService`. Its executable module is:

```text
python -m painting_inpaint_backend.runpod.handler
```

## Modal Adapter

The Modal adapter exposes separate zero-warm GPU classes for Fill, direct Canny, and the
hybrid Canny-to-Fill path, then routes them behind one authenticated HTTP endpoint. The
hybrid class mounts both the Fill and Canny Volumes under separate paths so no model
weights are baked into the image and no Hugging Face downloads happen during inference.
All classes use immutable GHCR images with `include_source=False`, explicit local model
paths, and offline Hugging Face settings. Results and errors cross the Modal boundary as
JSON strings.

The GPU class exposes both direct `restore` invocation for backend smoke tests and a
streaming generator used by the HTTP gateway. A small ASGI function authenticates a
dedicated bearer key, emits an immediate queued event, keeps the connection alive with
SSE comment heartbeats, and forwards shared progress events from the GPU method. It has
no model weights, GPU, warm-container setting, or duplicated inference logic.

Fill uses `PAINTING_INPAINT_BACKEND_IMAGE`. Canny uses
`PAINTING_INPAINT_CANNY_IMAGE`. Both references must use immutable commit tags or
digests.

## Dependency Ownership

`pyproject.toml` is the only dependency declaration and `uv.lock` is its resolved lock.
The Fill image exports `inference` and `runpod`. The Canny image exports `inference`
and the pinned GPLv3 `canny` extra. Modal local tooling uses `modal`. There is no
standalone `requirements.txt`.
