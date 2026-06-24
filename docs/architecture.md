# Backend Architecture

`painting-inpaint-backend` owns all server-side restoration behavior. The colleague-facing
`painting-inpaint-remote-client` remains a separate lightweight package and is not part of
this repository.

## Shared Core

`painting_inpaint_backend.core.InferenceService` is the only inference implementation.
It accepts the provider-neutral payload described in `client_contract.md` and returns the
same logical response for every provider.

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

The Modal adapter uses an immutable GHCR backend image with `include_source=False`.
It mounts `flux-fill-models` at `/models`, sets `MODEL_PATH` and `LORA_PATH`, enables
offline Hugging Face modes, and delegates to the same `InferenceService`. Results and
errors cross the Modal boundary as JSON strings to avoid local pickle dependencies.

The GPU class exposes both direct `restore` invocation for backend smoke tests and a
streaming generator used by the HTTP gateway. A small ASGI function authenticates a
dedicated bearer key, emits an immediate queued event, keeps the connection alive with
blank NDJSON heartbeats, and forwards shared progress events from the GPU method. It has
no model weights, GPU, warm-container setting, or duplicated inference logic.

The image reference must be supplied through `PAINTING_INPAINT_BACKEND_IMAGE` as a full
commit tag or digest. Mutable `latest` tags are rejected.

## Dependency Ownership

`pyproject.toml` is the only dependency declaration and `uv.lock` is its resolved lock.
The Docker image exports its `inference` and `runpod` extras from that lock. Modal local
tooling uses the `modal` extra. There is no standalone `requirements.txt`.
