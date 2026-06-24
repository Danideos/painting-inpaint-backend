# Painting Inpaint Backend

Shared FLUX Fill inference backend for painting restoration. One provider-neutral
`InferenceService` owns model loading, partial-noise scheduling, image validation,
LoRA scaling, hard compositing, progress events, and response construction. Thin
adapters expose that service through RunPod Serverless and Modal.

The base FLUX model, LoRA weights, Hugging Face caches, secrets, and generated runs are
not stored in Git or baked into the Docker image.

## Layout

```text
src/painting_inpaint_backend/core/    shared inference implementation
src/painting_inpaint_backend/runpod/  RunPod handler and streaming adapter
src/painting_inpaint_backend/modal/   Modal app and Volume tooling
scripts/                              Modal smoke and Volume commands
tests/                                core and provider adapter tests
assets/loras/                         ignored optional local LoRA location
```

The provider-neutral entry point is:

```python
from painting_inpaint_backend import InferenceService

output = InferenceService().run(payload)
```

The default input limit is 2,073,600 pixels, which accepts 1440x1440 images and masks.
Deployments may override it with `MAX_IMAGE_PIXELS`.

## Development

Dependencies are defined only in `pyproject.toml` and locked in `uv.lock`.

```powershell
uv sync --all-extras --dev
uv lock --check
uv run pytest
uv run ruff check .
uv run python -c "import painting_inpaint_backend.runpod.handler; print('ok')"
```

## Docker and RunPod

```powershell
docker build --platform linux/amd64 -t ghcr.io/danideos/painting-inpaint-backend:<sha> .
```

The image starts:

```text
python -m painting_inpaint_backend.runpod.handler
```

RunPod keeps its current request and streaming contracts. The top-level job body is
`{"input": <provider-neutral payload>}`. White mask pixels are edited; black pixels are
preserved exactly after hard compositing.

Typical RunPod environment:

```text
MODEL_ID=black-forest-labs/FLUX.1-Fill-dev
ALLOW_HF_DOWNLOAD=0
LORA_REPO_ID=danideos/durer-flux-fill-lora
LORA_FILENAME=pytorch_lora_weights.safetensors
LORA_REVISION=main
LORA_REQUIRED=1
```

`MODEL_PATH` takes precedence when configured and must be an existing local directory;
it is always loaded with `local_files_only=True`. Without `MODEL_PATH`, the backend
preserves the existing RunPod cache discovery and optional Hugging Face download flow.

## Modal

Modal reuses the existing resources:

```text
Volume: flux-fill-models
Secret: huggingface-secret
GPU: L40S
Model: /models/black-forest-labs/FLUX.1-Fill-dev
LoRA: /models/danideos/durer-flux-fill-lora
```

Set the backend image to an immutable 40-character commit tag:

```powershell
$env:PAINTING_INPAINT_BACKEND_IMAGE = `
  "ghcr.io/danideos/painting-inpaint-backend:<full-commit-sha>"

uv run python scripts/inspect_modal_volume.py
uv run python scripts/smoke_modal_registry_image.py
uv run python scripts/create_modal_smoke_payload.py
uv run python scripts/smoke_modal_inference.py `
  --payload generated/modal_smoke_payload.json `
  --out-dir local_runs/modal_smoke
```

Modal loads the same package from the pinned registry image with `include_source=False`.
Inference sets `MODEL_PATH`, `LORA_PATH`, and the Hugging Face offline flags, so request
handling cannot download weights. Responses cross the Modal boundary as JSON strings.

### Modal HTTP API

The colleague-facing client uses an authenticated SSE endpoint while direct Modal
invocation remains available for backend smoke tests. Create a dedicated API secret;
this key grants access only to this restoration endpoint and is not a Modal workspace
credential:

```powershell
$keyBytes = New-Object byte[] 32
[Security.Cryptography.RandomNumberGenerator]::Fill($keyBytes)
$restorationKey = [Convert]::ToBase64String($keyBytes)
uv run modal secret create --force painting-inpaint-restoration-api `
  "RESTORATION_API_KEY=$restorationKey"
```

Deploy with the same immutable image that contains the Modal generator method:

```powershell
$env:PAINTING_INPAINT_BACKEND_IMAGE = `
  "ghcr.io/danideos/painting-inpaint-backend:<full-commit-sha>"
uv run modal deploy -m painting_inpaint_backend.modal.app
```

The ASGI deployment exposes:

```text
POST /v1/restore/stream
Authorization: Bearer <RESTORATION_API_KEY>
Content-Type: application/json
Accept: text/event-stream
```

The request body is the provider-neutral restoration input without a RunPod `input`
wrapper. The response starts with `modal_queued`, streams the same structured model and
inference events as RunPod, and ends with `job_done` containing the normal output.
SSE comments are connection heartbeats and do not represent progress.

## LoRA

RunPod may download the LoRA from Hugging Face or use `LORA_PATH`. Modal points
`LORA_PATH` at its mounted Volume directory. `LORA_REQUIRED=1` makes missing or invalid
weights fail clearly. Per-request `lora_scale` defaults to `1.0` and may be set to zero.

The optional image-local path is:

```text
/app/assets/loras/pytorch_lora_weights.safetensors
```

Safetensors remain excluded from Git and Docker context unless that policy is changed
explicitly.

## Rollback Baselines

- RunPod image:
  `ghcr.io/danideos/flux-fill-worker:7eff1a8e323832a6e5d81c9f98b6d9db80744c4a`
- Modal native implementation commit: `1c2e9bb79c9576de09214726c48ef3266a972997`
- Modal native baseline artifacts: ignored under
  `local_runs/modal_native_baseline/`

Do not update either provider until the new commit-tagged image has passed its provider
smoke test.
