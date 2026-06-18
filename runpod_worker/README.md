# RunPod FLUX Fill Worker

RunPod Serverless worker for `black-forest-labs/FLUX.1-Fill-dev` restoration-oriented
inpainting with the project `partial_noise` schedule control and optional LoRA.

## Build

Build from the repository root so the Docker context includes both `runpod_worker/`
and `src/painting_inpaint/`:

```bash
docker build --platform linux/amd64 -f runpod_worker/Dockerfile -t ghcr.io/<owner>/flux-fill-worker:latest .
docker push ghcr.io/<owner>/flux-fill-worker:latest
```

The image includes code and dependencies, but not the base FLUX weights. The root
`.dockerignore` excludes data, runs, HF caches, secrets, and model weights, while
explicitly allowing `runpod_worker/loras/*.safetensors` so a LoRA can be baked in.

## LoRA

For smoke tests, LoRA is optional. For production, set:

```text
LORA_REQUIRED=1
```

Default baked-in path:

```text
/app/runpod_worker/loras/pytorch_lora_weights.safetensors
```

Before building, copy the selected git-ignored LoRA file to:

```text
runpod_worker/loras/pytorch_lora_weights.safetensors
```

Or set `LORA_PATH` to another mounted/downloaded file path. Never commit the LoRA
unless you intentionally change the repository policy.

## RunPod Endpoint

Recommended first endpoint settings:

- Worker type: Flex
- Min workers: `0`
- Max workers: `1`
- GPU: A100 80GB for correctness first
- Hugging Face cached model: `black-forest-labs/FLUX.1-Fill-dev`
- No network volume initially

Environment variables:

```text
HF_TOKEN=<gated Hugging Face token>
MODEL_ID=black-forest-labs/FLUX.1-Fill-dev
ALLOW_HF_DOWNLOAD=0
LORA_REQUIRED=1
LORA_PATH=/app/runpod_worker/loras/pytorch_lora_weights.safetensors
```

Set `ALLOW_HF_DOWNLOAD=1` only when you explicitly want the worker to download from
Hugging Face if the RunPod cache is missing.

## Request

Mask convention: white = inpaint/edit, black = preserve.

`partial_noise` is optional and defaults to `1.0`. The worker always uses
`FluxFillPartialNoisePipeline`; `partial_noise=1.0` runs the full scheduler, matching
normal full-strength FLUX Fill behavior. Lower values start later in the schedule and
preserve more source structure.

```json
{
  "input": {
    "prompt": "DURER_RESTO",
    "image_url": "https://example.com/input.png",
    "mask_url": "https://example.com/mask.png",
    "partial_noise": 0.65,
    "guidance_scale": 30.0,
    "num_inference_steps": 28,
    "seed": 123,
    "lora_scale": 1.0,
    "output_format": "png"
  }
}
```

For local smoke tests, use `image_base64` and `mask_base64` instead of URLs.

## Response

The first milestone returns a JSON object with:

- `image_base64`: hard-composited PNG/JPEG result
- `model`: cache/download source and dtype metadata
- `lora`: loaded/required path and adapter scaling mode
- `schedule_debug`: selected timestep slice for `partial_noise`
- `timings`: model loading and inference timings

Object storage upload can be added later.

## Local Checks

Import without loading FLUX:

```bash
uv run python -c "import runpod_worker.handler; print('ok')"
```

Run tests:

```bash
uv run pytest
uv run ruff check src/painting_inpaint scripts tests runpod_worker
```

## Safety

- Do not bake `HF_TOKEN` or other secrets into the image.
- Do not bake the FLUX base model or HF cache into the image.
- Do not use a paid network volume for the first milestone unless the cache strategy
  proves insufficient.
- For production, use `LORA_REQUIRED=1` so missing LoRA files fail clearly.
