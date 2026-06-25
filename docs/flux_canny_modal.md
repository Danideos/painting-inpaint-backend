# FLUX-Canny/LanPaint Modal Rollout

This method is Modal-only in its first release. RunPod remains FLUX Fill only, and the
remote client is not changed until the staged backend passes acceptance.

## Selected Artifacts

- Base model: `black-forest-labs/FLUX.1-Canny-dev`
- LoRA: rank-64 final from training run `20260606_212825_..._rank64`
- LoRA repository: private `danideos/durer-flux-canny-lora`
- LanPaint: 1.5.3 at commit `ba4bb687f1e48dc1f1bb2e8d7c90516bba4e2a3c`
- Modal Volume: `flux-canny-models`
- Public image: `ghcr.io/danideos/painting-inpaint-backend-canny:<sha>`

The Canny image contains GPLv3 LanPaint; keep the licensing note in
`THIRD_PARTY_NOTICES.md` current when changing that dependency.

## Prepare and Publish the LoRA

```powershell
uv run --extra inference python scripts/prepare_flux_canny_lora.py `
  --archive "C:\dev\kaiser-daniel\runs\metacentrum\flux_canny\flux_canny_rank16_32_64_full_runs.tgz"

$env:HF_TOKEN = "<read-write token>"
uv run python scripts/publish_flux_canny_lora.py
Remove-Item Env:HF_TOKEN
```

Record the returned immutable Hugging Face commit SHA.

## Populate the Canny Volume

```powershell
$env:PAINTING_INPAINT_BACKEND_IMAGE = `
  "ghcr.io/danideos/painting-inpaint-backend:<fill-commit-sha>"
$env:CANNY_MODEL_REVISION = "<40-character model SHA>"
$env:CANNY_LORA_REVISION = "<40-character LoRA SHA>"

uv run python scripts/populate_modal_canny_volume.py
uv run python scripts/inspect_modal_canny_volume.py
```

Both snapshots must report `ready=true` before inference deployment.

## Staging Deployment

```powershell
$env:MODAL_APP_NAME = "painting-inpaint-backend-modal-staging"
$env:PAINTING_INPAINT_BACKEND_IMAGE = `
  "ghcr.io/danideos/painting-inpaint-backend:<fill-commit-sha>"
$env:PAINTING_INPAINT_CANNY_IMAGE = `
  "ghcr.io/danideos/painting-inpaint-backend-canny:<canny-commit-sha>"

uv run modal deploy -m painting_inpaint_backend.modal.app
```

Create a Canny smoke payload and invoke the direct streaming method:

```powershell
uv run python scripts/create_modal_smoke_payload.py `
  --method flux_canny_lanpaint `
  --include-control-image `
  --out generated/modal_canny_smoke_payload.json

uv run python scripts/smoke_modal_inference.py `
  --payload generated/modal_canny_smoke_payload.json `
  --out-dir local_runs/modal_canny_512 `
  --stream
```

Do not deploy the production app until generated 512, Rosary 1024, and 1440 memory
smokes pass with live progress, local-only model loading, rank-64 LoRA metadata, and
exact outside-mask preservation.
