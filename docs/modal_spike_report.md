# Modal Spike Report

## Current Status

The standalone spike proved authentication, Volume population, and L40S FLUX Fill
inference. Its useful code is now consolidated into `painting-inpaint-backend`. The
first successful 512x512 native-spike run
loaded all weights from the Modal Volume, used no request-time Hugging Face download,
loaded the LoRA, completed eight inference steps, returned an image, and preserved
all unmasked pixels.

## Commands To Record

```powershell
uv sync --all-extras --dev
uv run modal token new
uv run python scripts/smoke_modal_cpu.py
uv run modal secret create huggingface-secret HF_TOKEN=<redacted>
uv run python scripts/populate_modal_volume.py
uv run python scripts/inspect_modal_volume.py
uv run python scripts/create_modal_smoke_payload.py
uv run python scripts/smoke_modal_inference.py --payload generated/modal_smoke_payload.json --out-dir local_runs/modal_smoke
```

## Local Verification

```text
uv run python --version
Python 3.11.15

uv run pytest
13 passed

uv run ruff check .
All checks passed

uv run python -c "import painting_inpaint_backend.modal.config; print('ok')"
ok

uv run --with build python -m build
Successfully built sdist and wheel

uv run python scripts/local_test_request.py --out generated/test_payload.json --size 64
wrote generated/test_payload.json
```

## Observations

| Scenario | Status | Observed latency | Notes |
| --- | --- | ---: | --- |
| Local Python | done | | Python 3.11.15 via `.python-version`. |
| Unit tests | done | | 13 passed. |
| Lint | done | | Ruff passed. |
| Package build | done | | sdist and wheel build succeeded. |
| Modal CLI version | done | | `modal client version: 1.5.1` |
| Modal profile | done | | Connected to the `danideos-kaiser` workspace. |
| Modal auth | done | | Browser authentication and token verification succeeded. |
| CPU smoke | done | 5.520 s | Remote function returned `message=ok`. |
| Volume population | done | 384.757 s | FLUX Fill: 54.069 GiB; LoRA: 0.167 GiB; both committed and marked ready. |
| Volume inspection | done | | Both expected directories exist and report `ready=true`. |
| First GPU invocation | historical failure | | The native spike tried to read `/root/requirements.txt`; the consolidated backend removes that duplicated dependency path. |
| First successful GPU smoke | done | 49.6 s observed | L40S, 512x512, 8 steps; pipeline load 8.639 s, LoRA 0.639 s, inference 26.330 s, peak allocated VRAM 23,206.85 MiB. |
| Consolidated-image FLUX inference | done | 94.3 s observed | L40S, 512x512, 8 steps; pipeline load 26.765 s, LoRA 1.292 s, inference 31.661 s. Model source was `explicit_local_path`, LoRA loaded from Volume, and hard composite preserved the outside mask. |
| Candidate-image FLUX inference | done | 81.1 s observed | Candidate `ee05730...`; L40S, 512x512, 8 steps; pipeline load 8.292 s, LoRA 0.481 s, inference 40.573 s, outside mask preserved. |
| Later likely-cold inference | pending | | |

## Evaluation Questions

1. Can Modal support the use case with zero warm workers? Yes for the initial smoke test; repeated cold tests remain.
2. Can FLUX Fill and LoRA weights be stored in a Volume and reused across cold starts? Yes.
3. Can inference run without downloading weights during requests? Yes. The consolidated backend must record `source=explicit_local_path` and `local_files_only=true`.
4. How much cold-start latency remains, and what dominates it? The native smoke observed 49.6 s. The first consolidated-image smoke observed 94.3 s, including 26.765 s pipeline loading and 31.661 s inference; image/container startup accounts for much of the remainder.
5. Does this look better than the current RunPod serverless setup? Pending.
6. What is the cleanest path to future client compatibility? Likely a provider transport layer.
7. Should this separate repo remain separate? No. Modal is now a provider adapter in the shared backend.

## Cleanup Resources

- App: `painting-inpaint-backend-modal`
- Smoke app: `painting-inpaint-backend-modal-smoke`
- Volume: `flux-fill-models`
- Secret: `huggingface-secret`

Cleanup:

```powershell
uv run modal volume delete flux-fill-models
uv run modal secret delete huggingface-secret
```

Modal Volume docs note that storage can still be billed for a few days after data
deletion due to storage snapshot accounting.
