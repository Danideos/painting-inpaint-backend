# Consolidated Backend Rollout

## Rollback

RunPod rollback image:

```text
ghcr.io/danideos/flux-fill-worker:7eff1a8e323832a6e5d81c9f98b6d9db80744c4a
```

Standalone Modal baseline commit:

```text
1c2e9bb79c9576de09214726c48ef3266a972997
```

The selected native Modal output and timing files are preserved locally under the
ignored `local_runs/modal_native_baseline/` directory.

## Candidate

```text
ghcr.io/danideos/flux-fill-worker:ee05730575717631cb8ce0ea5cc096030a538cd5
ghcr.io/danideos/painting-inpaint-backend:ee05730575717631cb8ce0ea5cc096030a538cd5
```

The GitHub Actions build published both names from one build. The legacy package is
anonymously pullable. The new `painting-inpaint-backend` GHCR package still requires a
one-time visibility change to public in GitHub package settings.

## Modal Acceptance

- Direct JSON-string invocation completed on an NVIDIA L40S.
- Output was 512x512 RGB PNG.
- Model source was `explicit_local_path` at
  `/models/black-forest-labs/FLUX.1-Fill-dev`.
- `local_files_only` was true and Hugging Face offline modes were active.
- LoRA loaded from `/models/danideos/durer-flux-fill-lora`.
- `outside_mask_changed_after_hard_composite` was false.
- Candidate observed wall time was 81.1 seconds; inference was 40.573 seconds.

## RunPod Acceptance

- Endpoint `sroyp06ttnv9rn` was updated from the rollback image to the candidate.
- Existing endpoint settings were preserved, including zero minimum workers.
- The unchanged remote client streamed queue, preprocessing, model, LoRA, 22 effective
  inference steps, postprocessing, and completion events.
- Output was 1440x1440 RGB PNG on an NVIDIA A40.
- Observed wall time was 436.4 seconds; worker inference was 79.853 seconds.
- The long delay occurred before worker progress and is attributable to scheduling/cold
  allocation rather than inference execution.
