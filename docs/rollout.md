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

The GitHub Actions build published both names from one build. Both packages were made
anonymously pullable for the migration. RunPod now references the new backend package,
and subsequent workflows publish only `painting-inpaint-backend`; historical
`flux-fill-worker` tags remain available for rollback.

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

- Endpoint `sroyp06ttnv9rn` was updated from the rollback image to the candidate and then
  switched to the identical `painting-inpaint-backend` image name.
- Existing endpoint settings were preserved, including zero minimum workers.
- The unchanged remote client streamed queue, preprocessing, model, LoRA, 22 effective
  inference steps, postprocessing, and completion events.
- Output was 1440x1440 RGB PNG on an NVIDIA A40.
- Observed wall time was 436.4 seconds; worker inference was 79.853 seconds.
- The long delay occurred before worker progress and is attributable to scheduling/cold
  allocation rather than inference execution.

## Modal Client Rollout

The Modal HTTP deployment must reference an immutable backend image containing both the
shared stream engine and `FluxFillModalBackend.restore_stream`. Create the Modal Secret
`painting-inpaint-restoration-api` with key `RESTORATION_API_KEY`, deploy the app, and
give applications only the resulting endpoint URL and dedicated restoration key.

Acceptance requires a real remote-client request to report queued, preprocessing,
model preparation, inference steps, postprocessing, and completion. The result must
still show L40S, explicit Volume model and LoRA paths, offline loading, and exact
outside-mask preservation. RunPod remains on its current image until its regression
smoke against the new shared stream code passes.

Final deployed backend image:

```text
ghcr.io/danideos/painting-inpaint-backend:e66a412b581261581465bc22ec0b2f049c731557
sha256:e927c53a15c487bc08864f4aa9d3439e025b5d41ccafb19b904a9c841f40e459
```

The image is anonymously pullable. The Modal app is deployed at the `restoration_api`
web function and is pinned to this immutable tag. The dedicated bearer key lives only
in Modal Secret `painting-inpaint-restoration-api` and the ignored Drive `.env`.

### HTTP Acceptance

- The new remote client completed a 1024x1024 Rosary center request in 79.455 seconds.
- The endpoint streamed queued, preprocessing, model preparation, 8 inference steps,
  postprocessing, and completion.
- The GPU was NVIDIA L40S; model source was `explicit_local_path` with
  `local_files_only=true`.
- LoRA loaded from `/models/danideos/durer-flux-fill-lora` and inference downloaded no
  weights.
- Inference took 31.450 seconds and outside-mask preservation remained exact.
- A final 512x512 confirmation against image `e66a412...` completed in 74.260 seconds
  with all public progress states and 4 inference steps.

### RunPod Client Regression

The existing RunPod endpoint image was not changed. The updated client defaulted to
RunPod, completed a 512x512 four-step request on NVIDIA A40 in 52.084 seconds, streamed
all public progress states, decoded the final image, and preserved pixels outside the
mask. Moving RunPod to the new backend image remains a separate deployment decision.
