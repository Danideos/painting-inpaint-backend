# Client Contract

This document records the logical restoration contract discovered from the current
RunPod worker and `painting-inpaint-remote-client`.

## Provider-Neutral Request

The logical request is a JSON object:

```json
{
  "prompt": "DURER_RESTO",
  "image_base64": "...",
  "mask_base64": "...",
  "partial_noise": 1.0,
  "guidance_scale": 30.0,
  "num_inference_steps": 30,
  "seed": 123,
  "lora_scale": 1.0,
  "output_format": "png",
  "negative_prompt": "",
  "max_sequence_length": 512
}
```

Required for Modal v1:

- `image_base64`
- `mask_base64`

Defaults:

- `prompt`: `DURER_RESTO`
- `partial_noise`: `1.0`
- `guidance_scale`: `30.0`
- `num_inference_steps`: `30`
- `lora_scale`: `1.0`
- `output_format`: `png`
- `negative_prompt`: empty string
- `max_sequence_length`: `512`

Mask convention: white means edit/inpaint, black means preserve.

## Provider-Neutral Response

The logical response is a JSON object:

```json
{
  "image_base64": "...",
  "output_format": "png",
  "width": 1024,
  "height": 1024,
  "mask_convention": "white = inpaint/edit, black = preserve",
  "timings": {},
  "gpu_memory": {},
  "model": {},
  "lora": {},
  "inference_settings": {},
  "schedule_debug": {},
  "outside_mask_changed_after_hard_composite": false
}
```

The public metadata/reporting layers must not include request `image_base64`,
`mask_base64`, tokens, or full payloads.

## RunPod-Specific Parts

RunPod wraps the request as:

```json
{"input": {"...": "..."}}
```

RunPod-specific behavior includes `/run`, `/status`, `/stream`, job IDs, queue
states, aggregate streaming events, and status values such as `IN_QUEUE`,
`COMPLETED`, `FAILED`, `CANCELLED`, and `TIMED_OUT`.

Modal should not imitate these internals unless future client compatibility
requires it. The first Modal spike uses direct function invocation with the
provider-neutral inner request.

## Future Compatibility Recommendation

Keep `RestorationClient` as the stable public API. Add a transport/provider layer
later:

- `RunPodTransport`: current queue/status/stream behavior.
- `ModalTransport`: direct Modal invocation or HTTP endpoint behavior.

The public result object can remain unchanged if Modal returns the provider-neutral
response shape above.

