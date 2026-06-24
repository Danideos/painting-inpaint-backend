# Timings

Record timing observations here after each Modal run.

| Run label | GPU | Queue/pending | Container/image startup | Volume/model load | LoRA load | Inference | Total observed | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| first successful cold smoke | L40S | not separately exposed | ~14.6 s combined overhead | 8.639 s pipeline total | 0.639 s | 26.330 s | 49.6 s | 512x512, 8 steps, image cached from prior failed attempts, peak allocated VRAM 23,206.85 MiB |
| first consolidated-image smoke | L40S | not separately exposed | included in 94.3 s observed | 26.765 s pipeline | 1.292 s | 31.661 s | 94.3 s | 512x512, 8 steps, immutable GHCR image `4b9b277...`, Volume-only load, hard composite preserved outside mask |
| candidate-image smoke | L40S | not separately exposed | included in 81.1 s observed | 8.292 s pipeline | 0.481 s | 40.573 s | 81.1 s | 512x512, 8 steps, candidate `ee05730...`, Volume-only offline load, outside mask preserved |
| HTTP remote-client Rosary smoke | L40S | included before first NDJSON event | HTTP request duration 76.5 s | 8.601 s pipeline | 0.582 s | 31.450 s | 79.455 s | 1024x1024, 8 steps, authenticated endpoint, all progress states, explicit Volume paths, outside mask preserved |
| final pinned HTTP confirmation | L40S | included before first NDJSON event | included in observed total | Volume-only local load | loaded | 4 streamed steps | 74.260 s | 512x512, image `e66a412...`, all public progress states, outside mask preserved |
| short-later | L40S | pending | pending | pending | pending | pending | pending | |
| likely-cold-later | L40S | pending | pending | pending | pending | pending | pending | |
