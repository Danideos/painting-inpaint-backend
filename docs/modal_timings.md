# Timings

Record timing observations here after each Modal run.

| Run label | GPU | Queue/pending | Container/image startup | Volume/model load | LoRA load | Inference | Total observed | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| first successful cold smoke | L40S | not separately exposed | ~14.6 s combined overhead | 8.639 s pipeline total | 0.639 s | 26.330 s | 49.6 s | 512x512, 8 steps, image cached from prior failed attempts, peak allocated VRAM 23,206.85 MiB |
| first consolidated-image smoke | L40S | not separately exposed | included in 94.3 s observed | 26.765 s pipeline | 1.292 s | 31.661 s | 94.3 s | 512x512, 8 steps, immutable GHCR image `4b9b277...`, Volume-only load, hard composite preserved outside mask |
| candidate-image smoke | L40S | not separately exposed | included in 81.1 s observed | 8.292 s pipeline | 0.481 s | 40.573 s | 81.1 s | 512x512, 8 steps, candidate `ee05730...`, Volume-only offline load, outside mask preserved |
| short-later | L40S | pending | pending | pending | pending | pending | pending | |
| likely-cold-later | L40S | pending | pending | pending | pending | pending | pending | |
