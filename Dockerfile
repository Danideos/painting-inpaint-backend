FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime AS base

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip uv

COPY pyproject.toml uv.lock README.md /app/

FROM base AS fill

ENV MODEL_ID=black-forest-labs/FLUX.1-Fill-dev \
    LORA_PATH=/app/assets/loras/pytorch_lora_weights.safetensors

RUN uv export --frozen --no-dev --extra inference --extra runpod \
        --no-emit-project --output-file /tmp/backend-requirements.txt \
    && uv pip install --system --requirements /tmp/backend-requirements.txt

COPY src/ /app/src/
COPY assets/ /app/assets/
RUN uv pip install --system --no-deps .

CMD ["python", "-m", "painting_inpaint_backend.runpod.handler"]

FROM base AS canny

RUN uv export --frozen --no-dev --extra inference --extra canny \
        --no-emit-project --output-file /tmp/backend-requirements.txt \
    && uv pip install --system --requirements /tmp/backend-requirements.txt

COPY src/ /app/src/
COPY assets/ /app/assets/
COPY THIRD_PARTY_NOTICES.md /app/THIRD_PARTY_NOTICES.md
RUN uv pip install --system --no-deps .

FROM fill AS final
