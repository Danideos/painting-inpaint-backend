FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src \
    MODEL_ID=black-forest-labs/FLUX.1-Fill-dev \
    LORA_PATH=/app/assets/loras/pytorch_lora_weights.safetensors

WORKDIR /app

RUN python -m pip install --upgrade pip uv

COPY pyproject.toml uv.lock README.md /app/
RUN uv export --frozen --no-dev --extra inference --extra runpod \
        --no-emit-project --output-file /tmp/backend-requirements.txt \
    && uv pip install --system --requirements /tmp/backend-requirements.txt

COPY src/ /app/src/
COPY assets/ /app/assets/
RUN uv pip install --system --no-deps .

CMD ["python", "-m", "painting_inpaint_backend.runpod.handler"]
