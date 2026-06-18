FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app:/app/src \
    MODEL_ID=black-forest-labs/FLUX.1-Fill-dev \
    LORA_PATH=/app/runpod_worker/loras/pytorch_lora_weights.safetensors

WORKDIR /app

COPY runpod_worker/requirements.txt /app/runpod_worker/requirements.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install -r /app/runpod_worker/requirements.txt

COPY src/ /app/src/
COPY runpod_worker/ /app/runpod_worker/

CMD ["python", "-m", "runpod_worker.handler"]
