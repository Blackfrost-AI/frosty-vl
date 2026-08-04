ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:26.07-py3
FROM ${BASE_IMAGE}

LABEL org.opencontainers.image.title="Blackfrost Frosty VL"
LABEL org.opencontainers.image.description="Frosty VL persona-steering plugin for Qwen3-VL text encoders (video and native-audio service)"
LABEL org.opencontainers.image.vendor="Blackfrost Research"

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ffmpeg git \
    && apt-get clean \
    && find /var/lib/apt/lists -mindepth 1 -delete

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install \
       'git+https://github.com/huggingface/diffusers.git@abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc' \
    && python -m pip install -r /app/requirements.txt

COPY server /app/server
COPY ui /app/ui
COPY scripts /app/scripts
COPY config /app/config

RUN chmod 0555 /app/scripts/*.sh /app/scripts/preflight.py /app/ui/webui.py \
    && mkdir -p /outputs

ENV FVL_MODEL=/models/base-pipeline \
    FVL_REFUSAL_DIR=/app/server/data/refusal_dir_L50.npy \
    FVL_SAFETY_DIR=/app/server/data/safety_dir_L50.npy \
    FVL_ABL_LAM=3.0 \
    FVL_SAFETY_LAM=3.0 \
    FVL_SERVE_OUT=/outputs \
    FVL_DEVICE=cuda:0 \
    FVL_REF_DEVICE=cuda:1 \
    FVL_HOST=0.0.0.0 \
    FVL_PORT=8899

EXPOSE 8899 8890
HEALTHCHECK --interval=30s --timeout=10s --start-period=30m --retries=20 \
  CMD curl --fail --silent http://127.0.0.1:8899/health | grep -q '"ready":true' || exit 1

CMD ["python", "-m", "uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8899", "--app-dir", "/app/server"]
