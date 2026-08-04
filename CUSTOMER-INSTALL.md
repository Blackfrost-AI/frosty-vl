# Customer installation checklist

This is the short installation-call version. Use `README.md` for architecture,
configuration, troubleshooting, and provenance details.

Frosty VL is a persona-steering plugin for Qwen3-VL text encoders. It ships no
model weights: you bring your own Qwen3-VL-based diffusion pipeline as the base
model.

This is the **SAFE release**: the golden-safety floor is ON by default
(`FVL_SAFETY_LAM=3.0`). All steering is a runtime, in-memory projection; the base
pipeline on disk is never modified.

## Customer prerequisites

- Linux x86_64 host with two B200-class GPUs;
- NVIDIA driver and NVIDIA Container Toolkit working with Docker;
- Docker Compose v2;
- enough free disk for your base pipeline (a full BF16 Qwen3-VL video pipeline
  is on the order of several hundred GB);
- your own copy of a compatible Qwen3-VL-based diffusion pipeline, which you
  obtain and license yourself.

Verify GPU containers first:

```bash
docker run --rm --gpus all nvcr.io/nvidia/pytorch:26.07-py3 \
  python -c 'import torch; print(torch.cuda.device_count()); print(torch.cuda.get_device_name(0))'
```

## Install

```bash
cd /opt/Frosty-VL-Deploy

# Provide your own Qwen3-VL base pipeline to persistent host storage.
# Either copy your pipeline directory to FVL_MODEL_DIR yourself, or fetch it from
# a model source you control (no default source is assumed):
python3 -m pip install --user --upgrade huggingface_hub
hf auth login
export FVL_MODEL_REPO=<your/qwen3-vl-pipeline-repo>
export FVL_MODEL_DIR=/models/base-pipeline
./scripts/download-model.sh

# Configure persistent host paths and local/private port bindings.
cp env.docker.template .env
mkdir -p /srv/frosty-vl/outputs
# Edit .env if the model/output paths or GPU IDs differ.

# Build the customer image, validate without loading weights, then start.
docker compose build
./scripts/container-preflight.sh
docker compose up -d
docker compose logs -f frosty-vl-server
```

Wait for both pipelines:

```bash
curl http://127.0.0.1:8899/health
```

```json
{"status":"ok","ready":true,"reference_ready":true}
```

Then:

```bash
./scripts/smoke-test.sh http://127.0.0.1:8899
```

Open `http://localhost:8890` for the Frosty VL Studio.

## Included product behavior

- FL2VA/text-to-video on GPU 0;
- image-to-video opening-frame mode;
- Ref2VA Same Person Lock on GPU 1;
- Scene Lab and FFmpeg scene assembly;
- embedded Frosty VL persona-steering direction plus golden-safety floor
  (ON in this SAFE release), both applied in memory;
- 2,400-second long-render timeout.

The base-pipeline mount is read-only and is never modified on disk.
