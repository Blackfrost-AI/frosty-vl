#!/usr/bin/env bash
# This plugin ships no model weights. Point it at your own Qwen3-VL-based
# diffusion pipeline, which you obtain and license yourself.
#
# You have two options:
#   1. Place your Qwen3-VL-based pipeline directory at FVL_MODEL_DIR yourself, or
#   2. Set FVL_MODEL_REPO (and optionally FVL_MODEL_REVISION) to your own model
#      source and let this script fetch it via the Hugging Face CLI.
# There is no default model source; FVL_MODEL_REPO is required if you use this
# script to download.
set -euo pipefail

model_repo=${FVL_MODEL_REPO:-}
model_revision=${FVL_MODEL_REVISION:-main}
model_dir=${1:-${FVL_MODEL_DIR:-}}

if [[ -z "$model_repo" ]]; then
  echo "FVL_MODEL_REPO is not set. This plugin ships no weights: set FVL_MODEL_REPO" >&2
  echo "to your own Qwen3-VL-based pipeline repository, or place the pipeline at" >&2
  echo "FVL_MODEL_DIR yourself and skip this script." >&2
  exit 2
fi
if [[ -z "$model_dir" ]]; then
  echo "Usage: FVL_MODEL_REPO=<your/repo> FVL_MODEL_DIR=/absolute/model/path $0" >&2
  exit 2
fi
if [[ "$model_dir" != /* ]]; then
  echo "FVL_MODEL_DIR must be an absolute path: $model_dir" >&2
  exit 2
fi
if ! command -v hf >/dev/null 2>&1; then
  echo "The Hugging Face CLI is required: python3 -m pip install --user huggingface_hub" >&2
  exit 2
fi
if ! hf auth whoami >/dev/null 2>&1 && [[ -z "${HF_TOKEN:-}" ]]; then
  echo "Authenticate first with 'hf auth login' or export HF_TOKEN." >&2
  exit 2
fi

mkdir -p "$model_dir"
hf download "$model_repo" --revision "$model_revision" --local-dir "$model_dir"

required=(
  modular_model_index.json text_encoder tokenizer processor transformer
  transformer_ref vae audio_vae scheduler audio_scheduler
)
for item in "${required[@]}"; do
  if [[ ! -e "$model_dir/$item" ]]; then
    echo "Download incomplete: missing $model_dir/$item" >&2
    exit 1
  fi
done
echo "Model download complete: $model_dir"
