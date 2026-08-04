#!/usr/bin/env bash
set -euo pipefail

bundle_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
env_file=${FROST_WAN_ENV_FILE:-$bundle_dir/config/wan.env}
if [[ -f "$env_file" ]]; then
  set -a
  source "$env_file"
  set +a
fi

: "${FROST_MODEL:?set FROST_MODEL to the Wan Diffusers checkpoint directory}"
if [[ ! -f "$FROST_MODEL/model_index.json" ]]; then
  echo "Wan checkpoint is incomplete: $FROST_MODEL/model_index.json is missing" >&2
  exit 1
fi

frost_host=${FROST_HOST:-127.0.0.1}
frost_port=${FROST_PORT:-8902}
frost_venv=${FROST_VENV:-$HOME/frosty-vl-venv}
exec "$frost_venv/bin/python" -m uvicorn wan_serve:app \
  --app-dir "$bundle_dir/server" --host "$frost_host" --port "$frost_port"
