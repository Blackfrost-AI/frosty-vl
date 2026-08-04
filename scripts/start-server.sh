#!/usr/bin/env bash
set -euo pipefail

bundle_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
env_file=${FVL_ENV_FILE:-$bundle_dir/config/server.env}
if [[ -f "$env_file" ]]; then
  set -a
  source "$env_file"
  set +a
fi

: "${FVL_MODEL:?set FVL_MODEL}"
: "${FVL_REFUSAL_DIR:?set FVL_REFUSAL_DIR}"

required_artifacts=("$FVL_MODEL/modular_model_index.json" "$FVL_REFUSAL_DIR")
if [[ ! "${FVL_SAFETY_LAM:-0}" =~ ^0([.]0+)?$ ]]; then
  : "${FVL_SAFETY_DIR:?set FVL_SAFETY_DIR when FVL_SAFETY_LAM is enabled}"
  required_artifacts+=("$FVL_SAFETY_DIR")
fi

for required in "${required_artifacts[@]}"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing required artifact: $required" >&2
    exit 2
  fi
done

fvl_venv=${FVL_VENV:-$HOME/frosty-vl-venv}
fvl_app_dir=${FVL_APP_DIR:-$bundle_dir/server}
fvl_host=${FVL_HOST:-0.0.0.0}
fvl_port=${FVL_PORT:-8899}

exec "$fvl_venv/bin/python" -m uvicorn serve:app \
  --host "$fvl_host" --port "$fvl_port" --app-dir "$fvl_app_dir"
