#!/usr/bin/env bash
set -euo pipefail

bundle_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
env_file=${FVL_UI_ENV_FILE:-$bundle_dir/config/ui.env}
if [[ -f "$env_file" ]]; then
  set -a
  source "$env_file"
  set +a
fi

export FVL_GEN_TIMEOUT=${FVL_GEN_TIMEOUT:-2400}
exec python3 "$bundle_dir/ui/webui.py"
