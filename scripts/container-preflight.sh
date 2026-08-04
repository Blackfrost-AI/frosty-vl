#!/usr/bin/env bash
set -euo pipefail

docker compose run --rm --no-deps frosty-vl-server python /app/scripts/preflight.py
