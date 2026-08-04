#!/usr/bin/env bash
set -euo pipefail

base=${1:-http://127.0.0.1:8899}
echo "Service info:"
curl --fail --silent --show-error "$base/" | python3 -m json.tool
echo "Health:"
curl --fail --silent --show-error "$base/health" | python3 -m json.tool
echo "OpenAPI includes generation fields:"
curl --fail --silent --show-error "$base/openapi.json" |
  python3 -c 'import json,sys; d=json.load(sys.stdin); print(json.dumps(d["components"]["schemas"]["GenRequest"], indent=2))'
