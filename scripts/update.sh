#!/usr/bin/env bash
# Frosty VL updater — updates FRAMEWORK CODE ONLY.
#
# It pulls the latest Frosty VL source, rebuilds the Studio container, and syncs
# the ComfyUI node into your ComfyUI install. It NEVER touches your base model
# weights or your steering vector files (comfyui/data/*.npy, comfyui/data/*.json,
# server/data/*.npy, server/data/*.json). Those are yours; they are preserved.
#
# Usage:
#   bash scripts/update.sh [--force]
#
# Environment:
#   FVL_COMFYUI_DIR  If set, the ComfyUI install root; the node is synced into
#                    $FVL_COMFYUI_DIR/custom_nodes/frosty-vl (your local
#                    comfyui/data/ vectors there are preserved, never overwritten).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

RAW_VERSION_URL="https://raw.githubusercontent.com/Blackfrost-AI/frosty-vl/main/VERSION"

FORCE=0
if [[ "${1:-}" == "--force" ]]; then
  FORCE=1
fi

log()  { printf '[update] %s\n' "$*"; }
warn() { printf '[update] WARNING: %s\n' "$*" >&2; }

# ---- 1. current version ----------------------------------------------------
if [[ -f VERSION ]]; then
  CURRENT="$(tr -d '[:space:]' < VERSION)"
else
  CURRENT="unknown"
fi
log "current version: ${CURRENT}"

# ---- 2. fetch latest version ----------------------------------------------
LATEST=""
if command -v curl >/dev/null 2>&1; then
  if LATEST_RAW="$(curl -fsSL "$RAW_VERSION_URL" 2>/dev/null)"; then
    LATEST="$(printf '%s' "$LATEST_RAW" | tr -d '[:space:]')"
    log "latest version:  ${LATEST}"
  else
    warn "could not reach ${RAW_VERSION_URL}; continuing without a version check."
  fi
else
  warn "curl not found; skipping remote version check."
fi

# ---- 3. decide whether to pull --------------------------------------------
DO_PULL=0
if [[ "$FORCE" -eq 1 ]]; then
  log "--force given; will pull regardless of version."
  DO_PULL=1
elif [[ -n "$LATEST" && "$LATEST" != "$CURRENT" ]]; then
  # A differing remote version means an update is available.
  log "a different version is available (${CURRENT} -> ${LATEST})."
  DO_PULL=1
else
  log "already up to date; use --force to pull anyway."
fi

if [[ "$DO_PULL" -eq 1 ]]; then
  if command -v git >/dev/null 2>&1 && [[ -d .git ]]; then
    log "git pull --ff-only origin main"
    if ! git pull --ff-only origin main; then
      warn "git pull failed (local changes or diverged history?); resolve manually."
    fi
  else
    warn "not a git checkout or git missing; skipping source pull."
  fi
fi

# ---- 4. Studio: rebuild + preflight + up ----------------------------------
have_compose() {
  if docker compose version >/dev/null 2>&1; then
    COMPOSE=(docker compose); return 0
  elif command -v docker-compose >/dev/null 2>&1; then
    COMPOSE=(docker-compose); return 0
  fi
  return 1
}

if command -v docker >/dev/null 2>&1 && have_compose; then
  log "Studio: ${COMPOSE[*]} build --pull"
  if "${COMPOSE[@]}" build --pull; then
    if [[ -f scripts/container-preflight.sh ]]; then
      log "Studio: preflight"
      if ! bash scripts/container-preflight.sh; then
        warn "container preflight reported problems; not starting the Studio."
      else
        log "Studio: ${COMPOSE[*]} up -d"
        "${COMPOSE[@]}" up -d || warn "Studio start failed; check 'docker compose logs'."
      fi
    else
      log "Studio: ${COMPOSE[*]} up -d"
      "${COMPOSE[@]}" up -d || warn "Studio start failed; check 'docker compose logs'."
    fi
  else
    warn "Studio image build failed; skipping start."
  fi
else
  log "Docker / Docker Compose not found; skipping the Studio update."
  log "  Install Docker Engine + Compose v2, then re-run: bash scripts/update.sh --force"
fi

# ---- 5. ComfyUI: sync the node --------------------------------------------
if [[ -n "${FVL_COMFYUI_DIR:-}" ]]; then
  DEST="${FVL_COMFYUI_DIR%/}/custom_nodes/frosty-vl"
  log "ComfyUI: syncing node into ${DEST} (your data/ vectors are preserved)."
  mkdir -p "$DEST"
  if command -v rsync >/dev/null 2>&1; then
    # --ignore-existing on data/ preserves the customer's local vector files.
    rsync -a --delete \
      --exclude 'data/*.npy' --exclude 'data/*.json' \
      --exclude '__pycache__' \
      "$REPO_ROOT/comfyui/" "$DEST/"
    # Ensure the data dir + its README exist without clobbering customer vectors.
    mkdir -p "$DEST/data"
    [[ -f "$REPO_ROOT/comfyui/data/README.md" ]] && \
      cp -f "$REPO_ROOT/comfyui/data/README.md" "$DEST/data/README.md"
  else
    warn "rsync not found; falling back to cp (customer vectors under data/ preserved)."
    mkdir -p "$DEST/data"
    # copy everything except the vector files
    ( cd "$REPO_ROOT/comfyui" && \
      find . -type f \
        ! -path './data/*.npy' ! -path './data/*.json' \
        ! -path '*/__pycache__/*' -print0 \
      | while IFS= read -r -d '' f; do
          mkdir -p "$DEST/$(dirname "$f")"
          cp -f "$f" "$DEST/$f"
        done )
  fi
  log "ComfyUI node updated. Restart ComfyUI to load the new node code."
else
  log "FVL_COMFYUI_DIR not set; skipping ComfyUI node sync."
  log "  To install the node: copy the comfyui/ folder to"
  log "  <ComfyUI>/custom_nodes/frosty-vl, add your data/ vectors, and restart ComfyUI."
  log "  Or set FVL_COMFYUI_DIR=<ComfyUI root> and re-run this script."
fi

log "done. Framework code updated; base model weights and vector files untouched."
