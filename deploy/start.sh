#!/usr/bin/env bash
set -euo pipefail

export EGOVERSE_REPO="${EGOVERSE_REPO:-/opt/EgoVerse}"
export EGOVERSE_VIEWER_CACHE_DIR="${EGOVERSE_VIEWER_CACHE_DIR:-/data/egoverse_viewer_cache}"
export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-8770}"
export REGION="${REGION:-${AWS_DEFAULT_REGION:-us-east-2}}"

mkdir -p "$EGOVERSE_VIEWER_CACHE_DIR"

if [ ! -f "${HOME}/.egoverse_env" ] && [ -x "${EGOVERSE_REPO}/egomimic/utils/aws/setup_secret.sh" ]; then
  "${EGOVERSE_REPO}/egomimic/utils/aws/setup_secret.sh"
fi

exec python /app/scripts/egoverse_handpose_viewer.py \
  --host "$HOST" \
  --port "$PORT" \
  --cache-dir "$EGOVERSE_VIEWER_CACHE_DIR"
