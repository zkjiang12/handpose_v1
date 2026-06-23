#!/usr/bin/env bash
set -euo pipefail

export EGOVERSE_CACHE_DIR="${EGOVERSE_CACHE_DIR:-/data/egoverse_cache}"
export EGOVERSE_REPO="${EGOVERSE_REPO:-/opt/EgoVerse}"

TRAIN_CSV="${TRAIN_CSV:-outputs/handpose_dataset/train.csv}"
TEST_CSV="${TEST_CSV:-outputs/handpose_dataset/test.csv}"
OUT_DIR="${OUT_DIR:-/runs/smoke}"
BATCH_SIZE="${BATCH_SIZE:-8}"
OVERFIT_BATCHES="${OVERFIT_BATCHES:-4}"
CACHE_MAX_ROWS="${CACHE_MAX_ROWS:-$((BATCH_SIZE * OVERFIT_BATCHES))}"
RESUME_ARG=()

if [ "${SETUP_EGOVERSE_SECRET:-1}" = "1" ] && [ ! -f "${HOME}/.egoverse_env" ] && [ -x "${EGOVERSE_REPO}/egomimic/utils/aws/setup_secret.sh" ]; then
  "${EGOVERSE_REPO}/egomimic/utils/aws/setup_secret.sh"
fi

if [ "${CACHE_BEFORE_TRAIN:-1}" = "1" ]; then
  python scripts/cache_egoverse_manifest_episodes.py \
    "$TRAIN_CSV" "$TEST_CSV" \
    --cache-dir "$EGOVERSE_CACHE_DIR" \
    --max-rows "$CACHE_MAX_ROWS"
fi

if [ "${RESUME:-}" != "" ]; then
  RESUME_ARG=(--resume "$RESUME")
elif [ -f "$OUT_DIR/last.pt" ] && [ "${AUTO_RESUME:-1}" = "1" ]; then
  RESUME_ARG=(--resume "$OUT_DIR/last.pt")
fi

python scripts/train_vit_egoverse_handpose.py \
  --train-csv "$TRAIN_CSV" \
  --test-csv "$TEST_CSV" \
  --out-dir "$OUT_DIR" \
  --epochs "${EPOCHS:-2}" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "${NUM_WORKERS:-4}" \
  --overfit-batches "$OVERFIT_BATCHES" \
  --max-eval-steps "${MAX_EVAL_STEPS:-4}" \
  --viz-every "${VIZ_EVERY:-1}" \
  --viz-samples "${VIZ_SAMPLES:-4}" \
  "${RESUME_ARG[@]}"
