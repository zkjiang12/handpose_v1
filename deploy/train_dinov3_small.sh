#!/usr/bin/env bash
set -euo pipefail

export EGOVERSE_CACHE_DIR="${EGOVERSE_CACHE_DIR:-/data/egoverse_cache}"
export EGOVERSE_REPO="${EGOVERSE_REPO:-/opt/EgoVerse}"

TRAIN_CSV="${TRAIN_CSV:-outputs/handpose_dataset/train.csv}"
TEST_CSV="${TEST_CSV:-outputs/handpose_dataset/test.csv}"
OUT_DIR="${OUT_DIR:-/runs/dinov3_small_001}"
RESUME_ARG=()
FREEZE_ARG=()
HEAD_ARGS=()

if [ "${SETUP_EGOVERSE_SECRET:-1}" = "1" ] && [ ! -f "${HOME}/.egoverse_env" ] && [ -x "${EGOVERSE_REPO}/egomimic/utils/aws/setup_secret.sh" ]; then
  "${EGOVERSE_REPO}/egomimic/utils/aws/setup_secret.sh"
fi

if [ "${CACHE_BEFORE_TRAIN:-0}" = "1" ]; then
  CACHE_ARGS=()
  if [ "${CACHE_MAX_ROWS:-}" != "" ]; then
    CACHE_ARGS+=(--max-rows "$CACHE_MAX_ROWS")
  fi
  if [ "${CACHE_MAX_EPISODES:-}" != "" ]; then
    CACHE_ARGS+=(--max-episodes "$CACHE_MAX_EPISODES")
  fi
  python scripts/cache_egoverse_manifest_episodes.py \
    "$TRAIN_CSV" "$TEST_CSV" \
    --cache-dir "$EGOVERSE_CACHE_DIR" \
    "${CACHE_ARGS[@]}"
fi

if [ "${RESUME:-}" != "" ]; then
  RESUME_ARG=(--resume "$RESUME")
elif [ -f "$OUT_DIR/last.pt" ] && [ "${AUTO_RESUME:-1}" = "1" ]; then
  RESUME_ARG=(--resume "$OUT_DIR/last.pt")
fi

if [ "${FREEZE_BACKBONE:-0}" = "1" ]; then
  FREEZE_ARG=(--freeze-backbone)
fi

HEAD_ARGS=(
  --head-type "${HEAD_TYPE:-linear}"
  --head-hidden-dims "${HEAD_HIDDEN_DIMS:-1024,512}"
  --head-dropout "${HEAD_DROPOUT:-0.0}"
)

python scripts/train_vit_egoverse_handpose.py \
  --train-csv "$TRAIN_CSV" \
  --test-csv "$TEST_CSV" \
  --out-dir "$OUT_DIR" \
  --epochs "${EPOCHS:-20}" \
  --batch-size "${BATCH_SIZE:-32}" \
  --num-workers "${NUM_WORKERS:-8}" \
  --model-name "${MODEL_NAME:-vit_small_patch16_dinov3}" \
  --backbone-source "${BACKBONE_SOURCE:-timm}" \
  --pretrained \
  --lr "${LR:-1e-4}" \
  --weight-decay "${WEIGHT_DECAY:-0.0001}" \
  "${HEAD_ARGS[@]}" \
  --save-every "${SAVE_EVERY:-10}" \
  --plot-every "${PLOT_EVERY:-1}" \
  --log-every-steps "${LOG_EVERY_STEPS:-25}" \
  --viz-every "${VIZ_EVERY:-1}" \
  --viz-per-epoch "${VIZ_PER_EPOCH:-1}" \
  --viz-samples "${VIZ_SAMPLES:-4}" \
  "${FREEZE_ARG[@]}" \
  "${RESUME_ARG[@]}"
