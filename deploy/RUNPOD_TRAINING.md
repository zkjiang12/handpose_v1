# RunPod Training

This is the simple Docker path for running EgoVerse hand-pose training on
RunPod. The Docker image contains code and dependencies only. EgoVerse cache
data and training outputs live on the RunPod persistent volume.

## Build and Push

From the repo root on your local machine:

```bash
export GHCR_IMAGE="ghcr.io/<owner>/<repo>/handpose-train"
export IMAGE_TAG="$(git rev-parse --short HEAD)"

docker build --platform linux/amd64 -f Dockerfile.train -t "$GHCR_IMAGE:$IMAGE_TAG" .
docker push "$GHCR_IMAGE:$IMAGE_TAG"
```

If the GitHub Container Registry package is private, authenticate on RunPod:

```bash
echo '<github-token>' | docker login ghcr.io -u '<github-username>' --password-stdin
```

## RunPod Volume Layout

Create a RunPod pod with a persistent volume. Use these paths on the pod:

```text
/workspace/egoverse_cache
/workspace/runs
```

The container mounts them as:

```text
/data/egoverse_cache
/runs
```

## Required Environment

Set the public EgoVerse bootstrap AWS values from the upstream EgoVerse README:

```bash
export AWS_ACCESS_KEY_ID='<public-egoverse-access-key-id>'
export AWS_SECRET_ACCESS_KEY='<public-egoverse-secret-access-key>'
export AWS_DEFAULT_REGION='us-east-2'
```

The training scripts run EgoVerse `setup_secret.sh` automatically when
`~/.egoverse_env` is missing.

## Smoke Run

Run this first. It caches only the first manifest rows needed by the tiny smoke
job, then trains for two epochs and writes to `/workspace/runs/smoke`.

```bash
docker run --gpus all --rm \
  -v /workspace/egoverse_cache:/data/egoverse_cache \
  -v /workspace/runs:/runs \
  -e AWS_ACCESS_KEY_ID \
  -e AWS_SECRET_ACCESS_KEY \
  -e AWS_DEFAULT_REGION \
  -e EGOVERSE_CACHE_DIR=/data/egoverse_cache \
  "$GHCR_IMAGE:$IMAGE_TAG" \
  bash deploy/train_smoke.sh
```

Expected outputs:

```text
/workspace/runs/smoke/config.json
/workspace/runs/smoke/metrics.jsonl
/workspace/runs/smoke/last.pt
/workspace/runs/smoke/loss_curve.png
/workspace/runs/smoke/viz/
```

## Resume Smoke Run

To verify resume, increase the target epoch count and point at the last
checkpoint:

```bash
docker run --gpus all --rm \
  -v /workspace/egoverse_cache:/data/egoverse_cache \
  -v /workspace/runs:/runs \
  -e AWS_ACCESS_KEY_ID \
  -e AWS_SECRET_ACCESS_KEY \
  -e AWS_DEFAULT_REGION \
  -e EGOVERSE_CACHE_DIR=/data/egoverse_cache \
  -e EPOCHS=3 \
  -e RESUME=/runs/smoke/last.pt \
  "$GHCR_IMAGE:$IMAGE_TAG" \
  bash deploy/train_smoke.sh
```

The script also auto-resumes from `/runs/smoke/last.pt` when it exists unless
`AUTO_RESUME=0` is set.

## Full Cache Warmup

Before a full run, cache all episodes referenced by the manifest:

```bash
docker run --gpus all --rm \
  -v /workspace/egoverse_cache:/data/egoverse_cache \
  -e AWS_ACCESS_KEY_ID \
  -e AWS_SECRET_ACCESS_KEY \
  -e AWS_DEFAULT_REGION \
  -e EGOVERSE_CACHE_DIR=/data/egoverse_cache \
  "$GHCR_IMAGE:$IMAGE_TAG" \
  python scripts/cache_egoverse_manifest_episodes.py \
    outputs/handpose_dataset/train.csv \
    outputs/handpose_dataset/test.csv \
    --cache-dir /data/egoverse_cache
```

## Real Training Run

```bash
docker run --gpus all --rm \
  -v /workspace/egoverse_cache:/data/egoverse_cache \
  -v /workspace/runs:/runs \
  -e AWS_ACCESS_KEY_ID \
  -e AWS_SECRET_ACCESS_KEY \
  -e AWS_DEFAULT_REGION \
  -e EGOVERSE_CACHE_DIR=/data/egoverse_cache \
  "$GHCR_IMAGE:$IMAGE_TAG" \
  bash deploy/train_vit_base.sh
```

## DINOv3-S Training Run

This runs the 21M-parameter DINOv3-S backbone through `timm` with the existing
hand-pose regression head:

```bash
docker run --gpus all --rm \
  -v /workspace/egoverse_cache:/data/egoverse_cache \
  -v /workspace/runs:/runs \
  -e AWS_ACCESS_KEY_ID \
  -e AWS_SECRET_ACCESS_KEY \
  -e AWS_DEFAULT_REGION \
  -e EGOVERSE_CACHE_DIR=/data/egoverse_cache \
  -e OUT_DIR=/runs/dinov3_small_001 \
  "$GHCR_IMAGE:$IMAGE_TAG" \
  bash deploy/train_dinov3_small.sh
```

For a cheaper first pass that only trains the hand-pose head:

```bash
-e FREEZE_BACKBONE=1
```

## Overnight DINOv3 Sweep

Use one pod/GPU per run. These commands assume the pod already has the repo at
`/workspace/handpose_v1`, the cache at `/workspace/egoverse_cache`, and runs at
`/workspace/runs`.

DINOv3-B with the linear head:

```bash
cd /workspace/handpose_v1
git pull --ff-only origin main

nohup env \
  EGOVERSE_CACHE_DIR=/workspace/egoverse_cache \
  OUT_DIR=/workspace/runs/dinov3_base_linear_001 \
  CACHE_BEFORE_TRAIN=0 \
  EPOCHS=100 \
  BATCH_SIZE=64 \
  NUM_WORKERS=8 \
  LR=1e-4 \
  SAVE_EVERY=10 \
  HEAD_TYPE=linear \
  bash deploy/train_dinov3_base.sh \
  > /workspace/runs/dinov3_base_linear_001.log 2>&1 &
```

DINOv3-B with the larger MLP head:

```bash
cd /workspace/handpose_v1
git pull --ff-only origin main

nohup env \
  EGOVERSE_CACHE_DIR=/workspace/egoverse_cache \
  OUT_DIR=/workspace/runs/dinov3_base_mlp_001 \
  CACHE_BEFORE_TRAIN=0 \
  EPOCHS=100 \
  BATCH_SIZE=64 \
  NUM_WORKERS=8 \
  LR=1e-4 \
  SAVE_EVERY=10 \
  HEAD_TYPE=mlp \
  HEAD_HIDDEN_DIMS=2048,1024 \
  HEAD_DROPOUT=0.1 \
  bash deploy/train_dinov3_base.sh \
  > /workspace/runs/dinov3_base_mlp_001.log 2>&1 &
```

DINOv3-S with the larger MLP head:

```bash
cd /workspace/handpose_v1
git pull --ff-only origin main

nohup env \
  EGOVERSE_CACHE_DIR=/workspace/egoverse_cache \
  OUT_DIR=/workspace/runs/dinov3_small_mlp_001 \
  CACHE_BEFORE_TRAIN=0 \
  EPOCHS=100 \
  BATCH_SIZE=64 \
  NUM_WORKERS=8 \
  LR=1e-4 \
  SAVE_EVERY=10 \
  HEAD_TYPE=mlp \
  HEAD_HIDDEN_DIMS=1024,512 \
  HEAD_DROPOUT=0.1 \
  bash deploy/train_dinov3_small.sh \
  > /workspace/runs/dinov3_small_mlp_001.log 2>&1 &
```

Check progress:

```bash
tail -n 5 /workspace/runs/<run_name>/metrics.jsonl
tail -f /workspace/runs/<run_name>.log
```

Useful overrides:

```bash
-e OUT_DIR=/runs/vit_base_002
-e EPOCHS=200
-e BATCH_SIZE=48
-e NUM_WORKERS=12
-e MODEL_NAME=vit_large_patch16_224
-e RESUME=/runs/vit_base_001/last.pt
```

## Multi-GPU

For a multi-GPU pod, run `torchrun` directly:

```bash
docker run --gpus all --rm \
  -v /workspace/egoverse_cache:/data/egoverse_cache \
  -v /workspace/runs:/runs \
  -e AWS_ACCESS_KEY_ID \
  -e AWS_SECRET_ACCESS_KEY \
  -e AWS_DEFAULT_REGION \
  -e EGOVERSE_CACHE_DIR=/data/egoverse_cache \
  "$GHCR_IMAGE:$IMAGE_TAG" \
  torchrun --nproc_per_node 4 scripts/train_vit_egoverse_handpose.py \
    --train-csv outputs/handpose_dataset/train.csv \
    --test-csv outputs/handpose_dataset/test.csv \
    --distributed \
    --out-dir /runs/vit_base_ddp_001 \
    --epochs 100 \
    --batch-size 64 \
    --num-workers 8 \
    --model-name vit_base_patch16_224 \
    --pretrained
```
