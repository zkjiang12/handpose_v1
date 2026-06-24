# handpose_v1

Tools for visually inspecting EgoVerse hand-pose labels and using that
inspection to decide whether the labels are good enough for downstream model
training.

The main working tool today is a browser-based EgoVerse handpose visualizer.
It loads EgoVerse clips, overlays 2D projected hand keypoints on the video, and
shows the matching 3D handpose in camera-frame coordinates.

## What This Repo Contains

```text
scripts/egoverse_handpose_viewer.py   Browser visualizer and Python backend
scripts/view_egoverse_keypoints.py    Older script for rendering preview videos
scripts/audit_egoverse_dataset.py     Local Zarr metadata/schema audit
scripts/build_egoverse_handpose_manifest.py
                                      Aria frame-level train/test manifest builder
scripts/check_egoverse_handpose_dataset.py
                                      Manifest and dataset smoke checks
scripts/train_vit_egoverse_handpose.py
                                      Basic ViT hand-pose training entrypoint
Dockerfile                            Docker image for local/prod deployment
deploy/start.sh                       Container startup script
deploy/requirements-viewer.txt        Minimal Python deps for the viewer
deploy/RUNPOD_TRAINING.md             RunPod Docker training workflow
deploy/README.md                      Deployment notes
```

The viewer depends on public EgoVerse data access. It does not store the
dataset in this repository. Episodes are downloaded on demand into a local or
container cache.

## Visualizer Features

- Browse clips by embodiment and task.
- Search by task, episode hash, lab, or metadata.
- Move clip-by-clip and task-by-task.
- Play frames or step frame-by-frame.
- Render 2D handpose overlays on the egocentric video.
- Render matching 3D left/right handposes beside the video.
- Use 3D view presets: `Home`, `Angle`, `Side`, `Top`.
- Use camera-frame 3D coordinates by default:
  - `X right`
  - `Y down`
  - `Z forward`
- Uses colored finger bones and left/right colored joints.

## Quick Start With Docker

Docker is the easiest way to run the same code locally and in production.

Build:

```bash
cd /path/to/handpose_v1
docker build --platform linux/amd64 -t egoverse-handpose-viewer .
```

Run:

```bash
docker run --rm \
  --platform linux/amd64 \
  --name egoverse-handpose-viewer-local \
  -p 8770:8770 \
  -e AWS_ACCESS_KEY_ID='<public-egoverse-access-key-id>' \
  -e AWS_SECRET_ACCESS_KEY='<public-egoverse-secret-access-key>' \
  -e AWS_DEFAULT_REGION='us-east-2' \
  -v egoverse-viewer-cache:/data/egoverse_viewer_cache \
  egoverse-handpose-viewer
```

Use the public EgoVerse AWS values from the upstream EgoVerse README for the
two placeholder credential fields.

Open:

```text
http://127.0.0.1:8770
```

The first clip load can be slow because the viewer downloads and caches a Zarr
episode. Later loads are faster when the cache is warm.

## Dataset Audit

Before training, run the dataset audit to see which cached episodes have the
fields needed for hand-pose supervision:

```bash
cd /Users/you/dev/handpose_v1
/Users/you/dev/EgoVerse/emimic/bin/python \
  scripts/audit_egoverse_dataset.py \
  --cache-dir /Users/you/data/egoverse_viewer_cache \
  --out-dir outputs/dataset_audit
```

If `--cache-dir` is omitted, the script checks the common local cache paths:

```text
/Users/zikangjiang/data/egoverse_viewer_cache
/Users/zikangjiang/data/egoverse_keypoint_cache
```

The audit writes:

```text
outputs/dataset_audit/episodes.csv         Per-episode fields, shapes, dtypes, warnings
outputs/dataset_audit/episodes.json        Same data in JSON
outputs/dataset_audit/source_summary.csv   Per-source aggregate counts
outputs/dataset_audit/source_summary.json  Same summary in JSON
```

For hand-pose training, the useful episodes are the ones with:

```text
images.front_1
obs_head_pose
left.obs_keypoints or right.obs_keypoints
```

## Aria Training Prep

Run these commands from the repo root:

```bash
cd /Users/zikangjiang/dev/handpose_v1
```

Build an Aria-only frame manifest from the audit output:

```bash
/Users/zikangjiang/dev/EgoVerse/emimic/bin/python \
  scripts/build_egoverse_handpose_manifest.py \
  --audit-csv outputs/dataset_audit/episodes.csv \
  --out-dir outputs/handpose_dataset \
  --source aria/human \
  --frame-stride 5 \
  --test-fraction 0.2 \
  --seed 42 \
  --min-valid-joints 15
```

The manifest uses canonical `left/right.obs_keypoints`, not Aria-native
`left/right.obs_aria_keypoints`. It keeps single-hand frames and splits by
episode so train and test do not share clips.

Run the lightweight dataset checks:

```bash
/Users/zikangjiang/dev/EgoVerse/emimic/bin/python \
  scripts/check_egoverse_handpose_dataset.py \
  --train-csv outputs/handpose_dataset/train.csv \
  --test-csv outputs/handpose_dataset/test.csv
```

Build visibility-filtered train/test manifests when training a single-frame RGB
model. The filter keeps a hand only when at least half of its valid 3D joints
project into the front RGB image, then drops rows with no remaining supervised
hand:

```bash
/Users/zikangjiang/dev/EgoVerse/emimic/bin/python \
  scripts/filter_egoverse_handpose_visibility.py \
  --train-csv outputs/handpose_dataset/train.csv \
  --test-csv outputs/handpose_dataset/test.csv \
  --out-dir outputs/handpose_dataset_visible \
  --min-visible-ratio 0.5
```

Check the filtered manifests:

```bash
/Users/zikangjiang/dev/EgoVerse/emimic/bin/python \
  scripts/check_egoverse_handpose_dataset.py \
  --train-csv outputs/handpose_dataset_visible/train.csv \
  --test-csv outputs/handpose_dataset_visible/test.csv
```

Dry-run the training entrypoint without training:

```bash
/Users/zikangjiang/dev/EgoVerse/emimic/bin/python \
  scripts/train_vit_egoverse_handpose.py \
  --train-csv outputs/handpose_dataset/train.csv \
  --test-csv outputs/handpose_dataset/test.csv \
  --dry-run
```

Start a real local baseline run when ready:

```bash
/Users/zikangjiang/dev/EgoVerse/emimic/bin/python \
  scripts/train_vit_egoverse_handpose.py \
  --train-csv outputs/handpose_dataset/train.csv \
  --test-csv outputs/handpose_dataset/test.csv \
  --epochs 20 \
  --batch-size 64 \
  --num-workers 8 \
  --viz-every 1 \
  --viz-samples 4 \
  --ranked-viz-every 1 \
  --ranked-viz-percentile 10 \
  --ranked-viz-max-samples 10 \
  --ranked-viz-max-per-episode 2 \
  --ranked-viz-min-frame-gap 60
```

If `--out-dir` is omitted, the script creates the next numbered run folder:
`outputs/vit_runs/run_001`, `outputs/vit_runs/run_002`, and so on. Pass
`--out-dir outputs/vit_runs/my_name` only when you want a custom run name.

For the smallest visual smoke run, train on a tiny subset and save overlays
every epoch:

```bash
/Users/zikangjiang/dev/EgoVerse/emimic/bin/python \
  scripts/train_vit_egoverse_handpose.py \
  --train-csv outputs/handpose_dataset/train.csv \
  --test-csv outputs/handpose_dataset/test.csv \
  --epochs 5 \
  --batch-size 8 \
  --num-workers 4 \
  --overfit-batches 4 \
  --viz-every 1 \
  --viz-samples 4 \
  --open-live-plot
```

Training writes:

```text
outputs/vit_runs/<run>/metrics.jsonl           Loss and MPJPE per epoch
outputs/vit_runs/<run>/loss_curve.png          Live-updated loss/MPJPE chart
outputs/vit_runs/<run>/metrics_live.html       Auto-refreshing chart + overlay page
outputs/vit_runs/<run>/last.pt                 Latest checkpoint
outputs/vit_runs/<run>/best.pt                 Best checkpoint by test MPJPE
outputs/vit_runs/<run>/checkpoints/epoch_*.pt  Per-epoch checkpoints
outputs/vit_runs/<run>/viz/*.png               Predicted-vs-GT overlays
outputs/vit_runs/<run>/viz3d/*.png             3D GT-vs-prediction plots
outputs/vit_runs/<run>/ranked_viz/*/*.png      Best/worst test overlays by MPJPE
outputs/vit_runs/<run>/ranked_viz3d/*/*.png    Best/worst 3D plots by MPJPE
outputs/vit_runs/<run>/ranked_viz/ranked_samples.jsonl
                                                Ranked sample index with MPJPE
```

Ranked visualizations prefer diverse examples: the selector takes one frame per
episode before taking a second frame from the same episode, caps each bucket at
two frames per episode by default, and requires a 60-frame gap between repeated
frames from one episode.

Overlay colors:

```text
GT left/right: green and blue, with small points and skeleton edges
Pred left/right: red and pink, with skeleton edges
```

For a multi-GPU run:

```bash
torchrun --nproc_per_node 4 \
  scripts/train_vit_egoverse_handpose.py \
  --train-csv outputs/handpose_dataset/train.csv \
  --test-csv outputs/handpose_dataset/test.csv \
  --distributed
```

To train with the 21M-parameter DINOv3-S backbone:

```bash
python scripts/train_vit_egoverse_handpose.py \
  --train-csv outputs/handpose_dataset/train.csv \
  --test-csv outputs/handpose_dataset/test.csv \
  --out-dir outputs/vit_runs/dinov3_small_001 \
  --model-name vit_small_patch16_dinov3 \
  --backbone-source timm \
  --pretrained \
  --epochs 20 \
  --batch-size 32 \
  --lr 1e-4 \
  --num-workers 8 \
  --viz-every 1 \
  --viz-per-epoch 1 \
  --viz-samples 4 \
  --ranked-viz-every 1
```

Or use the Docker-oriented wrapper:

```bash
bash deploy/train_dinov3_small.sh
```

The command uses `timm`'s DINOv3-S implementation. Set `FREEZE_BACKBONE=1` for
a cheaper linear-probe style run that only trains the hand-pose head.

For cloud GPU training with Docker on RunPod, see
[`deploy/RUNPOD_TRAINING.md`](deploy/RUNPOD_TRAINING.md).

## Local Python Run

Use this if you already have the EgoVerse repo installed locally.

Expected local layout:

```text
/Users/you/dev/
  EgoVerse/
  handpose_v1/
```

Set up EgoVerse credentials first:

```bash
cd /Users/you/dev/EgoVerse
aws configure
./egomimic/utils/aws/setup_secret.sh
```

Then run the viewer:

```bash
cd /Users/you/dev/handpose_v1
/Users/you/dev/EgoVerse/emimic/bin/python \
  scripts/egoverse_handpose_viewer.py \
  --host 127.0.0.1 \
  --port 8770
```

Open:

```text
http://127.0.0.1:8770
```

## Deploying Publicly

The simplest public deployment is:

```text
GitHub repo -> Render Web Service -> Docker build -> public URL
```

Recommended Render settings:

```text
Runtime: Docker
Branch: main
Dockerfile path: ./Dockerfile
Disk mount path: /data/egoverse_viewer_cache
```

Environment variables:

```text
AWS_ACCESS_KEY_ID=<public-egoverse-access-key-id>
AWS_SECRET_ACCESS_KEY=<public-egoverse-secret-access-key>
AWS_DEFAULT_REGION=us-east-2
REGION=us-east-2
HOST=0.0.0.0
EGOVERSE_VIEWER_CACHE_DIR=/data/egoverse_viewer_cache
```

Use the public EgoVerse AWS values from the upstream EgoVerse README for the
two placeholder credential fields.

Do not set `PORT` manually unless the deployment platform requires it. Render
normally injects `PORT` automatically.

See [deploy/README.md](deploy/README.md) for more deployment details.

## How It Works

The visualizer is a small Python backend that serves a browser UI.

```text
Browser UI
  -> /api/tasks
  -> /api/episodes
  -> /api/frame_bundle
  -> /api/prefetch

Python backend
  -> reads EgoVerse metadata from the public database
  -> downloads Zarr episodes from public R2 storage
  -> caches episodes on disk
  -> decodes JPEG frames
  -> draws 2D handpose overlays
  -> returns matching 3D handpose JSON
```

The frontend and backend currently live in one file:

```text
scripts/egoverse_handpose_viewer.py
```

That is intentional for now. This is a research/QA tool, and keeping the data
access, projection code, and UI together made it faster to debug the dataset.

## Important API Endpoints

```text
GET /                         Browser UI
GET /api/tasks                List tasks for current filters
GET /api/episodes             List matching episodes
GET /api/frame_bundle         Render frame + matching 3D pose in one response
GET /api/frame                Render only the 2D overlay JPEG
GET /api/pose                 Return only 3D pose JSON
GET /api/prefetch             Start background episode download
```

Playback uses `/api/frame_bundle` so the video frame and 3D pose stay in sync,
especially in production where network latency is higher.

## Troubleshooting

If Docker says it cannot connect to the daemon, open Docker Desktop and wait
until it is fully running:

```bash
open -a Docker
docker info
```

If the Docker build fails on Apple Silicon because of Python wheels, build for
AMD64:

```bash
docker build --platform linux/amd64 -t egoverse-handpose-viewer .
```

If the page loads but frames fail, check logs:

```bash
docker logs -f egoverse-handpose-viewer-local
```

If first clip load is slow, that is usually the Zarr episode download/cache
warming. Once cached, frame rendering should be much faster.

## Research Goal

The broader goal is to build a reliable pipeline for extracting and evaluating
3D hand pose from egocentric video. Before training a model, we need to know
whether the labels are good enough.

Core questions:

- How good is the existing EgoVerse hand-pose data when inspected visually?
- How accurate are the EgoVerse labels against ground truth, in millimeters?
- Can a model trained on EgoVerse generalize to our own egocentric video data?
- How do current hand-pose systems compare on the same inputs?
- Can multiple systems be combined into a consensus labeler?

Evaluation targets:

- Mean per-joint position error, in millimeters.
- Per-joint error for fingertips, wrist, and occluded joints.
- Temporal stability across frames.
- Failure modes such as occlusion, motion blur, hand-object interaction,
  left/right swaps, depth scale errors, and impossible hand geometry.
- Visual QA through 2D overlays and 3D hand renderings.
