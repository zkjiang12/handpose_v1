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
Dockerfile                            Docker image for local/prod deployment
deploy/start.sh                       Container startup script
deploy/requirements-viewer.txt        Minimal Python deps for the viewer
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
