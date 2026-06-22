# Deploying the EgoVerse Handpose Viewer

This folder contains the deployment helpers for the browser visualizer.

The app is deployed as a Docker service. The Docker image installs only the
viewer runtime dependencies, not the full EgoVerse training stack.

## Files

```text
Dockerfile                       Image definition, stored at repo root
deploy/start.sh                  Container startup script
deploy/requirements-viewer.txt   Minimal Python packages for the viewer
```

## Local Docker Test

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

Useful local commands:

```bash
docker logs -f egoverse-handpose-viewer-local
docker stop egoverse-handpose-viewer-local
docker rm -f egoverse-handpose-viewer-local
```

## Render Deployment

Render is the simplest public deployment target for this repo because it can:

- connect directly to GitHub,
- build the Dockerfile,
- run a long-lived Python service,
- provide HTTPS automatically,
- mount a persistent disk for cached EgoVerse episodes.

Steps:

1. Push this repo to GitHub.
2. In Render, create a new **Web Service**.
3. Connect the GitHub repo.
4. Select Docker runtime.
5. Use branch `main`.
6. Add a persistent disk mounted at:

   ```text
   /data/egoverse_viewer_cache
   ```

7. Add environment variables:

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

8. Deploy.

Render normally injects `PORT` automatically. Do not set `PORT` manually unless
Render asks for it.

## What Happens On Startup

`deploy/start.sh` does the following:

1. Sets default environment variables.
2. Creates the cache directory.
3. Runs EgoVerse `setup_secret.sh` if `~/.egoverse_env` does not exist.
4. Starts `scripts/egoverse_handpose_viewer.py`.

The AWS credentials above are the public EgoVerse bootstrap credentials from
the EgoVerse README. They are used to fetch the public read-only database and
R2 credentials at startup.

## Cache Behavior

The viewer does not download all EgoVerse data up front.

When a user opens a clip:

1. The app finds the episode in the public EgoVerse metadata database.
2. The app syncs that episode's Zarr directory into the cache.
3. The app renders frames from the local cached Zarr.

This means:

- first load for a new episode can be slow,
- repeated loads of the same episode are much faster,
- Render should use a persistent disk or the cache will reset on redeploy.

## Troubleshooting

If the homepage works but frames return `500`, check logs:

```bash
docker logs -f egoverse-handpose-viewer-local
```

If Docker build is slow on Apple Silicon, make sure you are using:

```bash
docker build --platform linux/amd64 -t egoverse-handpose-viewer .
```

If Render deploys but playback is slow, check:

- whether the cache disk is mounted,
- whether the current episode is still downloading,
- whether the latest GitHub commit has deployed,
- Render service CPU/memory limits.
