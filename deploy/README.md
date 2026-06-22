# Public EgoVerse Handpose Viewer Deploy

This is the simplest deployment path for the local viewer. The Docker image
installs only the viewer runtime dependencies, not the full EgoVerse training
stack.

## Build

```bash
docker build --platform linux/amd64 -t egoverse-handpose-viewer .
```

## Run Locally

Use the public EgoVerse AWS credentials from the EgoVerse README. The container
will call `egomimic/utils/aws/setup_secret.sh` on startup and write
`~/.egoverse_env` inside the container.

```bash
docker run --rm \
  -p 8770:8770 \
  -e AWS_ACCESS_KEY_ID='...' \
  -e AWS_SECRET_ACCESS_KEY='...' \
  -e AWS_DEFAULT_REGION='us-east-2' \
  -v egoverse-viewer-cache:/data/egoverse_viewer_cache \
  egoverse-handpose-viewer
```

Open:

```text
http://localhost:8770
```

## Deploy

Use any Docker host that supports a persistent volume:

- Render web service
- Fly.io app
- Railway service
- EC2 / GCP / Azure VM

Required environment variables:

```text
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
AWS_DEFAULT_REGION=us-east-2
REGION=us-east-2
HOST=0.0.0.0
PORT=<platform port, often injected automatically>
EGOVERSE_VIEWER_CACHE_DIR=/data/egoverse_viewer_cache
```

Mount a persistent volume at:

```text
/data/egoverse_viewer_cache
```

The app serves the same viewer UI and endpoints as local development.
