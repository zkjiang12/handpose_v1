FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    EGOVERSE_REPO=/opt/EgoVerse \
    EGOVERSE_VIEWER_CACHE_DIR=/data/egoverse_viewer_cache \
    HOST=0.0.0.0 \
    PORT=8770

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    git \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    unzip \
    && rm -rf /var/lib/apt/lists/*

RUN pip install uv awscli

RUN git clone --depth 1 https://github.com/GaTech-RL2/EgoVerse.git "$EGOVERSE_REPO"

WORKDIR /opt/EgoVerse
RUN uv pip install --system -r requirements.txt && \
    uv pip install --system -e .

WORKDIR /app
COPY . /app

RUN chmod +x /app/deploy/start.sh && \
    mkdir -p "$EGOVERSE_VIEWER_CACHE_DIR"

EXPOSE 8770

CMD ["/app/deploy/start.sh"]
