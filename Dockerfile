# Full RNDA stack (FastAPI + ingest + embeddings). Optimized for **layer caching**:
# - Big `pip install` reruns only when requirements-docker.txt changes.
# - App code changes only rerun the fast `pip install . --no-deps` step.
#
# Requires Docker BuildKit (default on Railway, Render, GitHub Actions).
# syntax=docker/dockerfile:1.4

FROM python:3.11-slim-bookworm

WORKDIR /app

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Most wheels are prebuilt; uncomment only if a package tries to compile from source.
# RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
#     && rm -rf /var/lib/apt/lists/*

# --- Layer A: heavy deps (cached until requirements-docker.txt changes) ---
COPY requirements-docker.txt .
# Railway requires `id=` on cache mounts: --mount=type=cache,id=<id>,target=...
RUN --mount=type=cache,id=pip-cache,target=/root/.cache/pip \
    pip install -U pip setuptools wheel \
    && pip install -r requirements-docker.txt

# --- Layer B: your package only (fast when you only edit code) ---
COPY pyproject.toml .
COPY rnda ./rnda
RUN pip install . --no-deps

# Hugging Face / sentence-transformers cache on ephemeral disk
ENV HF_HOME=/tmp/huggingface \
    TRANSFORMERS_CACHE=/tmp/huggingface

EXPOSE 8000

CMD ["sh", "-c", "exec uvicorn rnda.web.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
