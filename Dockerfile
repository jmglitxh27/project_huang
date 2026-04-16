# Full RNDA stack (FastAPI + ingest + embeddings). For cloud hosts: Render, Railway, Fly, etc.
# Listens on $PORT (defaults to 8000).

FROM python:3.11-slim-bookworm

WORKDIR /app

# PyMuPDF and scientific wheels often need a compiler for some platforms; slim has build-essential minimal set.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml /app/
COPY rnda /app/rnda

# Install package (pulls PyTorch, sentence-transformers, etc. — large image, expected).
RUN pip install --no-cache-dir -U pip setuptools wheel \
    && pip install --no-cache-dir .

# Optional: cache Hugging Face / sentence-transformers in a writable dir on serverless disks
ENV HF_HOME=/tmp/huggingface
ENV TRANSFORMERS_CACHE=/tmp/huggingface

EXPOSE 8000

# Cloud platforms set PORT; bind 0.0.0.0 so traffic reaches the container.
CMD ["sh", "-c", "exec uvicorn rnda.web.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
