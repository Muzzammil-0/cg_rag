FROM python:3.12-slim

# ---- Env ----
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/app/.cache/huggingface \
    TOKENIZERS_PARALLELISM=false \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860

# ---- System deps ----
# libgomp1  -> torch OpenMP runtime (required)
# build-essential -> some transitive deps compile from source
# curl      -> healthcheck
# git       -> some HF packages fetch via git during install
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---- CPU-only torch FIRST ----
# Prevents sentence-transformers from pulling the 2.5 GB CUDA build.
# Drops image size from ~5 GB to ~1.8 GB.
RUN pip install --upgrade pip wheel setuptools \
    && pip install --index-url https://download.pytorch.org/whl/cpu torch

# ---- Remaining Python deps ----
COPY requirements.txt .
RUN pip install --default-timeout=1000 --retries=10 -r requirements.txt

# ---- Pre-cache embedding + reranker models (own layer; survives rebuilds) ----
RUN python - <<'PY'
from sentence_transformers import SentenceTransformer, CrossEncoder
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
print('models cached')
PY

# ---- Project files (.dockerignore excludes data/vector_db, data/raw, .venv, .git) ----
COPY . .

# ---- Sanity: chunks file must be real JSON, not an LFS pointer ----
RUN test -s data/processed/chunks_v2.jsonl \
    && head -c 1 data/processed/chunks_v2.jsonl | grep -q '{' \
    || (echo "FATAL: chunks_v2.jsonl missing or is a git-lfs pointer. Run 'git lfs pull' on host." && exit 1)

EXPOSE 7860

# start-period 300s: safe ceiling for startup (build_if_needed hits fast path in ~1s
# with the mounted vector_db; full rebuild would take ~30min)
HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD curl -fsS http://localhost:7860/ > /dev/null || exit 1

CMD ["python", "app_gradio.py"]