FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for PyMuPDF, FAISS, and compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install CPU-only PyTorch FIRST to keep image size & RAM under 512MB
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download intfloat/multilingual-e5-small into the container so boot is instant
ENV HF_HOME=/app/.cache/huggingface
RUN mkdir -p /app/.cache/huggingface && \
    python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-small')"

# Copy the MedSure AI codebase and pre-built index
COPY . .

# Ensure SQLite Vault, uploads, and audit logs have write permissions
RUN mkdir -p data/uploads data/audit data/index && chmod -R 777 /app

EXPOSE 8000

# Render injects $PORT dynamically; default to 8000 locally
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
