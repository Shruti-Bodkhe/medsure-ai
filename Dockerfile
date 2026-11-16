FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install lightweight dependencies without heavy 500MB PyTorch for cloud container
COPY requirements.txt .
RUN grep -v -E "^(torch|sentence-transformers|sentence_transformers)" requirements.txt > req_cloud.txt && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r req_cloud.txt

COPY . .

RUN mkdir -p data/uploads data/audit data/index data/processed && chmod -R 777 /app

ENV MEDSURE_LOW_MEM=1
ENV TOKENIZERS_PARALLELISM=false
ENV PORT=8000

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
