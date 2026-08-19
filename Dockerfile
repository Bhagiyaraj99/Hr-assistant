FROM python:3.11-slim

WORKDIR /app

# System dependencies needed by chromadb's hnswlib (requires C++ compiler)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Use the Docker-specific requirements that pin CPU-only torch.
# This keeps the image lean (~1.75GB vs 8GB+ with CUDA torch).
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

COPY app/ ./app/
COPY data/ ./data/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]