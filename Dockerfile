# syntax=docker/dockerfile:1
FROM python:3.11-slim

WORKDIR /app

# System deps (optional: for building wheels)
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY src ./src
COPY app ./app

# Pre-train during build so image ships with a model (handy for demo)
RUN python -m src.train || true

# Expose and run
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
