FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

EXPOSE 8080 8501

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV SENTENCE_TRANSFORMERS_HOME=/app/.cache/sentence_transformers

VOLUME ["/app/.cache", "/app/data"]

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Default: run API server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]
