# ─── Stage 1: Builder ─────────────────────────────────────────────────────────
FROM python:3.14-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies into a prefix for easy copying
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --prefix=/install --no-cache-dir -r requirements.txt


# ─── Stage 2: Runtime ─────────────────────────────────────────────────────────
FROM python:3.14-slim AS runtime

LABEL maintainer="your-email@example.com"
LABEL version="2.0"
LABEL description="Skin Disease Detection Using Deep Learning"

# Security: run as non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app

# Runtime-only system libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source (matches new clean structure)
COPY --chown=appuser:appuser api.py         ./api.py
COPY --chown=appuser:appuser predict.py     ./predict.py
COPY --chown=appuser:appuser preprocess.py  ./preprocess.py
COPY --chown=appuser:appuser explainability.py ./explainability.py
COPY --chown=appuser:appuser logger.py      ./logger.py
COPY --chown=appuser:appuser dataset_loader.py ./dataset_loader.py
COPY --chown=appuser:appuser utils/         ./utils/
COPY --chown=appuser:appuser model/         ./model/
COPY --chown=appuser:appuser static/        ./static/

# Create temp dir for uploads and logs dir with correct permissions
RUN mkdir -p /tmp/skin-uploads logs && chown -R appuser:appuser /tmp/skin-uploads logs

# Switch to non-root user
USER appuser

# Environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000 \
    UPLOAD_DIR=/tmp/skin-uploads

EXPOSE 8000

# Health check — hits the /health endpoint every 30s
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Start FastAPI with Uvicorn
CMD ["sh", "-c", \
    "uvicorn api:app --host 0.0.0.0 --port ${PORT} --workers 2 --log-level info"]
