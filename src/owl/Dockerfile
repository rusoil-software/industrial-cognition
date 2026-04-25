# Copyright (c), The Rusoil Software Development Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# author : Konstantin Ustiuzhanin
# date   : 2026-Apr-23
# ==============================================================================

# ============================================================================
# Stage 1: Builder
# ============================================================================
# Multi-stage build reduces final image size by excluding build dependencies.
# We compile wheels here, then copy only the compiled artifacts to the runtime.

FROM python:3.11-slim as builder

# Set environment variables for Python
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install build dependencies (gcc, etc. for compiling C extensions)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Create a virtual environment in the builder
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip, setuptools, wheel
RUN pip install --upgrade pip setuptools wheel

# Copy requirements and build wheels
COPY requirements/base.txt .
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /wheels -r base.txt

# ============================================================================
# Stage 2: Runtime
# ============================================================================
# Minimal runtime image with only what's needed to serve requests.

FROM python:3.11-slim-bullseye

# Set metadata labels for container registries and orchestration
LABEL maintainer="your-team@example.com" \
      description="OWL 2 Object Detection Inference Service" \
      version="1.0.0"
LABEL org.opencontainers.image.title="OWL Service"
LABEL org.opencontainers.image.description="OWL 2 Object Detection Inference Service"
LABEL org.opencontainers.image.url="https://github.com/rusoil-software/industrial-cognition/docs"
LABEL org.opencontainers.image.source="https://github.com/rusoil-software/industrial-cognition/"
LABEL org.opencontainers.image.vendor="ФГБОУ ВО УГНТУ"
LABEL org.opencontainers.image.authors="your-team@example.com"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app:$PYTHONPATH"

# Install only runtime dependencies (no build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security (principle of least privilege)
# Use a numeric UID/GID to avoid issues across different systems
RUN groupadd -r appuser && useradd -r -g appuser -u 1000 appuser

# Create application directory
WORKDIR /app

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Copy wheels from builder and install them
COPY --from=builder /wheels /wheels
RUN pip install --no-cache /wheels/* && rm -rf /wheels

# Copy application code
# Use .dockerignore to exclude unnecessary files (see below)
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser .env.example .env.example

# Create directory for models with proper permissions
RUN mkdir -p /app/models && chown -R appuser:appuser /app/models

# Create directory for logs with proper permissions
RUN mkdir -p /app/logs && chown -R appuser:appuser /app/logs

# Switch to non-root user
USER appuser

# Health check: ping the health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/inference/owl2/health || exit 1

# Expose port (for documentation; actual binding is in docker-compose)
EXPOSE 8000

# Run the application with gunicorn for production
# Using 4 workers; adjust based on CPU cores and expected load
CMD ["gunicorn", \
     "src.main:app", \
     "--workers", "4", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "--log-level", "info", \
     "--timeout", "120"]