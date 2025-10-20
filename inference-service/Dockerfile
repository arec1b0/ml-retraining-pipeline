# -----------------------------------------------------------------
# Lightweight Production Dockerfile for ML Inference Service
# -----------------------------------------------------------------
# This Dockerfile creates a minimal container image for serving
# ML models from MLflow Model Registry using FastAPI.
#
# Key Features:
# - Lightweight base image (Python 3.11 slim)
# - Non-root user for security
# - Health check for Kubernetes
# - Optimized layer caching
# -----------------------------------------------------------------

FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies (minimal set)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r mlops && useradd -r -g mlops -u 1001 mlops

# Set working directory
WORKDIR /app

# Copy requirements first (for better layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/

# Change ownership to non-root user
RUN chown -R mlops:mlops /app

# Switch to non-root user
USER mlops

# Expose port
EXPOSE 8000

# Health check for Kubernetes liveness/readiness probes
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

