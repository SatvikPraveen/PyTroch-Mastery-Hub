# docker/Dockerfile
# Multi-stage production Dockerfile for ML training framework

# Build stage
FROM python:3.11-slim AS builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
COPY requirements-prod.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r requirements-prod.txt

# Production stage
FROM python:3.11-slim AS production

# Create non-root user
RUN groupadd -r mluser && useradd -r -g mluser mluser

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    libgomp1 \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs data models checkpoints results && \
    chown -R mluser:mluser /app

# Copy entrypoint script
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import torch; print('Health check passed')" || exit 1

# Switch to non-root user
USER mluser

# Set Python path
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Expose ports
EXPOSE 8000 8080

ENTRYPOINT ["/entrypoint.sh"]
CMD ["python", "-m", "training.train"]