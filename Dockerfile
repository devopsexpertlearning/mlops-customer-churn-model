# =============================================================================
# Stage 1: Builder
# =============================================================================
FROM python:3.12-slim AS builder

# Python settings
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Build dependencies (only for compilation)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Copy dependency file
COPY requirements-serve.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements-serve.txt && \
    pip uninstall -y nvidia-nccl-cu12


# Copy application source
COPY setup.py .
COPY src/ ./src

# Install application package (without reinstalling dependencies)
RUN pip install --no-deps .


# =============================================================================
# Stage 2: Runtime
# =============================================================================
FROM python:3.12-slim AS runner

# Python settings
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Install runtime packages only
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    tini \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd --gid 10001 appgroup && \
    useradd --uid 10001 --gid appgroup \
    --create-home \
    --shell /usr/sbin/nologin \
    appuser

# Copy virtual environment
COPY --from=builder /opt/venv /opt/venv

# Copy application artifacts
COPY --chown=appuser:appgroup configs/ ./configs
COPY --chown=appuser:appgroup models/ ./models


# Switch to non-root user
USER appuser

# Expose application port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s \
            --timeout=5s \
            --start-period=20s \
            --retries=3 \
CMD curl -fsS http://localhost:8000/health || exit 1

# Use tini for proper signal handling
ENTRYPOINT ["/usr/bin/tini", "--"]

# Production server
CMD ["gunicorn", "customer_churn.serve:app", "--worker-class", "uvicorn.workers.UvicornWorker", "--workers", "4", "--bind", "0.0.0.0:8000", "--timeout", "120", "--keep-alive", "5", "--access-logfile", "-", "--error-logfile", "-"]