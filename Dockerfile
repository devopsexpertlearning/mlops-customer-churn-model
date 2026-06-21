# Stage 1: Build virtual environment
FROM python:3.12-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime image
FROM python:3.12-slim AS runner

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
COPY src/ /app/src
COPY configs/ /app/configs
COPY models/ /app/models
COPY setup.py /app/

ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir .

EXPOSE 8000

ENTRYPOINT ["uvicorn", "customer_churn.serve:app", "--host", "0.0.0.0", "--port", "8000"]
