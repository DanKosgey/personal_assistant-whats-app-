# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    APP_HOME=/app \
    ENV=production \
    PORT=8001

WORKDIR ${APP_HOME}

# System dependencies
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
      curl ca-certificates tzdata dumb-init && \
    rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -u 10001 -m appuser

# Install Python dependencies (cache-friendly)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Ensure entrypoint is executable and log dir exists
RUN chmod +x ./entrypoint.sh && \
    mkdir -p /var/log/whatsapp-agent && chown -R appuser:appuser /var/log/whatsapp-agent

EXPOSE ${PORT}

USER appuser

HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
  CMD curl -fsS http://127.0.0.1:${PORT}/healthz || exit 1

ENTRYPOINT ["/usr/bin/dumb-init", "--"]
CMD ["./entrypoint.sh"]
