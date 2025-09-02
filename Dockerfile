# ===================== Base =====================
FROM python:3.11-slim AS base

# Faster, cleaner Python, and headless plotting
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# System packages:
# - libpq5: required by psycopg2-binary at runtime
# - tini: proper PID 1 for clean shutdowns (SIGTERM/SIGINT)
# - ca-certificates/curl: HTTPS, health checks, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
      libpq5 tini ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# Create unprivileged user
ARG APP_USER=appuser
ARG APP_UID=10001
RUN useradd -m -u ${APP_UID} ${APP_USER}

# Workdir
WORKDIR /app

# Copy and install Python deps first for better layer caching
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# Copy the rest of the code
# (includes: part1_backtest.py, part2_pseudo_live.py, part3_*.py, etc.)
COPY . /app

# Ensure permissions for the non-root user
RUN chown -R ${APP_USER}:${APP_USER} /app

# Switch to non-root
USER ${APP_USER}

# Expose Streamlit (optional dashboard)
EXPOSE 8501

# Use tini as PID 1 so signals are handled properly
ENTRYPOINT ["/usr/bin/tini","--"]

# Default command can be overridden by docker-compose services:
# - backtest: ["python","-u","part1_backtest.py"]
# - live_trader: ["python","-u","part2_pseudo_live.py"]
# - data_guard: ["python","-u","part3_data_guard.py"]
# - order_listener: ["python","-u","part3_order_listener.py"]
# - risk_worker: ["python","-u","part3_risk_worker.py"]
CMD ["streamlit","run","app/dashboard/Home.py","--server.port=8501","--server.address=0.0.0.0"]