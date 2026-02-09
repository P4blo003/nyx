# ==========================================================================================
# Author: Pablo González García.
# Created: 15/01/2026
# Last edited: 29/01/2026
# ==========================================================================================


# ==============================
# BASE IMAGE
# ==============================

FROM python:3.11-slim

# Set environment variables for reproducible, non-interactive builds.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/home/ai/.local/bin:$PATH"


# ==============================
# SYSTEM DEPENDENCIES
# ==============================

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ca-certificates \
    git \
    && rm -rf /var/lib/apt/lists/*


# ==============================
# USER
# ==============================

# Create a non-root user for security.
RUN useradd -m -u 1000 ai
WORKDIR /app
USER ai


# ==============================
# DEPENDENCIES
# ==============================

# Install Python dependencies in a separate layer for caching.
COPY --chown=ai:ai requirements.txt .
RUN python -m pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt


# ==============================
# APP CODE
# ==============================
COPY --chown=ai:ai src/ ./src