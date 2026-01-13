# ==========================================================================================
# Author: Pablo González García.
# Created: 13/01/2026
# Last edited: 13/01/2026
# ==========================================================================================


# ==============================
# BASE IMAGE
# ==============================

FROM python:3.13.9-slim


# ==============================
# METADATA
# ==============================

LABEL maintainer="Pablo González García"
LABEL version="0.1"
LABEL description="FastAPI service with global tokenizer"


# ==============================
# WORKING DIRECTORY
# ==============================

WORKDIR /app


# ==============================
# DEPENDENCIES
# ==============================

COPY ./app /app
COPY ./model_repository /model_repository
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


# ==============================
# RUNTIME
# ==============================

EXPOSE 8000