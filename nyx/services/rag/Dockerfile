# ==========================================================================================
# Author: Pablo González García.
# Created: 29/01/2026
# Last edited: 29/01/2026
# ==========================================================================================


# ==============================
# IMAGE
# ==============================

FROM python:3.13.9-slim
RUN useradd -m -u 1000 rag

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ .

RUN chown -R rag:rag /app

USER rag