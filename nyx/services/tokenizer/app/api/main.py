# ==========================================================================================
# Author: Pablo González García.
# Created: 13/01/2026
# Last edited: 13/01/2026
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# External:
from fastapi import FastAPI
# Internal:
from api.lifespan import lifespan
from api.router import token


# ==============================
# MAIN
# ==============================

# Initialize FastAPI application.
app:FastAPI = FastAPI(
    title="Tokenizer Service",
    version="1.0",
    docs_url="/swagger",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add endpoints.
app.include_router(token.router)