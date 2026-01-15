# ==========================================================================================
# Author: Pablo González García.
# Created: 15/01/2025
# Last edited: 15/01/2025
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# External:
from fastapi import FastAPI

# Internal:
from api.lifespan import lifespan
from api.routes.triton import model as triton_model
from api.routes.triton import server as triton_server


# ==============================
# CONSTANTS
# ==============================

# FastAPI application.
_app:FastAPI = FastAPI(
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)


# ==============================
# MAIN
# ==============================

# Add API routes to FastAPI application.
_app.include_router(
    prefix="/triton",
    router=triton_model._router,
    tags=["models"]
)
_app.include_router(
    prefix="/triton",
    router=triton_server._router,
    tags=["servers"]
)