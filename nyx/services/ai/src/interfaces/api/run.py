# ==========================================================================================
# Author: Pablo González García.
# Created: 20/01/2025
# Last edited: 20/01/2025
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
from contextlib import asynccontextmanager

# External:
from fastapi import FastAPI

# Internal:
from interfaces.api.routes import models, inference
from application.services.bootstrap import bootstrap_ai_service



# ==============================
# FUNCTIONS
# ==============================

@asynccontextmanager
async def lifespan(app:FastAPI):
    """
    
    """

    # Initializes configuration and services.
    await bootstrap_ai_service()

    yield


# ==============================
# MAIN
# ==============================

# Initializes FastAPI application.
app:FastAPI = FastAPI(
    title="AI Service",
    description="AI model serving platform with Triton Backend",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Register application middleware.

# Register application routes.
app.include_router(
    prefix="/api",
    router=models.router
)
app.include_router(
    prefix="/api",
    router=inference.router
)