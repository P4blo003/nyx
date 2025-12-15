# ==========================================================================================
# Author: Pablo González García.
# Created: 13/12/2025
# Last edited: 13/12/2025
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
from contextlib import asynccontextmanager
# External:
from fastapi import FastAPI
# Internal:
from api.routes.context import router as context_router


# ==============================
# FUNCTIONS
# ==============================

@asynccontextmanager
async def lifespan(app:FastAPI):
    """
    Lifespan context manager for startup and shutdown events.

    Yields:
        None.
    """

    # Returns.
    yield


# ==============================
# CONSTANTS
# ==============================

# Creates the app.
app:FastAPI = FastAPI(
    title="Rag-Service",
    lifespan=lifespan
)

# Includes the routes.
app.include_router(context_router)