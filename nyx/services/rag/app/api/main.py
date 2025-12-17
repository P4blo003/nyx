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
from api.routes.search import router as search_router
from api.routes.document import router as document_router
from api.middleware.logging import logging_middleware


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
app.include_router(search_router)
app.include_router(document_router)

# Includes middleware.
app.middleware("http")(logging_middleware)