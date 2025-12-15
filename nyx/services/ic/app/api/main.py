# ==========================================================================================
# Author: Pablo González García.
# Created: 11/12/2025
# Last edited: 11/12/2025
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
import signal
import asyncio
from contextlib import asynccontextmanager
# External:
from fastapi import FastAPI
# Internal:
from api.routes.classification import router as classifier_router


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
    title="IC Service",
    lifespan=lifespan
)

# Includes the routes.
app.include_router(classifier_router)